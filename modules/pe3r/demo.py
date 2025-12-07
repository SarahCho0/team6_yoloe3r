import math
import copy
import gradio as gr
import os
import torch
import numpy as np
import functools
import trimesh
import copy
from PIL import Image
from scipy.spatial.transform import Rotation
import requests
from io import BytesIO
import cv2
from typing import Any, Dict, Generator, List
import matplotlib.pyplot as pl
import glob
import json

# 모듈 경로가 사용자 환경에 맞게 설정되어 있다고 가정합니다.
from modules.pe3r.images import Images
from modules.dust3r.inference import inference
from modules.dust3r.image_pairs import make_pairs
from modules.dust3r.utils.image import load_images, rgb
from modules.dust3r.utils.device import to_numpy
from modules.dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from modules.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from copy import deepcopy

from modules.mobilesamv2.utils.transforms import ResizeLongestSide
from modules.llm_final_api.main_report import main_report
from modules.llm_final_api.main_new_looks import main_new_looks
from modules.llm_final_api.main_modify_looks import main_modify_looks

from modules.IR.listup import listup
from modules.IR.track_crop import crop


def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.ori_imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)

def mask_nms(masks, threshold=0.8):
    keep = []
    mask_num = len(masks)
    suppressed = np.zeros((mask_num), dtype=np.int64)
    for i in range(mask_num):
        if suppressed[i] == 1:
            continue
        keep.append(i)
        for j in range(i + 1, mask_num):
            if suppressed[j] == 1:
                continue
            intersection = (masks[i] & masks[j]).sum()
            if min(intersection / masks[i].sum(), intersection / masks[j].sum()) > threshold:
                suppressed[j] = 1
    return keep

def filter(masks, keep):
    ret = []
    for i, m in enumerate(masks):
        if i in keep: ret.append(m)
    return ret

def get_mask_from_yolo_seg(seg_model, image_np, conf=0.25):
    results = seg_model.predict(image_np, conf=conf, retina_masks=True, verbose=False)
    sam_mask = []
    if results[0].masks is not None:
        masks_data = results[0].masks.data
        img_area = image_np.shape[0] * image_np.shape[1]
        for mask in masks_data:
            bin_mask = mask > 0.5
            if bin_mask.sum() / img_area > 0.002:
                sam_mask.append(bin_mask)

    if len(sam_mask) == 0:
        return []
    sam_mask = torch.stack(sam_mask)
    sorted_sam_mask = sorted(sam_mask, key=(lambda x: x.sum()), reverse=True)
    keep = mask_nms(sorted_sam_mask)
    ret_mask = filter(sorted_sam_mask, keep)
    return ret_mask

@torch.no_grad
def get_cog_feats(images, pe3r):
    np_images = images.np_images
    cog_seg_maps = []
    rev_cog_seg_maps = []
    for i in range(len(np_images)):
        h, w = np_images[i].shape[:2]
        dummy_map = -np.ones((h, w), dtype=np.int64)
        cog_seg_maps.append(dummy_map)
        rev_cog_seg_maps.append(dummy_map)
    multi_view_clip_feats = torch.zeros((1, 1024))
    return cog_seg_maps, rev_cog_seg_maps, multi_view_clip_feats

def get_reconstructed_scene(outdir, pe3r, device, silent, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid):
    if len(filelist) < 2:
        raise gr.Error("Please input at least 2 images.")

    images = Images(filelist=filelist, device=device)
    
    cog_seg_maps, rev_cog_seg_maps, cog_feats = get_cog_feats(images, pe3r)
    imgs = load_images(images, rev_cog_seg_maps, size=512, verbose=not silent)

    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, pe3r.mast3r, device, batch_size=1, verbose=not silent)
    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene_1 = global_aligner(output, cog_seg_maps, rev_cog_seg_maps, cog_feats, device=device, mode=mode, verbose=not silent)
    lr = 0.01
    loss = scene_1.compute_global_alignment(tune_flg=True, init='mst', niter=niter, schedule=schedule, lr=lr)

    try:
        import torchvision.transforms as tvf
        ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        for i in range(len(imgs)):
            imgs[i]['img'] = ImgNorm(scene_1.imgs[i])[None]
        pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
        output = inference(pairs, pe3r.mast3r, device, batch_size=1, verbose=not silent)
        mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, cog_seg_maps, rev_cog_seg_maps, cog_feats, device=device, mode=mode, verbose=not silent)
        ori_imgs = scene.ori_imgs
        lr = 0.01
        loss = scene.compute_global_alignment(tune_flg=False, init='mst', niter=niter, schedule=schedule, lr=lr)
    except Exception as e:
        scene = scene_1
        scene.imgs = ori_imgs
        scene.ori_imgs = ori_imgs
        print(e)

    outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size)

    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d / confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        imgs.append(rgbimg[i])
        imgs.append(rgb(depths[i]))
        imgs.append(rgb(confs[i]))

    return scene, outfile, imgs


def get_3D_object_from_scene(outdir, pe3r, silent, text, threshold, scene, min_conf_thr, as_pointcloud, 
                 mask_sky, clean_depth, transparent_cams, cam_size):
    
    if not hasattr(scene, 'backup_imgs'):
        scene.backup_imgs = [img.copy() for img in scene.ori_imgs]
        print("DEBUG: Original images backed up.")

    print(f"Searching for: '{text}' using YOLO-World...")

    search_classes = [text] 
    pe3r.seg_model.set_classes(search_classes)

    original_images = scene.backup_imgs 
    masked_images = []

    for i, img in enumerate(original_images):
        img_input = img.copy()
        if img_input.dtype != np.uint8:
            if img_input.max() <= 1.0:
                img_input = (img_input * 255).astype(np.uint8)
            else:
                img_input = img_input.astype(np.uint8)

        conf_thr = 0.05 
        results = pe3r.seg_model.predict(img_input, conf=conf_thr, retina_masks=True, verbose=False)
        
        combined_mask = np.zeros(img.shape[:2], dtype=bool)
        found = False

        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            for mask in masks:
                if mask.shape != combined_mask.shape:
                    mask = cv2.resize(mask, (combined_mask.shape[1], combined_mask.shape[0]))
                combined_mask = np.logical_or(combined_mask, mask > 0.5)
                found = True
        
        if found:
            masked_img = img.copy()
            if img.dtype == np.uint8:
                masked_img[~combined_mask] = 30 
            else:
                masked_img[~combined_mask] = 0.1 
            masked_images.append(masked_img)
        else:
            masked_images.append(img * 0.1)

    scene.ori_imgs = masked_images
    scene.imgs = masked_images 

    outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size)
    
    return outfile

def highlight_selected_object(
    scene, mask_list, object_id_list,
    min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
    evt: gr.SelectData,
    outdir=None
): 
    if scene is None or not mask_list:
        print("⚠️ Scene or mask_list is empty.")
        return None

    if evt is None or not isinstance(evt, gr.SelectData):
        print(f"⚠️ Error: evt is {type(evt)}. Gradio failed to pass SelectData.")
        return None

    selected_index = evt.index
    print(f"🖱️ Clicked index: {selected_index}")

    if selected_index >= len(object_id_list):
        print("Error: Index out of range")
        return None
        
    target_obj_id = object_id_list[selected_index] 
    print(f"🎯 [Highlight] Target Object: {target_obj_id}")

    if not hasattr(scene, 'backup_imgs'):
        scene.backup_imgs = [img.copy() for img in scene.ori_imgs]

    masked_images = []
    original_images = scene.backup_imgs
    
    for i, img in enumerate(original_images):
        current_frame_masks = mask_list[i]
        target_mask = None
        if target_obj_id in current_frame_masks:
            target_mask = current_frame_masks[target_obj_id]
        
        img_h, img_w = img.shape[:2]
        processed_img = img.copy()
        
        if target_mask is not None:
            if target_mask.shape[:2] != (img_h, img_w):
                target_mask = cv2.resize(target_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            if processed_img.dtype == np.uint8:
                processed_img[~target_mask] = 30
            else:
                processed_img[~target_mask] = 0.1
        else:
            if processed_img.dtype == np.uint8:
                processed_img[:] = 30
            else:
                processed_img[:] = 0.1
                
        masked_images.append(processed_img)

    scene.ori_imgs = masked_images
    scene.imgs = masked_images

    if outdir is None:
        print("Error: outdir is None")
        return None

    outfile = get_3D_model_from_scene(outdir, False, scene, min_conf_thr, as_pointcloud, mask_sky, 
                                      clean_depth, transparent_cams, cam_size)
    
    return outfile

def set_scenegraph_options(inputfiles, winsize, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize = max(1, math.ceil((num_files - 1) / 2))
    if scenegraph_type == "swin":
        winsize = gr.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=True)
        refid = gr.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files - 1, step=1, visible=False)
    elif scenegraph_type == "oneref":
        winsize = gr.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gr.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files - 1, step=1, visible=True)
    else:
        winsize = gr.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gr.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files - 1, step=1, visible=False)
    return winsize, refid


def main_demo(tmpdirname, pe3r, device, server_name, server_port, silent=False):
    
    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, pe3r, device, silent)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent)
    get_3D_object_from_scene_fun = functools.partial(get_3D_object_from_scene, tmpdirname, pe3r, silent)

    # [NEW] 초기 생성 시에만 위/아래 모델을 동시에 채워주는 래퍼 함수
    def initial_recon_wrapper(*args):
        # 기존 recon_fun 실행
        scene_obj, model_path, gallery_imgs = recon_fun(*args)
        # 중요: model_path를 두 번 리턴 (위쪽 모델용, 아래쪽 원본 모델용)
        return scene_obj, model_path, model_path, gallery_imgs

    def save_style_json(selected_style):
        data = {"selected_style": selected_style}
        try:
            with open("modules/llm_final_api/style_choice.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"💾 [Saved] style_choice.json: {data}")
        except Exception as e:
            print(f"❌ [Error] 스타일 저장 실패: {e}")

    def save_user_choice_json(use_add, use_remove, use_change):
        data = {
            "use_add": use_add,
            "use_remove": use_remove,
            "use_change": use_change
        }
        try:
            with open("modules/llm_final_api/user_choice.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"💾 [Saved] user_choice.json: {data}")
        except Exception as e:
            print(f"❌ [Error] 유저 선택 저장 실패: {e}")

    def read_report_file(filename="report_analysis_result.txt"):
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                return f"파일 읽기 오류: {str(e)}"
        return "⚠️ 분석 결과 파일이 생성되지 않았습니다."

    def run_analysis_and_show_ui(input_files):
        image_paths = []
        if input_files:
            for f in input_files:
                path = f.name if hasattr(f, 'name') else f
                image_paths.append(path)
        
        if main_report:
            try:
                print(f"📊 [Info] 이미지 분석 시작 ({len(image_paths)}장)...")
                main_report(image_paths) 
            except Exception as e:
                print(f"❌ [Error] 분석 모듈 실행 실패: {e}")
                return f"### 분석 오류 발생\n{str(e)}", gr.update(visible=False), gr.update(visible=False)
        else:
            return "### 분석 모듈 로드 실패\nmain_report.py를 찾을 수 없습니다.", gr.update(visible=False), gr.update(visible=False)

        report_text = read_report_file("report_analysis_result.txt")
        return report_text, gr.update(visible=True, open=True), gr.update(visible=True, open=True)
    
    def generate_and_load_new_images():
        if main_new_looks:
            try:
                print("🎨 [Info] 새로운 룩 생성 시작...")
                main_new_looks()
            except Exception as e:
                print(f"❌ [Error] 이미지 생성 실패: {e}")
        else:
            print("⚠️ Error: main_new_looks 모듈이 로드되지 않았습니다.")

        output_dir = os.path.join(os.getcwd(), "apioutput")
        if not os.path.exists(output_dir):
            print(f"⚠️ Warning: {output_dir} 폴더가 존재하지 않습니다.")
            return []

        files = glob.glob(os.path.join(output_dir, "*.[pP][nN][gG]")) + \
                glob.glob(os.path.join(output_dir, "*.[jJ][pP][gG]")) + \
                glob.glob(os.path.join(output_dir, "*.[jJ][pP][eE][gG]"))
        
        files.sort(key=os.path.getmtime, reverse=True)
        selected_files = files[:3]
        print(f"📂 [Info] 로드된 파일: {selected_files}")
        return selected_files

    def generate_and_load_modified_images():
        if main_modify_looks:
            try:
                print("🎨 [Info] 새로운 룩 생성 시작...")
                main_modify_looks()
            except Exception as e:
                print(f"❌ [Error] 이미지 생성 실패: {e}")
        else:
            print("⚠️ Error: main_modify_looks 모듈이 로드되지 않았습니다.")

        output_dir = os.path.join(os.getcwd(), "apioutput")
        if not os.path.exists(output_dir):
            print(f"⚠️ Warning: {output_dir} 폴더가 존재하지 않습니다.")
            return []

        files = glob.glob(os.path.join(output_dir, "*.[pP][nN][gG]")) + \
                glob.glob(os.path.join(output_dir, "*.[jJ][pP][gG]")) + \
                glob.glob(os.path.join(output_dir, "*.[jJ][pP][eE][gG]"))
        
        files.sort(key=os.path.getmtime, reverse=True)
        selected_files = files[:3]
        print(f"📂 [Info] 로드된 파일: {selected_files}")
        return selected_files
    
    def backup_original_scene(scene, input_files):
        saved_paths = []
        if input_files:
            for f in input_files:
                path = f.name if hasattr(f, 'name') else f
                saved_paths.append(path)
        
        print(f"💾 [Backup] Scene과 파일 {len(saved_paths)}개가 원본으로 백업되었습니다.")
        return scene, saved_paths
    
    def backup_original_report(report_text):
        print("💾 [Backup] 분석 리포트 텍스트 백업 완료")
        return report_text

    def restore_original_scene(orig_scene, orig_inputs, orig_report, min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size):
        if orig_scene is None:
            return gr.update(), gr.update(), gr.update(), "⚠️ 저장된 원본이 없습니다."
        
        if hasattr(orig_scene, 'backup_imgs'):
            print("🔄 [Restore] 마스킹된 이미지를 원본으로 복구 중...")
            orig_scene.ori_imgs = [img.copy() for img in orig_scene.backup_imgs]
            orig_scene.imgs = [img.copy() for img in orig_scene.backup_imgs]
            
        restored_model_path = model_from_scene_fun(
            orig_scene, min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size
        )
        restored_report = orig_report if orig_report else "🔄 원본 리포트가 없습니다."

        print("↩️ [Restore] 원본 Scene 및 리포트 되돌리기 완료")
        return orig_scene, restored_model_path, orig_inputs, restored_report

    def run_and_display(input_files):
        image_paths = []
        if input_files:
            for f in input_files:
                path = f.name if hasattr(f, 'name') else f
                image_paths.append(path)
        else:
            print('no input')

        url_dict, mask_list, ordered_ids = listup(input_files)
        
        gallery_data = []
        for folder_id, url in url_dict.items():
            try:
                response = requests.get(url[0])
                image = Image.open(BytesIO(response.content))
                caption = f"Model Name : {url[1]}"
                gallery_data.append((image, caption))
            except Exception as e:
                print(f"Error loading image from {url[0]}: {e}")
                continue
                
        return gallery_data, mask_list, ordered_ids
    
    def on_gallery_select(scene, mask_data, id_list, 
                                      conf, pc, sky, clean, trans, size, 
                                      evt: gr.SelectData):
                    return highlight_selected_object(
                        scene, mask_data, id_list, 
                        conf, pc, sky, clean, trans, size, 
                        evt, 
                        outdir=tmpdirname 
                    )

    # -------------------------------------------------------------------------

    with gr.Blocks(title="IF U Demo", fill_width=True) as demo:
        scene = gr.State(None)

        original_scene = gr.State(None)       
        original_inputfiles = gr.State(None)
        original_report_text = gr.State(None) 
        mask_data_state = gr.State([])
        object_id_list_state = gr.State([])

        gr.Markdown("##🛋️ IF U Demo")

        with gr.Row():
            # --- 좌측 패널 (설정) ---
            with gr.Column(scale=1, min_width=320):
                inputfiles = gr.File(file_count="multiple", label="Input Images")
                
                with gr.Accordion("⚙️ Settings", open=False):
                    schedule = gr.Dropdown(["linear", "cosine"], value='linear', label="schedule")
                    niter = gr.Number(value=300, precision=0, label="num_iterations")
                    scenegraph_type = gr.Dropdown(
                        [("complete", "complete"), ("swin", "swin"), ("oneref", "oneref")],
                        value='complete', label="Scenegraph"
                    )
                    winsize = gr.Slider(value=1, minimum=1, maximum=1, step=1, visible=False)
                    refid = gr.Slider(value=0, minimum=0, maximum=0, step=1, visible=False)
                    min_conf_thr = gr.Slider(label="min_conf_thr", value=3.0, minimum=1.0, maximum=20)
                    cam_size = gr.Slider(label="cam_size", value=0.05, minimum=0.001, maximum=0.1)
                    as_pointcloud = gr.Checkbox(value=True, label="As pointcloud")
                    transparent_cams = gr.Checkbox(value=True, label="Transparent cameras")
                    mask_sky = gr.Checkbox(value=False, visible=False)
                    clean_depth = gr.Checkbox(value=True, visible=False)

                run_btn = gr.Button("3D로 변환", variant="primary", elem_classes=["primary-btn"])
                IR_btn = gr.Button("배치된 가구 제품명 찾기", variant="primary", elem_classes=["primary-btn"])
                
                revert_btn = gr.Button("↩️ 원본 되돌리기", variant="secondary")

                with gr.Accordion("🎨 분석리포트 적용", open=True, visible=False) as analysis_accordion:
                    add = gr.Checkbox(value=False, label="가구 배치 제안 반영해보기")
                    delete = gr.Checkbox(value=False, label="가구 제거 제안 반영해보기")
                    change = gr.Checkbox(value=False, label="가구 변경 제안 반영해보기")
                    run_suggested_change_btn= gr.Button("결과 생성", variant="primary")
                with gr.Accordion("방 분위기 바꿔보기", open=False, visible=False) as analysis_accordion1:
                    style = gr.Dropdown(["AI 추천","미니멀리즘","맥시멀리즘"], label="style")
                    run_style_change_btn = gr.Button("결과 생성", variant="primary")

            # --- 우측 패널 (3D 뷰어 2개 배치) ---
            with gr.Column(scale=5):
                # [위쪽] 현재 상태 (변경됨) - 화면 높이의 45%
                outmodel = gr.Model3D(
                    label="Current Model (Modified Look)", 
                    interactive=True,
                    height="65vh",
                    camera_position=(0.0, 0.0, 1.5)
                )
                
                # [아래쪽] 원본 상태 (고정됨) - 화면 높이의 45%
                orig_model_display = gr.Model3D(
                    label="Original Model (Reference)", 
                    interactive=True,
                    height="25vh",
                    camera_position=(0.0, 0.0, 2.0)
                )
                
                analysis_output = gr.Markdown(
                    value="여기에 공간 분석 결과가 표시됩니다.",
                    label="공간 분석 리포트",
                    elem_classes=["report-box"]
                )
                outgallery = gr.Gallery(visible=False)

            with gr.Column(scale=1):
                result_gallery = gr.Gallery(
                    label="Detected Objects", 
                    columns=1,            
                    height="auto",        
                    object_fit="contain"  
                )
                
        # ---------------------------------------------------------------------
        # [이벤트 연결]
        # ---------------------------------------------------------------------

        IR_btn.click(
            fn=run_and_display, 
            inputs=[inputfiles], 
            outputs=[result_gallery, mask_data_state, object_id_list_state]
        )

        result_gallery.select(
                    fn=on_gallery_select,
                    inputs=[
                        scene,                
                        mask_data_state,      
                        object_id_list_state, 
                        min_conf_thr,         
                        as_pointcloud,        
                        mask_sky,             
                        clean_depth,          
                        transparent_cams,     
                        cam_size              
                    ],
                    outputs=outmodel
                )

        # 1. [초기 생성] run_btn -> 위(outmodel)와 아래(orig_model_display) 둘 다 업데이트
        #    이곳에서만 분석 리포트를 생성합니다.
        recon_event = run_btn.click(
            fn=initial_recon_wrapper, 
            inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                    mask_sky, clean_depth, transparent_cams, cam_size,
                    scenegraph_type, winsize, refid],
            outputs=[scene, outmodel, orig_model_display, outgallery] 
        )
        
        recon_event.success(
            fn=backup_original_scene,
            inputs=[scene, inputfiles],
            outputs=[original_scene, original_inputfiles]
        )

        analysis_step = recon_event.then(
            fn=lambda: "⏳ 3D 생성이 완료되었습니다. 공간 분위기를 분석 중입니다...",
            inputs=None,
            outputs=analysis_output
        )

        finish_analysis_step = analysis_step.then(
            fn=run_analysis_and_show_ui,
            inputs=[inputfiles],
            outputs=[analysis_output, analysis_accordion, analysis_accordion1]
        )

        finish_analysis_step.success(
            fn=backup_original_report,
            inputs=[analysis_output],
            outputs=[original_report_text]
        )

        # ---------------------------------------------------------------------
        # [수정/스타일 변경 이벤트] -> 오직 outmodel(위쪽)만 업데이트
        # **여기서는 분석 리포트 생성을 다시 하지 않습니다.**
        # ---------------------------------------------------------------------
        
        # 스타일 변경
        suggestion_event = run_style_change_btn.click(
            fn=generate_and_load_new_images,
            inputs=None,
            outputs=inputfiles
        )

        suggestion_recon_event = suggestion_event.then(
            fn=recon_fun,
            inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                    mask_sky, clean_depth, transparent_cams, cam_size,
                    scenegraph_type, winsize, refid],
            outputs=[scene, outmodel, outgallery] # orig_model_display 제외
        )

        # (분석 리포트 재실행 부분 삭제됨)

        # 가구 변경
        modify_event = run_suggested_change_btn.click(
            fn=generate_and_load_modified_images,
            inputs=None,
            outputs=inputfiles
        )

        modify_recon_event = modify_event.then(
            fn=recon_fun,
            inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                    mask_sky, clean_depth, transparent_cams, cam_size,
                    scenegraph_type, winsize, refid],
            outputs=[scene, outmodel, outgallery] # orig_model_display 제외
        )

        # (분석 리포트 재실행 부분 삭제됨)

        # ---------------------------------------------------------------------
        # [되돌리기] -> outmodel을 원본과 같게 복구
        # ---------------------------------------------------------------------
        revert_btn.click(
            fn=restore_original_scene,
            inputs=[original_scene, original_inputfiles, original_report_text, 
                    min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size],
            outputs=[scene, outmodel, inputfiles, analysis_output]
        )

        #----------------------------------------------------------
        # 설정값 변경
        # -------------------------------------------------------
        style.change(fn=save_style_json, inputs=[style], outputs=None)

        checkbox_inputs = [add, delete, change]
        add.change(fn=save_user_choice_json, inputs=checkbox_inputs, outputs=None)
        delete.change(fn=save_user_choice_json, inputs=checkbox_inputs, outputs=None)
        change.change(fn=save_user_choice_json, inputs=checkbox_inputs, outputs=None)

        scenegraph_type.change(set_scenegraph_options, [inputfiles, winsize, refid, scenegraph_type], [winsize, refid])
        inputfiles.change(set_scenegraph_options, [inputfiles, winsize, refid, scenegraph_type], [winsize, refid])
        
        update_inputs = [scene, min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size]
        min_conf_thr.release(fn=model_from_scene_fun, inputs=update_inputs, outputs=outmodel)
        cam_size.change(fn=model_from_scene_fun, inputs=update_inputs, outputs=outmodel)
        as_pointcloud.change(fn=model_from_scene_fun, inputs=update_inputs, outputs=outmodel)
        mask_sky.change(fn=model_from_scene_fun, inputs=update_inputs, outputs=outmodel)
        clean_depth.change(fn=model_from_scene_fun, inputs=update_inputs, outputs=outmodel)
        transparent_cams.change(model_from_scene_fun, inputs=update_inputs, outputs=outmodel)

    demo.launch(share=True, server_name=server_name, server_port=server_port)
