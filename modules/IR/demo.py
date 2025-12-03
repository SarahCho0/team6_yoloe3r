import math
import copy

import gradio
import os
import torch
import numpy as np
import functools
import trimesh
import copy
from PIL import Image
from scipy.spatial.transform import Rotation

from modules.pe3r.images import Images

from modules.dust3r.inference import inference
from modules.dust3r.image_pairs import make_pairs
from modules.dust3r.utils.image import load_images, rgb
from modules.dust3r.utils.device import to_numpy
from modules.dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from modules.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from copy import deepcopy
import cv2
from typing import Any, Dict, Generator,List
import matplotlib.pyplot as pl

from modules.mobilesamv2.utils.transforms import ResizeLongestSide


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

def mask_to_box(mask):
    if mask.sum() == 0:
        return np.array([0, 0, 0, 0])
    
    # Get the rows and columns where the mask is 1
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # Get top, bottom, left, right edges
    top = np.argmax(rows)
    bottom = len(rows) - 1 - np.argmax(np.flip(rows))
    left = np.argmax(cols)
    right = len(cols) - 1 - np.argmax(np.flip(cols))
    
    return np.array([left, top, right, bottom])

def box_xyxy_to_xywh(box_xyxy):
    box_xywh = deepcopy(box_xyxy)
    box_xywh[2] = box_xywh[2] - box_xywh[0]
    box_xywh[3] = box_xywh[3] - box_xywh[1]
    return box_xywh

def get_seg_img(mask, box, image):
    image = image.copy()
    x, y, w, h = box
    # image[mask == 0] = np.array([0, 0, 0], dtype=np.uint8)
    box_area = w * h
    mask_area = mask.sum()
    if 1 - (mask_area / box_area) < 0.2:
        image[mask == 0] = np.array([0, 0, 0], dtype=np.uint8)
    else:
        random_values = np.random.randint(0, 255, size=image.shape, dtype=np.uint8)
        image[mask == 0] = random_values[mask == 0]
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h) 
    pad = np.zeros((l,l,3), dtype=np.uint8) # 
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

def slerp(u1, u2, t):
    """
    Perform spherical linear interpolation (Slerp) between two unit vectors.
    
    Args:
    - u1 (torch.Tensor): First unit vector, shape (1024,)
    - u2 (torch.Tensor): Second unit vector, shape (1024,)
    - t (float): Interpolation parameter
    
    Returns:
    - torch.Tensor: Interpolated vector, shape (1024,)
    """
    # Compute the dot product
    dot_product = torch.sum(u1 * u2)
    
    # Ensure the dot product is within the valid range [-1, 1]
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Compute the angle between the vectors
    theta = torch.acos(dot_product)
    
    # Compute the coefficients for the interpolation
    sin_theta = torch.sin(theta)
    if sin_theta == 0:
        # Vectors are parallel, return a linear interpolation
        return u1 + t * (u2 - u1)
    
    s1 = torch.sin((1 - t) * theta) / sin_theta
    s2 = torch.sin(t * theta) / sin_theta
    
    # Perform the interpolation
    return s1 * u1 + s2 * u2

def slerp_multiple(vectors, t_values):
    """
    Perform spherical linear interpolation (Slerp) for multiple vectors.
    
    Args:
    - vectors (torch.Tensor): Tensor of vectors, shape (n, 1024)
    - a_values (torch.Tensor): Tensor of values corresponding to each vector, shape (n,)
    
    Returns:
    - torch.Tensor: Interpolated vector, shape (1024,)
    """
    n = vectors.shape[0]
    
    # Initialize the interpolated vector with the first vector
    interpolated_vector = vectors[0]
    
    # Perform Slerp iteratively
    for i in range(1, n):
        # Perform Slerp between the current interpolated vector and the next vector
        t = t_values[i] / (t_values[i] + t_values[i-1])
        interpolated_vector = slerp(interpolated_vector, vectors[i], t)
    
    return interpolated_vector

@torch.no_grad
def get_mask_from_img_sam1(mobilesamv2, yolov8, sam1_image, yolov8_image, original_size, input_size, transform):
    sam_mask=[]
    img_area = original_size[0] * original_size[1]

    obj_results = yolov8(yolov8_image,device='cuda',retina_masks=False,imgsz=1024,conf=0.25,iou=0.95,verbose=False)
    input_boxes1 = obj_results[0].boxes.xyxy
    input_boxes1 = input_boxes1.cpu().numpy()
    input_boxes1 = transform.apply_boxes(input_boxes1, original_size)
    input_boxes = torch.from_numpy(input_boxes1).cuda()
    
    # obj_results = yolov8(yolov8_image,device='cuda',retina_masks=False,imgsz=512,conf=0.25,iou=0.9,verbose=False)
    # input_boxes2 = obj_results[0].boxes.xyxy
    # input_boxes2 = input_boxes2.cpu().numpy()
    # input_boxes2 = transform.apply_boxes(input_boxes2, original_size)
    # input_boxes2 = torch.from_numpy(input_boxes2).cuda()

    # input_boxes = torch.cat((input_boxes1, input_boxes2), dim=0)

    input_image = mobilesamv2.preprocess(sam1_image)
    image_embedding = mobilesamv2.image_encoder(input_image)['last_hidden_state']

    image_embedding=torch.repeat_interleave(image_embedding, 320, dim=0)
    prompt_embedding=mobilesamv2.prompt_encoder.get_dense_pe()
    prompt_embedding=torch.repeat_interleave(prompt_embedding, 320, dim=0)
    for (boxes,) in batch_iterator(320, input_boxes):
        with torch.no_grad():
            image_embedding=image_embedding[0:boxes.shape[0],:,:,:]
            prompt_embedding=prompt_embedding[0:boxes.shape[0],:,:,:]
            sparse_embeddings, dense_embeddings = mobilesamv2.prompt_encoder(
                points=None,
                boxes=boxes,
                masks=None,)
            low_res_masks, _ = mobilesamv2.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=prompt_embedding,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                simple_type=True,
            )
            low_res_masks=mobilesamv2.postprocess_masks(low_res_masks, input_size, original_size)
            sam_mask_pre = (low_res_masks > mobilesamv2.mask_threshold)
            for mask in sam_mask_pre:
                if mask.sum() / img_area > 0.002:
                    sam_mask.append(mask.squeeze(1))
    sam_mask=torch.cat(sam_mask)
    sorted_sam_mask = sorted(sam_mask, key=(lambda x: x.sum()), reverse=True)
    keep = mask_nms(sorted_sam_mask)
    ret_mask = filter(sorted_sam_mask, keep)

    return ret_mask

# [demo.py 내부에 추가할 함수]

@torch.no_grad
def get_mask_from_yolo_seg(seg_model, image_np, conf=0.25):
    """
    yoloe-11l-seg.pt 모델을 사용하여 탐지와 마스크 생성을 한 번에 수행합니다.
    """
    # retina_masks=True: 마스크를 원본 이미지 해상도로 출력 (품질 향상)
    results = seg_model.predict(image_np, conf=conf, retina_masks=True, verbose=False)
    
    sam_mask = []
    
    # 마스크가 감지되었는지 확인
    if results[0].masks is not None:
        # data 속성: (N, H, W) 형태의 마스크 텐서
        masks_data = results[0].masks.data
        img_area = image_np.shape[0] * image_np.shape[1]

        for mask in masks_data:
            # 이진화 (Binary Mask)
            bin_mask = mask > 0.5
            
            # 너무 작은 객체(전체 화면의 0.2% 미만)는 노이즈로 간주하고 제외
            if bin_mask.sum() / img_area > 0.002:
                sam_mask.append(bin_mask)

    if len(sam_mask) == 0:
        return []

    # 리스트를 텐서로 변환
    sam_mask = torch.stack(sam_mask)
    
    # NMS 적용 (마스크 겹침 제거)
    sorted_sam_mask = sorted(sam_mask, key=(lambda x: x.sum()), reverse=True)
    keep = mask_nms(sorted_sam_mask)
    ret_mask = filter(sorted_sam_mask, keep)

    return ret_mask


@torch.no_grad
def get_cog_feats(images, pe3r):
    # SigLIP을 안 쓰므로 복잡한 SAM+Feature 추출 과정이 필요 없음
    # 하지만 파이프라인 호환성을 위해 빈 리스트와 0으로 채워진 텐서 반환
    
    np_images = images.np_images
    cog_seg_maps = []
    rev_cog_seg_maps = []
    
    # 더미 데이터 생성 (기존 포맷 유지)
    for i in range(len(np_images)):
        h, w = np_images[i].shape[:2]
        # 세그멘테이션 맵을 -1(배경)로 채움
        dummy_map = -np.ones((h, w), dtype=np.int64)
        cog_seg_maps.append(dummy_map)
        rev_cog_seg_maps.append(dummy_map)

    # 더미 Feature (N, 1024) - N=1 (배경만 있음)
    multi_view_clip_feats = torch.zeros((1, 1024))

    return cog_seg_maps, rev_cog_seg_maps, multi_view_clip_feats

def get_reconstructed_scene(outdir, pe3r, device, silent, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    if len(filelist) < 2:
        raise gradio.Error("Please input at least 2 images.")

    images = Images(filelist=filelist, device=device)
    
    # try:
    cog_seg_maps, rev_cog_seg_maps, cog_feats = get_cog_feats(images, pe3r)
    imgs = load_images(images, rev_cog_seg_maps, size=512, verbose=not silent)
    # except Exception as e:
    #     rev_cog_seg_maps = []
    #     for tmp_img in images.np_images:
    #         rev_seg_map = -np.ones(tmp_img.shape[:2], dtype=np.int64)
    #         rev_cog_seg_maps.append(rev_seg_map)
    #     cog_seg_maps = rev_cog_seg_maps
    #     cog_feats = torch.zeros((1, 1024))
    #     imgs = load_images(images, rev_cog_seg_maps, size=512, verbose=not silent)

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
    # if mode == GlobalAlignerMode.PointCloudOptimizer:
    loss = scene_1.compute_global_alignment(tune_flg=True, init='mst', niter=niter, schedule=schedule, lr=lr)

    try:
        import torchvision.transforms as tvf
        ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        for i in range(len(imgs)):
            # print(imgs[i]['img'].shape, scene.imgs[i].shape, ImgNorm(scene.imgs[i])[None])
            imgs[i]['img'] = ImgNorm(scene_1.imgs[i])[None]
        pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
        output = inference(pairs, pe3r.mast3r, device, batch_size=1, verbose=not silent)
        mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, cog_seg_maps, rev_cog_seg_maps, cog_feats, device=device, mode=mode, verbose=not silent)
        ori_imgs = scene.ori_imgs
        lr = 0.01
        # if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(tune_flg=False, init='mst', niter=niter, schedule=schedule, lr=lr)
    except Exception as e:
        scene = scene_1
        scene.imgs = ori_imgs
        scene.ori_imgs = ori_imgs
        print(e)


    outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size)

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    # confs = to_numpy([c for c in scene.conf_2])
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


import copy # 코드 상단에 추가 필요

def get_3D_object_from_scene(outdir, pe3r, silent, clsidx, threshold, scene, min_conf_thr, as_pointcloud, 
                 mask_sky, clean_depth, transparent_cams, cam_size, mask_dict):
    
    # -------------------------------------------------------------------
    # [수정 1] 원본 이미지 백업 로직 추가
    # -------------------------------------------------------------------
    # scene 객체에 'backup_imgs'가 없다면(첫 실행이라면), 현재 이미지를 원본으로 저장
    if not hasattr(scene, 'backup_imgs'):
        # 리스트 내부의 numpy 배열까지 안전하게 복사하기 위해 deepcopy 사용 권장
        # 만약 deepcopy가 너무 느리다면: scene.backup_imgs = [img.copy() for img in scene.ori_imgs]
        scene.backup_imgs = [img.copy() for img in scene.ori_imgs]
        print("DEBUG: Original images backed up.")


    # 1. YOLO-World 클래스 설정
    # search_classes = [text] 
    # pe3r.seg_model.set_classes(search_classes)

    # -------------------------------------------------------------------
    # [수정 2] 검색 대상 이미지를 'scene.ori_imgs'가 아닌 '백업본'에서 가져옴
    # -------------------------------------------------------------------
    # 항상 깨끗한 원본에서 검색을 시작함
    original_images = scene.backup_imgs 
    masked_images = []

    imgshape = original_images[0].shape[:2]

    for k, v in mask_dict.items():
        if k==clsidx:
            for mask in masks:
    

    # 3. 각 이미지에 대해 YOLO-World 추론 수행
    for k, masks in mask_dict.items():
        if k!=clsidx:
            pass
        else:
            for i, img in enumerate(original_images):
                # 이미지 포맷 보정
                img_input = img.copy()
                if img_input.dtype != np.uint8:
                    if img_input.max() <= 1.0:
                        img_input = (img_input * 255).astype(np.uint8)
                    else:
                        img_input = img_input.astype(np.uint8)
                
                # 마스크 합치기
                combined_mask = np.zeros(imgshape, dtype=bool)
                found = False

                if masks is not None:
                    for mask in masks:
                        if mask.shape != combined_mask.shape:
                            mask = cv2.resize(mask, (combined_mask.shape[1], combined_mask.shape[0]))
                        combined_mask = np.logical_or(combined_mask, mask > 0.5)
                        found = True
                
                # 4. 이미지 마스킹 처리
                if found:
                    masked_img = img.copy()
                    if img.dtype == np.uint8:
                        # 찾은 물체 외에는 어둡게 처리 (30)
                        masked_img[~combined_mask] = 30 
                    else:
                        masked_img[~combined_mask] = 0.1 
                    masked_images.append(masked_img)
                else:
                    # 못 찾았으면 전체를 어둡게
                    masked_images.append(img * 0.1)

    # 5. Scene의 이미지를 마스킹된 이미지로 교체 (뷰어용)
    # 원본(backup)은 건드리지 않고, 현재 보여주는 이미지(ori_imgs, imgs)만 교체
    scene.ori_imgs = masked_images
    scene.imgs = masked_images 

    # 6. GLB 모델 추출
    outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size)
    
    return outfile


def set_scenegraph_options(inputfiles, winsize, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize = max(1, math.ceil((num_files - 1) / 2))
    if scenegraph_type == "swin":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=True)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files - 1, step=1, visible=False)
    elif scenegraph_type == "oneref":
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files - 1, step=1, visible=True)
    else:
        winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files - 1, step=1, visible=False)
    return winsize, refid


def main_demo(tmpdirname, pe3r, device, server_name, server_port, silent=False):
#    scene, outfile, imgs = get_reconstructed_scene(
#         outdir=tmpdirname,  pe3r=pe3r, device=device, silent=silent,
#         filelist=['/home/hujie/pe3r/datasets/mipnerf360_ov/bonsai/black_chair/images/DSCF5590.png',
#                   '/home/hujie/pe3r/datasets/mipnerf360_ov/bonsai/black_chair/images/DSCF5602.png',
#                   '/home/hujie/pe3r/datasets/mipnerf360_ov/bonsai/black_chair/images/DSCF5609.png'],
#         schedule="linear", niter=300, min_conf_thr=3.0, as_pointcloud=False, mask_sky=True, clean_depth=True, transparent_cams=False, 
#         cam_size=0.05, scenegraph_type="complete", winsize=1, refid=0)
    
    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, pe3r, device, silent)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent)
    get_3D_object_from_scene_fun = functools.partial(get_3D_object_from_scene, tmpdirname, pe3r, silent)

    with gradio.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="PE3R Demo") as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML('<h2 style="text-align: center;">PE3R Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            with gradio.Row():
                schedule = gradio.Dropdown(["linear", "cosine"],
                                           value='linear', label="schedule", info="For global alignment!",
                                           visible=False)
                niter = gradio.Number(value=300, precision=0, minimum=0, maximum=5000,
                                      label="num_iterations", info="For global alignment!",
                                      visible=False)
                scenegraph_type = gradio.Dropdown([("complete: all possible image pairs", "complete"),
                                                   ("swin: sliding window", "swin"),
                                                   ("oneref: match one image with all", "oneref")],
                                                  value='complete', label="Scenegraph",
                                                  info="Define how to make pairs",
                                                  interactive=True,
                                                  visible=False)
                winsize = gradio.Slider(label="Scene Graph: Window Size", value=1,
                                        minimum=1, maximum=1, step=1, visible=False)
                refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0, maximum=0, step=1, visible=False)

            run_btn = gradio.Button("Reconstruct")

            with gradio.Row():
                # adjust the confidence threshold
                min_conf_thr = gradio.Slider(label="min_conf_thr", value=3.0, minimum=1.0, maximum=20, step=0.1, visible=False)
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.05, minimum=0.001, maximum=0.1, step=0.001, visible=False)
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                # two post process implemented
                mask_sky = gradio.Checkbox(value=False, label="Mask sky", visible=False)
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps", visible=False)
                transparent_cams = gradio.Checkbox(value=True, label="Transparent cameras")

            with gradio.Row():
                text_input = gradio.Textbox(label="Query Text")
                threshold = gradio.Slider(label="Threshold", value=0.85, minimum=0.0, maximum=1.0, step=0.01)

            find_btn = gradio.Button("Find")

            outmodel = gradio.Model3D()
            outgallery = gradio.Gallery(label='rgb,depth,confidence', columns=3, height="100%",
                                        visible=False)

            # events
            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, winsize, refid, scenegraph_type],
                                   outputs=[winsize, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, winsize, refid, scenegraph_type],
                              outputs=[winsize, refid])
            run_btn.click(fn=recon_fun,
                          inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                                  mask_sky, clean_depth, transparent_cams, cam_size,
                                  scenegraph_type, winsize, refid],
                          outputs=[scene, outmodel, outgallery])
            min_conf_thr.release(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size],
                                 outputs=outmodel)
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size],
                            outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size],
                                 outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size],
                            outputs=outmodel)
            clean_depth.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size],
                               outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                            clean_depth, transparent_cams, cam_size],
                                    outputs=outmodel)
            find_btn.click(fn=get_3D_object_from_scene_fun,
                             inputs=[text_input, threshold, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size],
                            outputs=outmodel)
    demo.launch(share=True, server_name=server_name, server_port=server_port)