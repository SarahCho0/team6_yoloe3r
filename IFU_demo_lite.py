"""
IFU_demo_lite.py — 경량 버전 (유진홈센터 적용용)
====================================================
LLM API 키 불필요. 다음 기능만 수행:
  1. 이미지 3장 입력 (Gradio UI)
  2. YOLOE 객체 탐지 + 세그멘테이션
  3. PE3R(DUSt3R+MASt3R+SAM2) 또는 VGGT로 3D 재구성
  4. 감지된 객체 목록 출력
  5. 3D 뷰어 (GLB)

제거된 기능:
  - LLM 분석 리포트 (report_client, google.genai)
  - 이미지 스타일 변경 (style_client)
  - 프롬프트 검수 (for_retry)
  - 가구 변경/추가/제거 생성 기능
====================================================
"""

import os
import sys
import math
import copy
import tempfile
import argparse
import builtins
import datetime
from collections import Counter
from copy import deepcopy

sys.path.append(os.path.abspath('./modules'))

import gradio as gr
import torch
import numpy as np
import functools
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
import cv2
import matplotlib.pyplot as pl

# 3D 재구성 모듈
from modules.pe3r.images import Images
from modules.pe3r.models import Models
from modules.dust3r.inference import inference
from modules.dust3r.image_pairs import make_pairs
from modules.dust3r.utils.image import load_images, rgb
from modules.dust3r.utils.device import to_numpy
from modules.dust3r.viz import add_scene_cam, CAM_COLORS, cat_meshes, pts3d_to_trimesh
from modules.dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# ── VGGT 선택적 로드 (없으면 PE3R 사용) ─────────────────────
try:
    from modules.vggt_reconstruction import load_vggt_model, reconstruct_with_vggt, vggt_result_to_glb
    _DEVICE_FOR_VGGT = "cuda" if torch.cuda.is_available() else "cpu"
    _vggt_model = load_vggt_model(device=_DEVICE_FOR_VGGT)
    USE_VGGT = _vggt_model is not None
except Exception as _e:
    USE_VGGT = False
    _vggt_model = None
    print(f"ℹ️ VGGT 비활성화 → PE3R 사용 ({_e})")

if USE_VGGT:
    print("✅ 재구성 백본: VGGT")
else:
    print("ℹ️ 재구성 백본: PE3R (DUSt3R + MASt3R)")


# ─────────────────────────────────────────────────────────────
# 헬퍼: 씬 → GLB 변환
# ─────────────────────────────────────────────────────────────

def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world,
                                  cam_size=0.05, cam_color=None,
                                  as_pointcloud=False, transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d     = to_numpy(pts3d)
    imgs      = to_numpy(imgs)
    focals    = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = [pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]) for i in range(len(imgs))]
        scene.add_geometry(trimesh.Trimesh(**cat_meshes(meshes)))

    for i, pose_c2w in enumerate(cams2world):
        color = cam_color[i] if isinstance(cam_color, list) else (cam_color or CAM_COLORS[i % len(CAM_COLORS)])
        add_scene_cam(scene, pose_c2w, color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    scene.apply_transform(np.linalg.inv(cams2world[0]))
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('x', 180, degrees=True).as_matrix()
    scene.apply_transform(rot)
    translate = np.eye(4)
    translate[2, 3] = -2.0
    scene.apply_transform(translate)

    outfile = os.path.join(outdir, 'scene.glb')
    if not silent:
        print(f'(GLB 저장: {outfile})')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False,
                             mask_sky=False, clean_depth=False, transparent_cams=False, cam_size=0.05):
    if scene is None:
        return None
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()
    rgbimg    = scene.ori_imgs
    focals    = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    pts3d     = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world,
                                         as_pointcloud=as_pointcloud,
                                         transparent_cams=transparent_cams,
                                         cam_size=cam_size, silent=silent)


# ─────────────────────────────────────────────────────────────
# 헬퍼: 마스크 처리
# ─────────────────────────────────────────────────────────────

def _mask_nms(masks, threshold=0.8):
    keep = []
    suppressed = np.zeros(len(masks), dtype=np.int64)
    for i in range(len(masks)):
        if suppressed[i]:
            continue
        keep.append(i)
        for j in range(i + 1, len(masks)):
            if suppressed[j]:
                continue
            inter = (masks[i] & masks[j]).sum()
            if min(inter / masks[i].sum(), inter / masks[j].sum()) > threshold:
                suppressed[j] = 1
    return keep


def _filter_masks(masks, keep):
    return [m for i, m in enumerate(masks) if i in keep]


def _mask_to_box(mask):
    if mask.sum() == 0:
        return np.array([0, 0, 0, 0])
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return np.array([
        np.argmax(cols),                                    # left
        np.argmax(rows),                                    # top
        len(cols) - 1 - np.argmax(np.flip(cols)),          # right
        len(rows) - 1 - np.argmax(np.flip(rows)),          # bottom
    ])


def _box_xyxy_to_xywh(box):
    b = deepcopy(box)
    b[2] -= b[0]
    b[3] -= b[1]
    return b


def _get_class_embedding(class_id, feature_dim=1024):
    g = torch.Generator().manual_seed(int(class_id) + 100)
    emb = torch.randn(feature_dim, generator=g)
    return emb / emb.norm(dim=-1, keepdim=True)


# ─────────────────────────────────────────────────────────────
# YOLOE 탐지
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def _get_mask_and_class_from_yolo(seg_model, image_np, original_size, conf=0.25):
    results = seg_model.predict(image_np, conf=conf, retina_masks=True, verbose=False)
    sam_masks, class_ids = [], []
    img_area = original_size[0] * original_size[1]

    if results[0].masks is not None:
        for mask, cls in zip(results[0].masks.data, results[0].boxes.cls):
            bin_mask = mask > 0.5
            if bin_mask.sum() / img_area > 0.002:
                sam_masks.append(bin_mask)
                class_ids.append(int(cls.item()))

    if not sam_masks:
        return [], []

    sam_masks  = torch.stack(sam_masks)
    class_ids  = torch.tensor(class_ids)
    order      = torch.argsort(sam_masks.sum(dim=(1, 2)), descending=True)
    sam_masks  = sam_masks[order]
    class_ids  = class_ids[order]
    keep       = _mask_nms(sam_masks)
    return _filter_masks(sam_masks, keep), [class_ids[i].item() for i in keep]


@torch.no_grad()
def _get_cog_feats(images, pe3r):
    """
    YOLOE + SAM2 다중 뷰 특징 추출.
    pe3r/demo.py의 get_cog_feats와 동일하되, obj_id_to_class_id도 함께 반환.
    """
    cog_seg_maps, rev_cog_seg_maps = [], []
    inference_state = pe3r.sam2.init_state(
        images=images.sam2_images,
        video_height=images.sam2_video_size[0],
        video_width=images.sam2_video_size[1],
    )
    mask_num = 0
    obj_id_to_class_id = {}

    masks, class_ids = _get_mask_and_class_from_yolo(
        pe3r.seg_model, images.np_images[0], images.np_images_size[0]
    )
    for i, mask in enumerate(masks):
        pe3r.sam2.add_new_mask(inference_state=inference_state,
                               frame_idx=0, obj_id=mask_num, mask=mask)
        obj_id_to_class_id[mask_num] = class_ids[i]
        mask_num += 1

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in pe3r.sam2.propagate_in_video(inference_state):
        sam2_masks = (out_mask_logits > 0.0).squeeze(1)
        video_segments[out_frame_idx] = {
            oid: sam2_masks[i].cpu().numpy()
            for i, oid in enumerate(out_obj_ids)
        }
        if out_frame_idx == 0:
            continue
        yolo_masks, yolo_class_ids = _get_mask_and_class_from_yolo(
            pe3r.seg_model, images.np_images[out_frame_idx], images.np_images_size[out_frame_idx]
        )
        for i, ym in enumerate(yolo_masks):
            overlap = any(
                min((ym & sm).sum() / ym.sum(), (ym & sm).sum() / sm.sum()) > 0.25
                for sm in sam2_masks
            )
            if not overlap:
                video_segments[out_frame_idx][mask_num] = ym.cpu().numpy()
                obj_id_to_class_id[mask_num] = yolo_class_ids[i]
                mask_num += 1

    multi_view_clip_feats = torch.zeros((mask_num + 1, 1024))
    for obj_id in range(mask_num):
        if obj_id in obj_id_to_class_id:
            multi_view_clip_feats[obj_id] = _get_class_embedding(obj_id_to_class_id[obj_id])

    for frame_idx in range(len(video_segments)):
        img = images.np_images[frame_idx]
        rev_seg = -np.ones(img.shape[:2], dtype=np.int64)
        for oid, mask in sorted(video_segments[frame_idx].items(),
                                key=lambda x: np.count_nonzero(x[1])):
            if mask.sum() > 0:
                rev_seg[mask] = oid
        rev_cog_seg_maps.append(rev_seg)

        seg = -np.ones(img.shape[:2], dtype=np.int64)
        for oid, mask in sorted(video_segments[frame_idx].items(),
                                key=lambda x: np.count_nonzero(x[1]), reverse=True):
            if mask.sum() == 0:
                continue
            box = np.int32(_box_xyxy_to_xywh(_mask_to_box(mask)))
            if box[2] > 0 and box[3] > 0:
                seg[mask] = oid
        cog_seg_maps.append(seg)

    return cog_seg_maps, rev_cog_seg_maps, multi_view_clip_feats, obj_id_to_class_id


# ─────────────────────────────────────────────────────────────
# 메인 재구성 함수
# ─────────────────────────────────────────────────────────────

def get_reconstructed_scene(outdir, pe3r, device, silent,
                             filelist, schedule, niter, min_conf_thr,
                             as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                             scenegraph_type, winsize, refid):
    """
    3D 재구성 실행.
    VGGT 활성화 시 VGGT → GLB 경로 사용, 비활성화 시 PE3R 경로 사용.

    Returns:
        (scene_or_None, glb_path, gallery_imgs, obj_id_to_class_id)
    """
    if len(filelist) < 2:
        raise gr.Error("이미지를 2장 이상 입력해주세요.")

    file_paths = [f.name if hasattr(f, 'name') else f for f in filelist]

    # ── VGGT 경로 ────────────────────────────────────────────
    if USE_VGGT and _vggt_model is not None:
        print("🔄 VGGT 재구성 중...")
        try:
            vggt_result = reconstruct_with_vggt(file_paths, _vggt_model, device=_DEVICE_FOR_VGGT)
            glb_path = vggt_result_to_glb(
                vggt_result, output_path=os.path.join(outdir, "vggt_scene.glb")
            )
            # YOLOE 탐지는 PE3R 경로와 동일하게 수행
            images = Images(filelist=filelist, device=device)
            _, _, _, obj_id_to_class_id = _get_cog_feats(images, pe3r)
            gallery_imgs = []
            for dm in vggt_result["depth_maps"]:
                cmap = pl.get_cmap('jet')
                dm_norm = dm / (dm.max() + 1e-8)
                gallery_imgs.append(rgb(cmap(dm_norm)))
            return None, glb_path, gallery_imgs, obj_id_to_class_id
        except Exception as e:
            print(f"⚠️ VGGT 실패, PE3R로 폴백: {e}")

    # ── PE3R 경로 ────────────────────────────────────────────
    images = Images(filelist=filelist, device=device)
    obj_id_to_class_id = {}

    try:
        cog_seg_maps, rev_cog_seg_maps, cog_feats, obj_id_to_class_id = _get_cog_feats(images, pe3r)
        imgs = load_images(images, rev_cog_seg_maps, size=512, verbose=not silent)
    except Exception as e:
        print(f"특징 추출 오류: {e}")
        rev_cog_seg_maps = [-np.ones(img.shape[:2], dtype=np.int64) for img in images.np_images]
        cog_seg_maps = rev_cog_seg_maps
        cog_feats = torch.zeros((1, 1024))
        imgs = load_images(images, rev_cog_seg_maps, size=512, verbose=not silent)

    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    sg = scenegraph_type
    if sg == "swin":
        sg = f"swin-{winsize}"
    elif sg == "oneref":
        sg = f"oneref-{refid}"

    pairs  = make_pairs(imgs, scene_graph=sg, prefilter=None, symmetrize=True)
    output = inference(pairs, pe3r.mast3r, device, batch_size=1, verbose=not silent)
    mode   = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer

    scene_1 = global_aligner(output, cog_seg_maps, rev_cog_seg_maps, cog_feats,
                              device=device, mode=mode, verbose=not silent)
    scene_1.compute_global_alignment(tune_flg=True, init='mst', niter=niter,
                                      schedule=schedule, lr=0.01)

    try:
        import torchvision.transforms as tvf
        ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5,)*3, (0.5,)*3)])
        for i in range(len(imgs)):
            imgs[i]['img'] = ImgNorm(scene_1.imgs[i])[None]
        pairs  = make_pairs(imgs, scene_graph=sg, prefilter=None, symmetrize=True)
        output = inference(pairs, pe3r.mast3r, device, batch_size=1, verbose=not silent)
        scene  = global_aligner(output, cog_seg_maps, rev_cog_seg_maps, cog_feats,
                                device=device, mode=mode, verbose=not silent)
        ori_imgs = scene.ori_imgs
        scene.compute_global_alignment(tune_flg=False, init='mst', niter=niter,
                                        schedule=schedule, lr=0.01)
    except Exception as e:
        scene = scene_1
        ori_imgs = scene_1.ori_imgs
        scene.imgs = ori_imgs
        scene.ori_imgs = ori_imgs
        print(f"Refinement 실패, 초기 scene 사용: {e}")

    glb_path = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud,
                                        mask_sky, clean_depth, transparent_cams, cam_size)

    depths    = to_numpy(scene.get_depthmaps())
    confs     = to_numpy(list(scene.im_conf))
    cmap      = pl.get_cmap('jet')
    dmax      = max(d.max() for d in depths) if depths else 1
    cmax      = max(d.max() for d in confs)  if confs  else 1
    depths    = [d / dmax for d in depths]
    confs     = [cmap(d / cmax) for d in confs]

    gallery_imgs = []
    for i in range(len(scene.imgs)):
        gallery_imgs.extend([scene.imgs[i], rgb(depths[i]), rgb(confs[i])])

    return scene, glb_path, gallery_imgs, obj_id_to_class_id


# ─────────────────────────────────────────────────────────────
# 객체 목록 포매팅
# ─────────────────────────────────────────────────────────────

def _format_detected_objects(obj_id_to_class_id: dict, seg_model) -> str:
    if not obj_id_to_class_id:
        return "감지된 객체 없음"
    names = getattr(seg_model, 'names', {})
    label_counts = Counter(
        names.get(int(cls_id), f"class_{cls_id}")
        for cls_id in obj_id_to_class_id.values()
    )
    total = sum(label_counts.values())
    lines = [f"감지된 객체 총 {total}개:"]
    for name, cnt in label_counts.most_common():
        lines.append(f"  • {name}: {cnt}개")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Gradio UI 설정
# ─────────────────────────────────────────────────────────────

def set_scenegraph_options(inputfiles, winsize, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles else 1
    max_win   = max(1, math.ceil((num_files - 1) / 2))
    if scenegraph_type == "swin":
        return gr.update(visible=True, value=max_win, maximum=max_win), gr.update(visible=False)
    elif scenegraph_type == "oneref":
        return gr.update(visible=False), gr.update(visible=True, maximum=num_files - 1)
    return gr.update(visible=False), gr.update(visible=False)


# ─────────────────────────────────────────────────────────────
# main_demo_lite
# ─────────────────────────────────────────────────────────────

def main_demo_lite(tmpdirname, pe3r, device, server_name, server_port, silent=False):
    recon_fn          = functools.partial(get_reconstructed_scene, tmpdirname, pe3r, device, silent)
    model_from_scene  = functools.partial(get_3D_model_from_scene, tmpdirname, silent)

    def _recon_wrapper(*args):
        scene_obj, glb_path, gallery_imgs, obj_dict = recon_fn(*args)
        obj_text = _format_detected_objects(obj_dict, pe3r.seg_model)
        return scene_obj, glb_path, gallery_imgs, obj_text

    backend_label = "VGGT" if USE_VGGT else "PE3R"

    with gr.Blocks(title="IF U Lite") as demo:
        gr.Markdown(f"## 🧊 IF U Lite — 3D 재구성 + 객체 감지 (`{backend_label}`)")
        gr.Markdown("API 키 불필요. 이미지를 업로드하고 재구성을 시작하세요.")

        scene_state = gr.State(None)

        with gr.Row():
            # ── 좌측: 입력 및 설정
            with gr.Column(scale=1, min_width=300):
                inputfiles = gr.File(file_count="multiple", label="입력 이미지 (2~3장)")

                with gr.Accordion("⚙️ 고급 설정", open=False):
                    schedule  = gr.Dropdown(["linear", "cosine"], value="linear", label="schedule")
                    niter     = gr.Number(value=300, precision=0, label="반복 횟수")
                    scenegraph_type = gr.Dropdown(
                        [("complete", "complete"), ("swin", "swin"), ("oneref", "oneref")],
                        value="complete", label="Scenegraph",
                    )
                    winsize   = gr.Slider(value=1, minimum=1, maximum=1, step=1, visible=False)
                    refid     = gr.Slider(value=0, minimum=0, maximum=0, step=1, visible=False)
                    min_conf_thr   = gr.Slider(label="min_conf_thr", value=3.0, minimum=1.0, maximum=20)
                    cam_size       = gr.Slider(label="cam_size", value=0.05, minimum=0.001, maximum=0.1)
                    as_pointcloud  = gr.Checkbox(value=True,  label="포인트 클라우드")
                    transparent_cams = gr.Checkbox(value=True, label="투명 카메라")
                    mask_sky    = gr.Checkbox(value=False, visible=False)
                    clean_depth = gr.Checkbox(value=True,  visible=False)

                run_btn = gr.Button("▶ 3D 재구성 시작", variant="primary")

                object_list = gr.Textbox(
                    label="감지된 객체 목록",
                    lines=12,
                    interactive=False,
                    placeholder="재구성 완료 후 감지된 객체가 표시됩니다.",
                )

            # ── 중앙: 3D 뷰어
            with gr.Column(scale=2):
                outmodel = gr.Model3D(
                    label=f"3D 재구성 결과 ({backend_label})",
                    interactive=True,
                    height="70vh",
                )

            # ── 우측: 깊이맵/갤러리
            with gr.Column(scale=1):
                outgallery = gr.Gallery(
                    label="RGB / 깊이맵 / 신뢰도",
                    columns=3,
                    height="auto",
                    object_fit="contain",
                )

        # ── 이벤트 연결
        run_btn.click(
            fn=_recon_wrapper,
            inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                    mask_sky, clean_depth, transparent_cams, cam_size,
                    scenegraph_type, winsize, refid],
            outputs=[scene_state, outmodel, outgallery, object_list],
        )

        scenegraph_type.change(set_scenegraph_options,
                               [inputfiles, winsize, refid, scenegraph_type], [winsize, refid])
        inputfiles.change(set_scenegraph_options,
                          [inputfiles, winsize, refid, scenegraph_type], [winsize, refid])

        # 슬라이더/체크박스 실시간 업데이트 (PE3R scene 있을 때만 동작)
        _update_inputs = [scene_state, min_conf_thr, as_pointcloud,
                          mask_sky, clean_depth, transparent_cams, cam_size]
        for elem in [min_conf_thr, cam_size, as_pointcloud, mask_sky, clean_depth, transparent_cams]:
            trigger = elem.release if isinstance(elem, gr.Slider) else elem.change
            trigger(fn=model_from_scene, inputs=_update_inputs, outputs=outmodel)

    demo.launch(share=True, server_name=server_name, server_port=server_port)


# ─────────────────────────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────────────────────────

def _set_print_with_timestamp(fmt="%Y-%m-%d %H:%M:%S"):
    _orig = builtins.print
    def _p(*args, **kwargs):
        _orig(f'[{datetime.datetime.now().strftime(fmt)}]', end=' ')
        _orig(*args, **kwargs)
    builtins.print = _p


def _get_args():
    parser = argparse.ArgumentParser(prog="IFU_demo_lite")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--local_network", action="store_true", default=False)
    g.add_argument("--server_name", type=str, default=None)
    parser.add_argument("--server_port", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tmp_dir", type=str, default=None)
    parser.add_argument("--silent", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = _get_args()
    _set_print_with_timestamp()

    if args.tmp_dir:
        os.makedirs(args.tmp_dir, exist_ok=True)
        tempfile.tempdir = args.tmp_dir

    server_name = args.server_name or ("0.0.0.0" if args.local_network else "127.0.0.1")

    pe3r = Models(device=args.device)

    with tempfile.TemporaryDirectory(suffix="_pe3r_lite") as tmp:
        if not args.silent:
            print(f"임시 디렉토리: {tmp}")
        main_demo_lite(tmp, pe3r, args.device, server_name, args.server_port, silent=args.silent)
