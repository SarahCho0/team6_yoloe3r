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
import requests
from io import BytesIO

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
import glob

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

def get_3D_object_from_scene(outdir, pe3r, silent, text, threshold, scene, min_conf_thr, as_pointcloud, 
                 mask_sky, clean_depth, transparent_cams, cam_size):
    
    # -------------------------------------------------------------------
    # [수정 1] 원본 이미지 백업 로직 추가
    # -------------------------------------------------------------------
    # scene 객체에 'backup_imgs'가 없다면(첫 실행이라면), 현재 이미지를 원본으로 저장
    if not hasattr(scene, 'backup_imgs'):
        # 리스트 내부의 numpy 배열까지 안전하게 복사하기 위해 deepcopy 사용 권장
        # 만약 deepcopy가 너무 느리다면: scene.backup_imgs = [img.copy() for img in scene.ori_imgs]
        scene.backup_imgs = [img.copy() for img in scene.ori_imgs]
        print("DEBUG: Original images backed up.")

    print(f"Searching for: '{text}' using YOLO-World...")

    # 1. YOLO-World 클래스 설정
    search_classes = [text] 
    pe3r.seg_model.set_classes(search_classes)

    # -------------------------------------------------------------------
    # [수정 2] 검색 대상 이미지를 'scene.ori_imgs'가 아닌 '백업본'에서 가져옴
    # -------------------------------------------------------------------
    # 항상 깨끗한 원본에서 검색을 시작함
    original_images = scene.backup_imgs 
    masked_images = []

    # 3. 각 이미지에 대해 YOLO-World 추론 수행
    for i, img in enumerate(original_images):
        # 이미지 포맷 보정
        img_input = img.copy()
        if img_input.dtype != np.uint8:
            if img_input.max() <= 1.0:
                img_input = (img_input * 255).astype(np.uint8)
            else:
                img_input = img_input.astype(np.uint8)

        # 추론 (Confidence Threshold 설정)
        conf_thr = 0.05 
        
        # YOLO 추론
        results = pe3r.seg_model.predict(img_input, conf=conf_thr, retina_masks=True, verbose=False)
        
        # 마스크 합치기
        combined_mask = np.zeros(img.shape[:2], dtype=bool)
        found = False

        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
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

def highlight_selected_object(
    scene, mask_list, object_id_list,  # 입력 데이터
    min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, # 설정값
    evt: gradio.SelectData, # 클릭 이벤트 데이터 (입력값 뒤에 배치)
    outdir=None # 경로 (마지막에 키워드로 받음)
): 
    """
    갤러리 선택 시 호출되는 함수
    """
    # 1. 예외 처리: 데이터가 없거나 이벤트가 잘못 들어온 경우
    if scene is None or not mask_list:
        print("⚠️ Scene or mask_list is empty.")
        return None

    if evt is None or not isinstance(evt, gradio.SelectData):
        print(f"⚠️ Error: evt is {type(evt)}. Gradio failed to pass SelectData.")
        return None

    # 2. 선택된 인덱스 가져오기
    selected_index = evt.index
    print(f"🖱️ Clicked index: {selected_index}")

    if selected_index >= len(object_id_list):
        print("Error: Index out of range")
        return None
        
    target_obj_id = object_id_list[selected_index] 
    print(f"🎯 [Highlight] Target Object: {target_obj_id}")

    # 3. Scene 백업 확인 (원본 보존)
    if not hasattr(scene, 'backup_imgs'):
        scene.backup_imgs = [img.copy() for img in scene.ori_imgs]

    # 4. 마스크 적용 로직
    masked_images = []
    original_images = scene.backup_imgs
    
    for i, img in enumerate(original_images):
        current_frame_masks = mask_list[i]
        
        target_mask = None
        if target_obj_id in current_frame_masks:
            target_mask = current_frame_masks[target_obj_id]
        
        img_h, img_w = img.shape[:2]
        processed_img = img.copy()
        
        # 마스크 처리 (선택된 객체 외에는 어둡게)
        if target_mask is not None:
            # 크기 보정
            if target_mask.shape[:2] != (img_h, img_w):
                target_mask = cv2.resize(target_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            
            if processed_img.dtype == np.uint8:
                processed_img[~target_mask] = 30
            else:
                processed_img[~target_mask] = 0.1
        else:
            # 객체가 없는 프레임은 전체 어둡게
            if processed_img.dtype == np.uint8:
                processed_img[:] = 30
            else:
                processed_img[:] = 0.1
                
        masked_images.append(processed_img)

    # 5. Scene 이미지 교체
    scene.ori_imgs = masked_images
    scene.imgs = masked_images

    # 6. 3D 모델 재생성
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


import gradio as gr
import functools
import os
import sys
import json
# [참고] 외부 함수들은 그대로 유지
# get_reconstructed_scene, get_3D_model_from_scene, get_3D_object_from_scene, set_scenegraph_options

def main_demo(tmpdirname, pe3r, device, server_name, server_port, silent=False):
    
    # 1. 3D 모델 생성 로직
    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, pe3r, device, silent)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent)
    get_3D_object_from_scene_fun = functools.partial(get_3D_object_from_scene, tmpdirname, pe3r, silent)

    def save_style_json(selected_style):
        """스타일 선택 시 style_choice.json 저장"""
        data = {"selected_style": selected_style}
        try:
            with open("modules/llm_final_api/style_choice.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"💾 [Saved] style_choice.json: {data}")
        except Exception as e:
            print(f"❌ [Error] 스타일 저장 실패: {e}")

    def save_user_choice_json(use_add, use_remove, use_change):
        """체크박스 변경 시 user_choice.json 저장"""
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

    # -------------------------------------------------------------------------
    # [수정됨] 분석 및 UI 업데이트 전담 함수
    # -------------------------------------------------------------------------
    def read_report_file(filename="report_analysis_result.txt"):
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                return f"파일 읽기 오류: {str(e)}"
        return "⚠️ 분석 결과 파일이 생성되지 않았습니다."

    def run_analysis_and_show_ui(input_files):
        """
        분석을 수행하고 -> 결과 텍스트와 -> 아코디언을 보이게 하는 명령을 함께 반환
        """
        #1. 이미지 경로 추출
        image_paths = []
        if input_files:
            for f in input_files:
                path = f.name if hasattr(f, 'name') else f
                image_paths.append(path)
        
        # 2. 분석 실행
        if main_report:
            try:
                print(f"📊 [Info] 이미지 분석 시작 ({len(image_paths)}장)...")
                # main_report(image_paths) # 함수명이 report라고 가정 (코드에 맞게 수정 필요)
                # 혹시 함수명이 run_analysis라면 아래 주석 해제
                main_report(image_paths) 
            except Exception as e:
                print(f"❌ [Error] 분석 모듈 실행 실패: {e}")
                # 에러가 나도 아코디언은 띄우지 않거나, 에러 로그를 리턴
                return f"### 분석 오류 발생\n{str(e)}", gr.update(visible=False)
        else:
            return "### 분석 모듈 로드 실패\nmain_report.py를 찾을 수 없습니다.", gr.update(visible=False)

        # 3. 결과 반환 (텍스트, 아코디언 보이기 Update)
        report_text = read_report_file("report_analysis_result.txt")
        return report_text, gr.update(visible=True, open=True), gr.update(visible=True, open=True)
    
    def generate_and_load_new_images():
        """
        1. main_new_looks 실행
        2. apioutput_style 폴더의 이미지 파일들을 inputfiles로 반환
        """
        # 1. 생성 모듈 실행
        if main_new_looks:
            try:
                print("🎨 [Info] 새로운 룩 생성 시작...")
                main_new_looks()
            except Exception as e:
                print(f"❌ [Error] 이미지 생성 실패: {e}")
                # 에러 발생 시 빈 리스트 반환보다는 None을 반환하거나 에러 처리
        else:
            print("⚠️ Error: main_new_looks 모듈이 로드되지 않았습니다.")

        # 2. apioutput 폴더에서 파일 가져오기
        output_dir = os.path.join(os.getcwd(), "apioutput")
        if not os.path.exists(output_dir):
            print(f"⚠️ Warning: {output_dir} 폴더가 존재하지 않습니다.")
            return []

        # png, jpg, jpeg 파일 검색
        files = glob.glob(os.path.join(output_dir, "*.[pP][nN][gG]")) + \
                glob.glob(os.path.join(output_dir, "*.[jJ][pP][gG]")) + \
                glob.glob(os.path.join(output_dir, "*.[jJ][pP][eE][gG]"))
        
        # 최신 파일 3개만 가져오거나 전체를 가져옴 (요청사항: 3장의 이미지)
        # 생성 순서대로 정렬 (수정 시간이 최신인 것)
        files.sort(key=os.path.getmtime, reverse=True)
        
        selected_files = files[:3]
        print(f"📂 [Info] 로드된 파일: {selected_files}")
        
        return selected_files
    def generate_and_load_modified_images():
        """
        1. main_modify_looks 실행
        2. apioutput_modify 폴더의 이미지 파일들을 inputfiles로 반환
        """
        # 1. 생성 모듈 실행
        if main_modify_looks:
            try:
                print("🎨 [Info] 새로운 룩 생성 시작...")
                main_modify_looks()
            except Exception as e:
                print(f"❌ [Error] 이미지 생성 실패: {e}")
                # 에러 발생 시 빈 리스트 반환보다는 None을 반환하거나 에러 처리
        else:
            print("⚠️ Error: main_modify_looks 모듈이 로드되지 않았습니다.")

        # 2. apioutput 폴더에서 파일 가져오기
        output_dir = os.path.join(os.getcwd(), "apioutput")
        if not os.path.exists(output_dir):
            print(f"⚠️ Warning: {output_dir} 폴더가 존재하지 않습니다.")
            return []

        # png, jpg, jpeg 파일 검색
        files = glob.glob(os.path.join(output_dir, "*.[pP][nN][gG]")) + \
                glob.glob(os.path.join(output_dir, "*.[jJ][pP][gG]")) + \
                glob.glob(os.path.join(output_dir, "*.[jJ][pP][eE][gG]"))
        
        # 최신 파일 3개만 가져오거나 전체를 가져옴 (요청사항: 3장의 이미지)
        # 생성 순서대로 정렬 (수정 시간이 최신인 것)
        files.sort(key=os.path.getmtime, reverse=True)
        
        selected_files = files[:3]
        print(f"📂 [Info] 로드된 파일: {selected_files}")
        
        return selected_files
    
    # -------------------------------------------------------------------------
    # [되돌리기(Revert) 관련 함수 - NEW]
    # -------------------------------------------------------------------------
    # 1. 초기 생성 시 백업 저장
    def backup_original_scene(scene, input_files):
        """Reconstruct 버튼 클릭 시 생성된 scene과 입력 파일을 백업"""
        
        # [수정된 부분] input_files 안에 있는 객체가 파일 래퍼인지 문자열인지 확인 후 '경로 문자열'만 저장
        saved_paths = []
        if input_files:
            for f in input_files:
                # f가 Gradio 파일 객체(_TemporaryFileWrapper)라면 .name을 가져오고,
                # 이미 문자열(경로)라면 그대로 사용
                path = f.name if hasattr(f, 'name') else f
                saved_paths.append(path)
        
        print(f"💾 [Backup] Scene과 파일 {len(saved_paths)}개가 원본으로 백업되었습니다.")
        
        # 수정된 경로 리스트(saved_paths)를 저장해야 나중에 에러가 안 납니다.
        return scene, saved_paths
    
    def backup_original_report(report_text):
        """생성된 분석 리포트 텍스트를 백업"""
        print("💾 [Backup] 분석 리포트 텍스트 백업 완료")
        return report_text

    # 2. 되돌리기 버튼 클릭 시 복구
    def restore_original_scene(orig_scene, orig_inputs, orig_report, min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size):
        """백업된 scene, 파일, 리포트를 복구하고 3D 모델 뷰어 업데이트"""
        if orig_scene is None:
            return gr.update(), gr.update(), gr.update(), "⚠️ 저장된 원본이 없습니다."
        
        # -------------------------------------------------------------------
        # [핵심 수정] 이미지 원상복구 로직 추가
        # -------------------------------------------------------------------
        # get_3D_object_from_scene에서 'backup_imgs'를 만들어 두었으므로,
        # 되돌리기 시 이 백업본을 다시 메인 이미지(ori_imgs, imgs)로 덮어씌워야 합니다.
        if hasattr(orig_scene, 'backup_imgs'):
            print("🔄 [Restore] 마스킹된 이미지를 원본으로 복구 중...")
            # 리스트 컴프리헨션으로 안전하게 복사
            orig_scene.ori_imgs = [img.copy() for img in orig_scene.backup_imgs]
            orig_scene.imgs = [img.copy() for img in orig_scene.backup_imgs]
            
            # (선택 사항) 복구 후 백업본을 삭제하고 싶다면 아래 주석 해제
            # del orig_scene.backup_imgs 
            # 하지만 검색을 또 할 수도 있으니 놔두는 것을 추천합니다.
        # -------------------------------------------------------------------

        # 저장된 scene 객체로부터 다시 3D 모델 파일 생성
        # (이제 orig_scene의 이미지가 밝은 원본으로 돌아왔으므로 밝은 모델이 생성됨)
        restored_model_path = model_from_scene_fun(
            orig_scene, min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size
        )
        
        # 리포트 복구 (없을 경우 기본 메시지)
        restored_report = orig_report if orig_report else "🔄 원본 리포트가 없습니다."

        print("↩️ [Restore] 원본 Scene 및 리포트 되돌리기 완료")
        
        # 순서: Scene, 3D모델경로, 입력파일, 분석리포트텍스트
        return orig_scene, restored_model_path, orig_inputs, restored_report
    #-----------------------------------------
    # IR
    #-----------------------------------------
    def run_and_display(input_files):
        """
        listup()을 실행하고 결과를 Gradio 갤러리 형식으로 변환합니다.
        """

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
                
                # (이미지 객체, 캡션) 튜플 형태로 리스트에 추가
                caption = f"Model Name : {url[1]}"
                gallery_data.append((image, caption))
                
            except Exception as e:
                print(f"Error loading image from {url[0]}: {e}")
                continue
                
        return gallery_data, mask_list, ordered_ids
    
    def on_gallery_select(scene, mask_data, id_list, 
                                      conf, pc, sky, clean, trans, size, 
                                      evt: gr.SelectData): # evt를 명시적으로 선언
                    
                    return highlight_selected_object(
                        scene, mask_data, id_list, 
                        conf, pc, sky, clean, trans, size, 
                        evt, 
                        outdir=tmpdirname  # main_demo의 변수 tmpdirname 사용
                    )

    # -------------------------------------------------------------------------

    with gr.Blocks(title="PE3R Demo", fill_width=True) as demo:
        scene = gr.State(None)

        # [NEW] 원본 복구를 위한 상태 변수
        original_scene = gr.State(None)       
        original_inputfiles = gr.State(None)
        original_report_text = gr.State(None) # 리포트 백업용
        mask_data_state = gr.State([])
        object_id_list_state = gr.State([])

        gr.Markdown("## 🧊 IF Demo")

        with gr.Row():
            # --- 좌측 패널 ---
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

                # with gradio.Row():
                #     text_input = gradio.Textbox(label="Query Text")
                #     threshold = gradio.Slider(label="Threshold", value=0.85, minimum=0.0, maximum=1.0, step=0.01)
                # find_btn = gradio.Button("Find")
                
                # [수정됨] 초기에는 보이지 않도록 visible=False 설정
                # 변수명(analysis_accordion)을 할당해야 나중에 업데이트 가능
                with gr.Accordion("🎨 분석리포트 적용", open=True, visible=False) as analysis_accordion:
                    add = gr.Checkbox(value=False, label="가구 배치 제안 반영해보기")
                    delete = gr.Checkbox(value=False, label="가구 제거 제안 반영해보기")
                    change = gr.Checkbox(value=False, label="가구 변경 제안 반영해보기")
                    run_suggested_change_btn= gr.Button("결과 생성", variant="primary")
                with gr.Accordion("방 분위기 바꿔보기", open=False, visible=False) as analysis_accordion1:
                    style = gr.Dropdown(["AI 추천","미니멀리즘","맥시멀리즘"], label="style")
                    run_style_change_btn = gr.Button("결과 생성", variant="primary")

            # --- 우측 패널 ---
            with gr.Column(scale=2):
                outmodel = gr.Model3D(label="3D Reconstruction Result", interactive=True)
                
                analysis_output = gr.Markdown(
                    value="여기에 공간 분석 결과가 표시됩니다.",
                    label="공간 분석 리포트",
                    elem_classes=["report-box"]
                )
                outgallery = gr.Gallery(visible=False)
            with gr.Column():
                
                # columns=1로 설정하면 이미지가 세로로 한 줄씩 나옵니다.
                # object_fit="contain"은 이미지가 잘리지 않고 전체가 보이게 합니다.
                result_gallery = gr.Gallery(
                    label="Detected Objects", 
                    columns=1,            # [핵심] 세로 정렬을 위해 1열로 설정
                    height="auto",        # 높이 자동 조절
                    object_fit="contain"  # 이미지 비율 유지
                )
                
                # 버튼 클릭 시 함수 실행 -> 갤러리에 출력
        IR_btn.click(
            fn=run_and_display, 
            inputs=[inputfiles], 
            outputs=[result_gallery, mask_data_state, object_id_list_state] # State에 저장
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

        # ---------------------------------------------------------------------
        # [이벤트 흐름 1: 기본 Reconstruct 버튼 (원본 생성)]
        # ---------------------------------------------------------------------
        # 1. 3D 생성
        recon_event = run_btn.click(
            fn=recon_fun,
            inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                    mask_sky, clean_depth, transparent_cams, cam_size,
                    scenegraph_type, winsize, refid],
            outputs=[scene, outmodel, outgallery]
        )
        
        # 2. [Backup Scene] 생성 성공 시 Scene과 파일 백업
        recon_event.success(
            fn=backup_original_scene,
            inputs=[scene, inputfiles],
            outputs=[original_scene, original_inputfiles]
        )

        # 3. 로딩 메시지
        analysis_step = recon_event.then(
            fn=lambda: "⏳ 3D 생성이 완료되었습니다. 공간 분위기를 분석 중입니다...",
            inputs=None,
            outputs=analysis_output
        )

        # 4. 분석 결과 표시
        finish_analysis_step = analysis_step.then(
            fn=run_analysis_and_show_ui,
            inputs=[inputfiles],
            outputs=[analysis_output, analysis_accordion, analysis_accordion1]
        )

        # 5. [Backup Report] 분석이 끝나고 UI에 표시된 후, 그 텍스트를 백업
        finish_analysis_step.success(
            fn=backup_original_report,
            inputs=[analysis_output], # 화면에 출력된 텍스트를 가져옴
            outputs=[original_report_text]
        )

        # ---------------------------------------------------------------------
        # [이벤트 흐름 2: 되돌리기 (Revert) 버튼]
        # ---------------------------------------------------------------------
        revert_btn.click(
            fn=restore_original_scene,
            # 원본 데이터(Scene, 파일, 리포트) + 시각화 옵션들을 입력으로 받음
            inputs=[original_scene, original_inputfiles, original_report_text, 
                    min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size],
            # 현재 상태 업데이트
            outputs=[scene, outmodel, inputfiles, analysis_output]
        )

        #------------------------------------------------
        # 스타일변경
        #------------------------------------------------

        suggestion_event = run_style_change_btn.click(
            fn=generate_and_load_new_images,
            inputs=None,
            outputs=inputfiles  # apioutput의 이미지들이 여기로 들어감
        )

        # 2. 업데이트된 InputFiles로 Reconstruct 자동 실행 (run_btn 로직 복제)
        # 주의: inputs에 [inputfiles, ...] 를 넣으면 갱신된 파일이 들어갑니다.
        suggestion_recon_event = suggestion_event.then(
            fn=recon_fun,
            inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                    mask_sky, clean_depth, transparent_cams, cam_size,
                    scenegraph_type, winsize, refid],
            outputs=[scene, outmodel, outgallery]
        )

        # 3. 분석 메시지 표시
        suggestion_analysis_step = suggestion_recon_event.then(
            fn=lambda: "⏳ 새로운 디자인을 3D로 변환 중입니다. 다시 분석 중...",
            inputs=None,
            outputs=analysis_output
        )

        # 4. 분석 결과 다시 표시
        suggestion_analysis_step.then(
            fn=run_analysis_and_show_ui,
            inputs=[inputfiles],
            outputs=[analysis_output, analysis_accordion, analysis_accordion1]
        )

        #------------------------------------------------------------
        # modify
        # ----------------------------------------------------------


        modify_event = run_suggested_change_btn.click(
            fn=generate_and_load_modified_images,
            inputs=None,
            outputs=inputfiles  # apioutput의 이미지들이 여기로 들어감
        )

        # 2. 업데이트된 InputFiles로 Reconstruct 자동 실행 (run_btn 로직 복제)
        # 주의: inputs에 [inputfiles, ...] 를 넣으면 갱신된 파일이 들어갑니다.
        modify_recon_event = modify_event.then(
            fn=recon_fun,
            inputs=[inputfiles, schedule, niter, min_conf_thr, as_pointcloud,
                    mask_sky, clean_depth, transparent_cams, cam_size,
                    scenegraph_type, winsize, refid],
            outputs=[scene, outmodel, outgallery]
        )

        # 3. 분석 메시지 표시
        modify_analysis_step = modify_recon_event.then(
            fn=lambda: "⏳ 새로운 디자인을 3D로 변환 중입니다. 다시 분석 중...",
            inputs=None,
            outputs=analysis_output
        )

        # 4. 분석 결과 다시 표시
        modify_analysis_step.then(
            fn=run_analysis_and_show_ui,
            inputs=[inputfiles],
            outputs=[analysis_output, analysis_accordion, analysis_accordion1]
        )

        #----------------------------------------------------------
        # 이외 설정값 변경
        # -------------------------------------------------------
        style.change(fn=save_style_json, inputs=[style], outputs=None)

        checkbox_inputs = [add, delete, change]
        add.change(fn=save_user_choice_json, inputs=checkbox_inputs, outputs=None)
        delete.change(fn=save_user_choice_json, inputs=checkbox_inputs, outputs=None)
        change.change(fn=save_user_choice_json, inputs=checkbox_inputs, outputs=None)



        # --- 나머지 이벤트 연결 (기존 유지) ---
        scenegraph_type.change(set_scenegraph_options, [inputfiles, winsize, refid, scenegraph_type], [winsize, refid])
        inputfiles.change(set_scenegraph_options, [inputfiles, winsize, refid, scenegraph_type], [winsize, refid])
        
        update_inputs = [scene, min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size]
        min_conf_thr.release(fn=model_from_scene_fun, inputs=update_inputs, outputs=outmodel)
        cam_size.change(fn=model_from_scene_fun, inputs=update_inputs, outputs=outmodel)
        as_pointcloud.change(fn=model_from_scene_fun, inputs=update_inputs, outputs=outmodel)
        mask_sky.change(fn=model_from_scene_fun, inputs=update_inputs, outputs=outmodel)
        clean_depth.change(fn=model_from_scene_fun, inputs=update_inputs, outputs=outmodel)
        transparent_cams.change(model_from_scene_fun, inputs=update_inputs, outputs=outmodel)
        # find_btn.click(fn=get_3D_object_from_scene_fun,
        #                      inputs=[text_input, threshold, scene, min_conf_thr, as_pointcloud, mask_sky,
        #                               clean_depth, transparent_cams, cam_size],
        #                     outputs=outmodel)

    demo.launch(share=True, server_name=server_name, server_port=server_port)
