"""
VGGT 기반 3D 재구성 모듈
PE3R(DUSt3R+MASt3R) 재구성 파트를 대체하는 선택적 모듈.
VGGT 미설치 시 PE3R로 자동 폴백.
"""

import torch
import numpy as np
import sys
import os


def load_vggt_model(device: str = "cuda"):
    """VGGT 모델 로드 (HuggingFace에서 자동 다운로드)"""
    try:
        # Colab: /content/vggt, 로컬: PATH에 vggt가 있으면 동작
        vggt_path = os.environ.get("VGGT_PATH", "/content/vggt")
        if os.path.exists(vggt_path):
            sys.path.insert(0, vggt_path)

        from vggt.models.vggt import VGGT
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        model.eval()
        print("✅ VGGT 모델 로드 완료")
        return model
    except Exception as e:
        print(f"⚠️ VGGT 로드 실패: {e}")
        print("   PE3R로 폴백합니다.")
        return None


def reconstruct_with_vggt(image_paths: list, model, device: str = "cuda") -> dict:
    """
    VGGT로 3D 재구성.

    Returns:
        dict:
            point_cloud  : np.ndarray (N, 3)
            depth_maps   : list of np.ndarray  (H, W)
            camera_poses : np.ndarray (N, 4, 4)
            confidence   : np.ndarray (N,)
            raw_point_maps: torch.Tensor (1, N, H, W, 3)  — 추가 처리용
            raw_conf_maps : torch.Tensor (1, N, H, W)
    """
    from vggt.utils.load_fn import load_and_preprocess_images

    images, _ = load_and_preprocess_images(image_paths)
    images = images.to(device)

    with torch.no_grad():
        # PyTorch 2.x: torch.amp.autocast 사용
        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else torch.inference_mode()
        )
        with autocast_ctx:
            predictions = model(images)

    point_maps   = predictions["world_points"]   # (1, N, H, W, 3)
    conf_maps    = predictions["depth_conf"]      # (1, N, H, W)
    depth_maps   = predictions["depth"]           # (1, N, H, W)
    camera_poses = predictions["extrinsics"]      # (1, N, 4, 4)

    pts  = point_maps[0].cpu().float().numpy().reshape(-1, 3)
    conf = conf_maps[0].cpu().float().numpy().reshape(-1)
    mask = conf > 0.1

    return {
        "point_cloud":    pts[mask],
        "depth_maps":     [depth_maps[0, i].cpu().float().numpy() for i in range(depth_maps.shape[1])],
        "camera_poses":   camera_poses[0].cpu().float().numpy(),
        "confidence":     conf[mask],
        "raw_point_maps": point_maps,
        "raw_conf_maps":  conf_maps,
    }


def save_ply(point_cloud: np.ndarray, output_path: str = "reconstruction.ply") -> str:
    """포인트 클라우드를 PLY 파일로 저장. open3d 없으면 npy로 대체."""
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"✅ PLY 저장: {output_path} ({len(point_cloud):,} points)")
        return output_path
    except ImportError:
        npy_path = output_path.replace(".ply", ".npy")
        np.save(npy_path, point_cloud)
        print(f"✅ NPY 저장 (open3d 없음): {npy_path}")
        return npy_path


def vggt_result_to_glb(result: dict, output_path: str = "vggt_scene.glb") -> str:
    """
    VGGT 재구성 결과를 Gradio Model3D 호환 GLB 파일로 변환.
    trimesh를 사용해 포인트 클라우드를 GLB로 내보냄.

    Returns:
        GLB 파일 경로 (성공) 또는 None (실패)
    """
    try:
        import trimesh
        pts = result["point_cloud"]
        if len(pts) == 0:
            print("⚠️ VGGT: 포인트 클라우드가 비어 있습니다.")
            return None

        # 포인트 수가 너무 많으면 다운샘플링 (GLB 파일 크기 제한)
        if len(pts) > 200_000:
            idx = np.random.choice(len(pts), 200_000, replace=False)
            pts = pts[idx]

        colors = np.ones((len(pts), 4), dtype=np.uint8) * [180, 180, 220, 255]
        pcd = trimesh.PointCloud(pts, colors=colors)
        scene = trimesh.Scene(pcd)
        scene.export(output_path)
        print(f"✅ VGGT GLB 저장: {output_path}")
        return output_path
    except Exception as e:
        print(f"⚠️ GLB 변환 실패: {e}")
        return None
