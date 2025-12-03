import cv2
import os
import numpy as np

def crop_by_result(result, filename, output_dir='output_crops', label_map=None):
    """
    YOLO/SAM 결과를 기반으로 이미지를 Crop하여 저장하고,
    해당 객체들의 Raw Mask 데이터를 딕셔너리로 반환하는 함수.
    
    Args:
        result: Ultralytics 모델 예측 결과 객체
        filename: 원본 이미지 파일명
        output_dir: 저장할 루트 디렉토리
        label_map: {class_id(int): class_name(str)} 형태의 매핑 딕셔너리 (옵션)
        
    Returns:
        mask_dict: { "폴더명(ClassName)": [raw_mask_array, ...] }
    """
    
    orig_img = result.orig_img
    h, w, _ = orig_img.shape

    # 반환할 딕셔너리 초기화
    mask_dict = {}

    # 탐지된 객체가 없으면 빈 딕셔너리 반환
    if result.masks is None:
        # print(f"[{filename}] 탐지된 객체가 없습니다.")
        return mask_dict

    # 데이터 추출
    # masks: (N, H_out, W_out) - 모델 출력 해상도의 마스크
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy()

    # 중복 저장 방지용 (Shape 기준)
    seen = []

    for i, mask in enumerate(masks):
        # ---------------------------------------------------------
        # [Step 1] 저장될 폴더명(= Class Name) 결정
        # ---------------------------------------------------------
        idx = int(cls_ids[i])
        
        # 1순위: 외부에서 주입된 label_map (예: YOLO가 찾은 실제 이름)
        if label_map is not None and idx in label_map:
            class_name = label_map[idx]
        # 2순위: 모델 내장 이름
        elif result.names and idx in result.names:
            class_name = result.names[idx]
        # 3순위: 그냥 숫자 ID
        else:
            class_name = str(idx)

        # ---------------------------------------------------------
        # [Step 2] 이미지 저장용 프로세싱 (Crop & RGBA)
        # ---------------------------------------------------------
        # 시각화를 위해 원본 크기로 리사이징
        mask_u8 = mask.astype(np.uint8)
        resized_mask = cv2.resize(mask_u8, (w, h))
        binary_mask = (resized_mask > 0.5).astype(np.uint8) * 255

        # 투명 배경 이미지 생성
        b, g, r = cv2.split(orig_img)
        rgba_img = cv2.merge([b, g, r, binary_mask])

        # Bounding Box 좌표 추출 및 클리핑
        x1, y1, x2, y2 = boxes[i].astype(int)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        
        # 이미지 Crop
        crop_img = rgba_img[y1:y2, x1:x2]

        # ---------------------------------------------------------
        # [Step 3] 파일 저장 및 데이터 수합
        # ---------------------------------------------------------
        # 중복 검사 (이미지 Shape 기준)
        if crop_img.shape not in seen:
            seen.append(crop_img.shape)
            
            # 1. 파일 저장 (폴더명 = class_name)
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(filename))[0]
            file_name = f"{base_name}_{i}.png"
            save_path = os.path.join(class_dir, file_name)
            
            cv2.imwrite(save_path, crop_img)
            # print(f"[저장 완료] {save_path}")

            # 2. 마스크 딕셔너리 저장 (Key = class_name)
            if class_name not in mask_dict:
                mask_dict[class_name] = []
            
            # [중요] 가공되지 않은 Raw Mask 원본을 저장
            mask_dict[class_name].append(mask)

    return mask_dict