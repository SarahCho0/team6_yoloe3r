from ultralytics.models.sam import SAM2DynamicInteractivePredictor
from ultralytics import YOLOE
from .crop_by_result import crop_by_result 
import os

def crop(img_path):
    overrides = dict(conf=0.01, task="segment", mode="predict", imgsz=1024, model="sam2_t.pt", save=False)
    predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=50)

    model = YOLOE('yoloe-11l-seg.pt')
    names = [
        "Kitchen Cabinet", "Mini Kitchen", "Kitchen Island/Cart", "Kitchen Appliance", 
        "Kitchen Countertop", "Kitchen Pantry", "Kitchen System", "Office Desk/Chair Set", 
        "Conference Chair", "Gaming Furniture", "Conference Table", "Desk/Chair Set", 
        "Office Chair", "Computer Desk", "Vanity Chair/Stool", "Toddler Chair", 
        "Childrens Chair", "Childrens Table", "Step Stool", "Bench", "Cafe Furniture", 
        "Stool", "Bar Table/Chair", "Coffee/Side Table", "Chair", "Table", 
        "Dining Furniture", "Chaise Longue/Couch", "Footstool", "Sofa Bed", 
        "Armchair", "Sofa", "Bedroom Set", "Bed with Mattress", "Bedside Table", 
        "Bed Frame", "Shoe Cabinet", "Storage Unit", "Toy Storage", "Hallway Set", 
        "Partition", "Drawer/Nightstand", "Storage System", "Sideboard/Console Table", 
        "Trolley", "TV/Media Furniture", "Outdoor Storage", "Warehouse Storage", "cabinet"
    ]

    # img_dir = 'sample_test'
    # 프레임 순서 보장
    # imgs = sorted(os.listdir(img_dir))
    imgs = img_path
    
    # [핵심] 모든 프레임의 마스크 데이터를 모을 리스트
    all_frames_masks = [] 

    for i, img in enumerate(imgs):
        imgsrc = img
        current_frame_dict = {}

        if i == 0:
            # YOLO 클래스 설정 (탐지 정확도를 위해 필요)
            model.set_classes(names, model.get_text_pe(names))
            results = model.predict(imgsrc)
            
            # SAM 메모리 업데이트 (YOLO 박스 ID 0, 1, 2... 그대로 사용)
            predictor(source=imgsrc, 
                      bboxes=results[0].boxes.xyxy.cpu().numpy(), 
                      obj_ids=[k for k in range(len(results[0].boxes))], 
                      update_memory=True)
            
            results = predictor(source=imgsrc)
            os.makedirs('output_crops/', exist_ok=True)
            
            # 매핑 없이 호출 -> 폴더명 "0", "1"... 생성
            current_frame_dict = crop_by_result(results[0], img)
            
        else:
            results = predictor(source=imgsrc)  
            current_frame_dict = crop_by_result(results[0], img)

        # 리스트에 추가
        all_frames_masks.append(current_frame_dict)
        
        if current_frame_dict:
            print(f"Frame {i}: Saved objects {list(current_frame_dict.keys())}")
    return all_frames_masks

    # --- 3D 생성 함수 호출 예시 ---
    # 이제 target_idx는 "0", "1" 같은 문자열이어야 합니다.
    # 사용자가 'output_crops' 폴더에서 직접 확인 후 원하는 번호를 넣습니다.
    
    # target_idx = "0" 
    
    # get_3D_object_from_scene(
    #     ...,
    #     clsidx=target_idx,          # "0" 입력
    #     mask_list=all_frames_masks  # 수합된 리스트 전달
    # )

if __name__ == "__main__":  
    crop()