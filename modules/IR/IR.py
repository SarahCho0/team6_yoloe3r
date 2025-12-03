import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import faiss
import os
import glob
from tqdm import tqdm
import sys
import requests
import pandas as pd
from pandas import Series, DataFrame
import sqlite3
import urllib.request
import io
import cv2
import numpy as np
from ultralytics import YOLOE
import time


# ----------------------------------------------------
# 1. 모델 로드 및 전처리 정의
# ----------------------------------------------------

def IR():

    # DINOv2 모델 로드 함수
    def load_dinov2_vits():
        try:
            # dinov2_vits14 (ViT-Small) 모델 로드
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        except Exception as e:
            print(f"DINOv2 모델 로드 실패: {e}")
            return None

        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        print("DINOv2 ViT-S 모델 로드 완료.")
        return model

    # 이미지 전처리 파이프라인
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # 로컬/마운트된 파일에서 이미지 로드 및 특징 벡터 추출 함수
    def extract_features_from_mounted_drive(file_path, model, transform):
        try:
            # file_path는 이제 '/content/drive/MyDrive/...' 형태의 경로가 됩니다.
            img = Image.open(file_path).convert("RGB")
        except FileNotFoundError:
            print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인하세요: {file_path}")
            return None
        except Exception as e:
            print(f"이미지 로드 오류: {e}")
            return None

        img_tensor = transform(img).unsqueeze(0)
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        with torch.no_grad():
            features = model(img_tensor)
            # 특징 벡터를 L2 정규화 (코사인 유사도 계산을 위함)
            features = F.normalize(features, p=2, dim=1)

        return features.cpu().numpy() # NumPy 배열로 변환하여 반환

    # DINOv2 모델 로드
    dinov2_model = load_dinov2_vits()

    names = [
        "Kitchen Cabinet",              # 700292: 주방 캐비닛
        "Mini Kitchen",                 # 22957: 미니 주방
        "Kitchen Island/Cart",          # 10471: 아일랜드 식탁/주방 카트
        "Kitchen Appliance",            # ka002: 주방 가전
        "Kitchen Countertop",           # 24264: 주방 조리대
        "Kitchen Pantry",               # 16200: 주방 팬트리
        "Kitchen System",               # ka003: 주방 시스템
        "Office Desk/Chair Set",        # 700424: 오피스 책상/의자 세트
        "Conference Chair",             # 47068: 회의실 의자
        "Gaming Furniture",             # 55002: 게임용 가구
        "Conference Table",             # 54173: 회의 테이블
        "Desk/Chair Set",               # 53249: 책상/의자 세트
        "Office Chair",                 # 20652: 의자/사무실 의자
        "Computer Desk",                # 20649: 책상/컴퓨터 책상
        "Vanity Chair/Stool",           # 59250: 화장대 의자/스툴
        "Toddler Chair",                # 45782: 영유아 의자
        "Childrens Chair",              # bc004: 어린이 의자
        "Childrens Table",              # 18768: 어린이 테이블
        "Step Stool",                   # 20611: 스텝 스툴/스텝 의자
        "Bench",                        # 700319: 벤치
        "Cafe Furniture",               # 19141: 카페 가구
        "Stool",                        # 22659: 스툴
        "Bar Table/Chair",              # 16244: 바 테이블/의자
        "Coffee/Side Table",            # 10705: 커피테이블/사이드테이블
        "Chair",                        # 700676: 의자
        "Table",                        # 700675: 테이블
        "Dining Furniture",             # 700417: 다이닝 가구
        "Chaise Longue/Couch",          # 57527: 긴 의자/카우치
        "Footstool",                    # 20926: 풋스툴/발받침대
        "Sofa Bed",                     # 10663: 소파침대/소파베드
        "Armchair",                     # fu006: 암체어/안락의자
        "Sofa",                         # fu003: 소파
        "Bedroom Set",                  # 54992: 침실 가구 세트
        "Bed with Mattress",            # 700513: 매트리스 포함 침대
        "Bedside Table",                # 20656: 침대 협탁
        "Bed Frame",                    # bm003: 침대/침대프레임
        "Shoe Cabinet",                 # 10456: 신발장
        "Storage Unit",                 # 10385: 수납유닛/수납장
        "Toy Storage",                  # 20474: 장난감 수납/정리함
        "Hallway Set",                  # 700411: 복도 가구 세트
        "Partition",                    # 46080: 칸막이/파티션
        "Drawer/Nightstand",            # st004: 서랍장/침대협탁
        "Storage System",               # 46052: 수납 솔루션 시스템
        "Sideboard/Console Table",      # 30454: 거실장/찬장/콘솔테이블
        "Trolley",                      # fu005: 트롤리/미니 카트
        "TV/Media Furniture",           # 10475: TV/멀티미디어 가구
        "Outdoor Storage",              # 21958: 야외용 수납가구
        "Warehouse Storage",
        "cabinet"           # 700440: 창고 수납
    ]




    from torch.jit import save
    # --- 1. 파일 경로 및 검색 설정 ---
    INDEX_FILE_PATH = "modules/IR/DB/dinov2_product_catalog2.faiss" # 저장된 .faiss 파일 이름
    TEST_DIR = 'output_crops'

    # 마운트된 Google Drive의 테스트 이미지 폴더 경로 (사용자 환경에 맞게 수정하세요!)
    folders = os.listdir(TEST_DIR)

    K = 5 # 상위 K개 유사 이미지를 찾습니다.

    print("\n===== 4. 테스트 이미지 검증 시작 =====")

    # --- 2. FAISS 인덱스 로컬 파일에서 로드 ---
    loaded_index = None
    try:
        if os.path.exists(INDEX_FILE_PATH):
            # faiss.read_index를 사용하여 저장된 인덱스 파일을 메모리로 로드
            loaded_index = faiss.read_index(INDEX_FILE_PATH)
            print(f"FAISS 인덱스 로드 완료: 총 {loaded_index.ntotal}개 항목.")
        else:
            print(f"❌ 오류: FAISS 인덱스 파일 '{INDEX_FILE_PATH}'을 찾을 수 없습니다.")
            print("3단계에서 이 이름으로 파일을 저장했는지 확인하거나 경로를 수정하세요.")
    except Exception as e:
        print(f"❌ FAISS 인덱스 로드 중 오류 발생: {e}")


    output_result = [] # 최종 결과를 저장할 리스트

    # TEST_DIR/0/image.jpg 구조라고 가정
    for folder_name in sorted(os.listdir(TEST_DIR)):
        folder_path = os.path.join(TEST_DIR, folder_name)
        
        # 폴더가 아니면 건너뜀 (혹시 모를 .DS_Store 등 방지)
        if not os.path.isdir(folder_path):
            continue

        print(f"\nProcessing Folder: {folder_name} ...")
        
        # 이 폴더(가구 하나)에서 나온 모든 검색 후보를 저장할 리스트
        # 형식: {'id': vector_id, 'score': similarity_score, 'file': filename}
        folder_candidates = []

        # 1. 폴더 내 이미지들 순회하며 검색
        image_files = os.listdir(folder_path)
        if not image_files:
            continue # 빈 폴더 스킵

        for img in image_files:
            imgpath = os.path.join(folder_path, img)
            
            # 특징 추출
            query_vector_np = extract_features_from_mounted_drive(
                imgpath, dinov2_model, transform
            )

            if query_vector_np is not None:
                # 검색 실행 (Top K)
                distances, indices = loaded_index.search(query_vector_np, K)
                
                # 결과 수집
                for i in range(K):
                    folder_candidates.append({
                        'id': indices[0][i],
                        'score': distances[0][i],
                        'file': img # 어떤 이미지에서 나왔는지 (디버깅용)
                    })
            else:
                print(f"  [Warning] '{img}' 특징 추출 실패")

        # 2. 결과가 없으면 건너뜀
        if not folder_candidates:
            print(f"  -> 결과 없음")
            continue

        # 3. 최적의 결과 선정 로직 (Voting & Scoring)
        # 데이터를 처리하기 쉽게 DataFrame으로 변환
        df_cand = pd.DataFrame(folder_candidates)

        # (1) ID별로 몇 번 등장했는지 카운트 (겹치는 게 있는지 확인)
        id_counts = df_cand['id'].value_counts()
        
        # (2) 각 ID별 최고 점수(Max Score) 계산
        # 같은 ID가 여러 번 나왔다면 그 중 가장 높은 점수를 대표 점수로 씀
        id_max_scores = df_cand.groupby('id')['score'].max()

        # (3) 정렬 기준 생성:
        # 1순위: 등장 횟수(count) 내림차순 (많이 겹칠수록 확실한 정답)
        # 2순위: 점수(score) 내림차순 (횟수가 같다면 유사도가 높은 게 정답)
        
        # 결과를 담을 임시 리스트
        final_ranking = []
        for vector_id in id_counts.index:
            count = id_counts[vector_id]
            score = id_max_scores[vector_id]
            final_ranking.append((vector_id, count, score))
        
        # 정렬 실행 (count DESC, score DESC)
        final_ranking.sort(key=lambda x: (x[1], x[2]), reverse=True)

        # ---------------------------------------------------------
        # 4. 최종 1위 선정 및 DB 조회
        best_id, best_count, best_score = final_ranking[0]
        labels = np.load('modules/IR/catalog_labels2.npy') # 데이터 로드. @파일명
        model_name=labels[best_id]
        # DB에서 이름 가져오기 
        # try:
        #     df_db = pd.read_sql(f"SELECT product_name FROM products_images WHERE product_name={model_name}", con)
        #     model_name = df_db['product_name'][0] if not df_db.empty else "Unknown"
        # except Exception as e:
        #     model_name = "DB Error"

        # 결과 저장
        result_entry = {
            'folder_id': folder_name,       # 0, 1, 2...
            'predicted_name': model_name,   # 모델명       # 벡터 ID   # 몇 개의 이미지에서 겹쳤는지
            'max_similarity': best_score    # 최고 유사도 점수
        }
        output_result.append(result_entry)

        # 콘솔 출력 (확인용)
        match_type = "겹침(Overlap)" if best_count > 1 else "단독(Highest Score)"
        print(f"  -> 최종 선정: {model_name} (ID: {best_id})")
        print(f"     근거: {match_type}, 등장횟수: {best_count}/{len(image_files)}, 유사도: {best_score:.4f}")

    # 모든 폴더 처리 후 결과 확인
    print(output_result)
    return output_result
