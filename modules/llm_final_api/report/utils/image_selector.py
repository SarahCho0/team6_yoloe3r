import os
from google import genai
from google.genai import types
from PIL import Image
import shutil
from typing import List

# config 파일의 API_KEY와 모델명을 사용.
# 실제 main 함수에서 config를 import 할 것이므로, 여기서는 함수 인자로 받도록 함.

def select_best_image(api_key: str, model_name: str, input_paths: List[str], selected_output_path: str) -> str:
    """
    주어진 3장의 이미지 경로 중, 가구가 가장 많고 분석에 적합한 1장의 이미지를 선택하고,
    그 이미지를 selected_output_path에 복사하여 저장한 후 경로를 반환합니다.

    Args:
        api_key (str): Google GenAI API Key.
        model_name (str): 사용할 AI 모델 ('gemini-2.0-flash').
        input_paths (List[str]): 3장의 입력 이미지 경로 리스트.
        selected_output_path (str): 선택된 이미지를 복사하여 저장할 경로.
        
    Returns:
        str: 최종 선택된 이미지의 경로 (selected_output_path).
    """
    print("------ 3장 중 최적 이미지 선택 시작 ------")
    # Gemini API 클라이언트 초기화.
    client = genai.Client(api_key=api_key)
    best_image_path = None
    # 가구 개수 추적용 변수. 초기값을 -1로 설정하여 어떤 이미지도 선택되지 않은 초기 상태를 나타냄.
    max_furniture_count = -1
    
    # ------ AI 프롬프트: 각 이미지의 가구 개수를 정확히 세도록 지시. ------
    selection_prompt = """
    주어진 이미지에 보이는 '가구(furniture)'와 '주요 데코 요소'의 개수를 정확히 세어서
    '숫자'만 출력해 주세요. 가구와 데코 요소의 경계가 모호할 경우, 방의 분석에 중요하다고
    판단되는 항목(침대, 소파, 테이블, 의자, 선반, TV, 주요 조명 등)만 포함하세요.
    예시: 5
    """

    for path in input_paths:
        if not os.path.exists(path):
            print(f" 경고: 파일 없음 - {path}. 이 경로는 건너뜁니다.")
            continue
            
        print(f"  -> 이미지 분석 중: {path}")
        
        # 이미지 파일을 바이너리(바이트) 형태로 읽어 Gemini API의 입력 파트(Part)로 전달 준비.
        try:
            # 이미지 바이트 로드.
            with open(path, "rb") as f:
                img_bytes = f.read()

            # Gemini-2.5-flash 모델 호출: 이미지와 프롬프트를 함께 전달하여 가구 개수 분석 요청.
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    selection_prompt
                ]
            )
            
            # 응답 텍스트에서 숫자만 추출하는 파싱 로직.
            count_text = response.text.strip()
            
            # 문자열에서 숫자(digit)만 필터링하여 합친 후, 정수형으로 변환.
            # LLM이 숫자 외의 문자를 포함하더라도 안정적으로 숫자를 추출하기 위함.
            furniture_count = int("".join(filter(str.isdigit, count_text)))
            
            print(f"   분석 결과: 가구 {furniture_count}개")
            
            # 최대 가구 수 업데이트: 가구 수가 더 많은 이미지를 '최적 이미지'로 선택.
            if furniture_count > max_furniture_count:
                max_furniture_count = furniture_count
                best_image_path = path

        except Exception as e:
            print(f"  AI 분석 오류: {e}. 이 경로는 0개로 간주합니다.")
            
    # ------ 최종 선택 및 파일 복사 ------
    if best_image_path:
        # shutil.copyfile을 사용하여 선택된 이미지를 지정된 경로로 복사.
        shutil.copyfile(best_image_path, selected_output_path)
        print(f"\n 최종 선택된 이미지: {best_image_path}")
        print(f"    -> {selected_output_path}에 복사 완료.")
        return selected_output_path
    else:
        print("\n 오류: 분석할 수 있는 유효한 이미지 경로가 없습니다.")
        return ""


