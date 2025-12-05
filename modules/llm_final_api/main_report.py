import time
import json
import os
import sys
import shutil
from typing import Dict, Any

# =========================================================================
# 모듈 임포트: 절대 경로 반영 및 fallback 처리
# =========================================================================

# 1. config 모듈 임포트 (팀 환경 설정 가정)
try:
    from config import * except ImportError:
    # config 파일이 없을 경우 임시 기본값 설정 (실제 환경에서는 팀 구성원의 config 파일이 있어야 함)
    API_KEY = "dummy_api_key"
    REPORT_MODEL = "gemini-2.5-flash-preview-09-2025"
    INITIAL_IMAGE_PATHS = ["path/to/image.jpg"]
    SELECTED_IMAGE_PATH = "selected_image.jpg"

# 2. report 하위 모듈 임포트
try:
    # report/utils/report_parser.py 임포트
    from report.utils.report_parser import parse_report_output
    # report/report_client.py 임포트
    from report.report_client import run_report_model
    # report/report_prompt.py 임포트
    from report.report_prompt import report_prompt
    # report/summarize_prompt.py 임포트
    from report.summarize_prompt import SUMMARY_TEMPLATE

except ImportError:
    # 임포트 경로 문제 발생 시 (특히 Canvas 환경이나 로컬 환경의 경로 문제)
    # 현재 파일의 디렉토리를 기준으로 필요한 경로를 sys.path에 추가하여 재시도
    print("report 하위 모듈 임포트 실패. 경로 설정 재시도...")
    
    # 가정: main_report.py는 'modules/llm_final_api/report/' 안에 있음
    # 따라서 'modules/llm_final_api/'를 sys.path에 추가합니다.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir) # report/ -> llm_final_api/
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    try:
        from report.utils.report_parser import parse_report_output
        from report.report_client import run_report_model
        from report.report_prompt import report_prompt
        from report.summarize_prompt import SUMMARY_TEMPLATE
    except ImportError as e:
        print(f"최종 임포트 실패: {e}. 실행을 계속할 수 없습니다.")
        # 이 시점에서 필요한 모듈이 없으므로, 더 이상의 실행은 무의미합니다.
        sys.exit(1)


from ultralytics import YOLOE # YOLOE는 별도 설치된 모듈이므로 그대로 임포트

# =========================================================================
# 요약 리포트 파일 생성 함수 (SUMMARY_TEMPLATE 사용)
# =========================================================================
def create_summary_report_file(parsed_data: Dict[str, Any], raw_report_text: str):
    """
    파싱된 데이터를 기반으로 Gradio UI에 보여줄 요약 리포트 템플릿을 생성합니다.
    """
    
    # 데이터 구조 확인 및 기본값 설정
    mood_details = parsed_data.get("mood_details", [])
    
    # 1. 분위기 정의 및 유형별 확률 데이터 추출 및 기본값 설정
    # 파싱 실패 시 fallback 값은 요청하신 템플릿의 플레이스홀더와 유사하게 설정
    mood1_word = mood_details[0].get("word", "분위기1") if len(mood_details) > 0 else "분위기1"
    # 'percentage' 대신 'percent' 키를 사용하도록 수정합니다. (이전 파서의 구조에 따라 다름)
    mood1_percent = str(mood_details[0].get("percent", "확률1")) if len(mood_details) > 0 else "확률1"
    mood2_word = mood_details[1].get("word", "분위기2") if len(mood_details) > 1 else "분위기2"
    mood2_percent = str(mood_details[1].get("percent", "확률2")) if len(mood_details) > 1 else "확률2"
    mood3_word = mood_details[2].get("word", "분위기3") if len(mood_details) > 2 else "분위기3"
    mood3_percent = str(mood_details[2].get("percent", "확률3")) if len(mood_details) > 2 else "확률3"

    # 2. 가구 추천 데이터 추출 및 기본값 설정 (파서의 새로운 구조를 따름)
    furniture_recs = parsed_data.get("furniture_recommendations", {})
    
    # 가구 추가
    add_item = furniture_recs.get("add", {}).get("name", "추가 가구")

    # 가구 제거
    rem_item = furniture_recs.get("remove", {}).get("name", "제거 가구")

    # 가구 변경
    change_item = furniture_recs.get("change", {}).get("old_name", "변경 가구")
    rec_item = furniture_recs.get("change", {}).get("new_name", "추천 가구")


    # 3. 추천 스타일 데이터 추출 및 기본값 설정
    rec_styles = parsed_data.get("recommended_styles", [])
    
    # 여기서 'style' 키를 사용하여 추천 스타일 이름을 정확히 가져옵니다.
    rec_style = rec_styles[0].get("style", "추천 분위기") if rec_styles and rec_styles[0].get("style") else "추천 분위기"
    
    # 전체 분위기 스타일 (general_style)
    general_style = parsed_data.get("general_style", "내추럴하고 아늑한 절충적")

    # =====================================================================
    # raw_report_text의 목차 번호를 요약 리포트 형식에 맞게 변경 (토글 내용)
    # =====================================================================
    # 원본 LLM 출력: ## 3. 가구 추천, ## 4. 이런 스타일 어떠세요?
    # 요약 리포트: ## 2. 가구 추천, ## 3. 이런 스타일 어떠세요?
    modified_raw_text = raw_report_text.replace(
        "## 4. 이런 스타일 어떠세요?",
        "## 3. 이런 스타일 어떠세요?"
    )
    modified_raw_text = modified_raw_text.replace(
        "## 3. 가구 추천",
        "## 2. 가구 추천"
    )

    # =====================================================================
    # 최종 요약 콘텐츠 생성 (SUMMARY_TEMPLATE에 데이터 포매팅)
    # =====================================================================
    # SUMMARY_TEMPLATE의 구조를 모르므로, 이전 버전에서 사용했던 방식처럼 템플릿에 맞춰 포매팅합니다.
    # 단, 'SUMMARY_TEMPLATE'가 정의되어 있다고 가정하고 진행합니다.
    summary_content = SUMMARY_TEMPLATE.format(
        general_style=general_style,
        mood1_word=mood1_word,
        mood1_percent=mood1_percent,
        mood2_word=mood2_word,
        mood2_percent=mood2_percent,
        mood3_word=mood3_word,
        mood3_percent=mood3_percent,
        add_item=add_item,
        rem_item=rem_item,
        change_item=change_item,
        rec_item=rec_item,
        rec_style=rec_style,
        modified_raw_text=modified_raw_text
    )

    summary_output_path = "report_summarize.txt"
    with open(summary_output_path, "w", encoding="utf-8") as f:
        # 템플릿에 불필요한 공백이 있을 수 있으므로 strip()을 적용하여 출력
        f.write(summary_content.strip())
    
    print(f"요약 리포트 파일 생성 완료: {summary_output_path}")
    return summary_output_path
# =========================================================================


def main_report(img_path):
    # ----- 1단계: YOLOE를 이용한 최적의 입력 이미지 1장 선택 ------
    # (YOLOE 관련 로직은 사용자 요청에 따라 유지됨)
    model = YOLOE("yoloe-11s-seg.pt")
    max_cnt = 0
    max_idx = 0
    
    # YOLOE 예측은 시간이 걸릴 수 있으므로, 단일 이미지 리스트를 기대합니다.
    for i, img in enumerate(img_path):
        # YOLOE 모델을 사용하여 바운딩 박스 개수 확인
        results = model.predict(img)
        # results[0].boxes는 DetBoxes 객체이며, len()으로 바운딩 박스 개수를 얻습니다.
        current_cnt = len(results[0].boxes) 
        
        if current_cnt > max_cnt:
            max_idx = i
            max_cnt = current_cnt

    final_input_path = img_path[max_idx]
    print('최적 입력 이미지 : ' + final_input_path)
    shutil.copyfile(img_path[max_idx], SELECTED_IMAGE_PATH)

    # ------ 2단계: 공간 분석 리포트 생성 ------
    try:
        # Gemini에 이미지 + 분석용 프롬프트 전달
        raw_report_text = run_report_model(
            api_key=API_KEY,
            model_name=REPORT_MODEL,
            image_path=final_input_path,
            prompt=report_prompt,
        )

        time.sleep(1)

        # 텍스트 분석 및 파싱
        parsed_data = parse_report_output(raw_report_text)

        # 1. 원본 리포트 파일 저장 (기존 report_analysis_result.txt)
        report_output_path = "report_analysis_result.txt"
        with open(report_output_path, "w", encoding="utf-8") as f:
            f.write(raw_report_text)

        # 2. 파싱된 JSON 파일 저장
        parsed_json_path = "parsed_report.json"
        with open(parsed_json_path, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=4)
        
        # 3. 요약 리포트 파일 생성 (report_summarize.txt)
        create_summary_report_file(parsed_data, raw_report_text)


    except Exception as e:
        print(f"2단계 (리포트 분석) 중 에러 발생: {e}")
        return


if __name__ == "__main__":
    # 2. main_report 함수 호출 시 인수를 전달
    try:
        # config.py에 정의된 변수를 사용한다고 가정
        main_report(INITIAL_IMAGE_PATHS)
    except NameError:
        print("오류: 'INITIAL_IMAGE_PATHS' 변수를 config.py에서 찾을 수 없습니다. config.py 파일과 변수 이름을 확인하세요.")
    except Exception as e:
        print(f"스크립트 실행 중 예상치 못한 에러 발생: {e}")
