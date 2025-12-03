import time
import json
# 1. 상대 경로 임포트를 절대 경로 임포트로 수정
from config import * 
from report.utils.report_parser import parse_report_output
from report.report_client import run_report_model
from report.report_prompt import report_prompt
from ultralytics import YOLOE # select_best_image 로직을 YOLOE로 대체했으므로 

def main_report(img_path):
    # ----- 1단계: YOLOE를 이용한 최적의 입력 이미지 1장 선택 ------
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
        # else: pass (생략 가능)

    final_input_path = img_path[max_idx]
    print('최적 입력 이미지 : ' + final_input_path)

    # ------ 2단계: 공간 분석 리포트 생성 ------
    try:
        # ... (2단계 로직은 그대로 유지) ...
        # Gemini에 이미지 + 분석용 프롬프트 전달
        raw_report_text = run_report_model(
            api_key=API_KEY,
            model_name=REPORT_MODEL,
            image_path=final_input_path,
            prompt=report_prompt,
        )

        time.sleep(1)

        parsed_data = parse_report_output(raw_report_text)

        report_output_path = "report_analysis_result.txt"
        with open(report_output_path, "w", encoding="utf-8") as f:
            f.write(raw_report_text)

        parsed_json_path = "parsed_report.json"
        with open(parsed_json_path, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=4)

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
