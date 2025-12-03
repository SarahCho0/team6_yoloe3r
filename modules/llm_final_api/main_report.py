import time
import json
from .config import *
from .report.utils.image_selector import select_best_image
from .report.utils.report_parser import parse_report_output
from .report.report_client import run_report_model
from .report.report_prompt import report_prompt
from ultralytics import YOLOE


def main_report(img_path):
    # ----- 1단계: 3장의 이미지 중 최적의 입력 이미지 1장 선택 ------
    model = YOLOE("yoloe-11s-seg.pt")
    max_cnt=0
    maxidx = 0
    for i, img in enumerate(img_path):
        results = model.predict(img)
        if len(results[0].boxes)>max_cnt:
            maxidx = i
            max_cnt=len(results[0].boxes)
        else:
            pass

    # final_input_path = select_best_image(
    #     api_key=API_KEY, 
    #     model_name=REPORT_MODEL,        # 리포트는 Gemini-2.5-flash 사용
    #     input_paths=img_path,    # ui 에서 입력한 이미지
    #     selected_output_path=SELECTED_IMAGE_PATH,
    # )

    # if not final_input_path:
    #     print("유효한 입력 이미지를 확인하세요.")
    #     return

    print('최적 입력 이미지 : '+img_path[maxidx])

    # ------ 2단계: 공간 분석 리포트 생성 ------
    try:
        # Gemini에 이미지 + 분석용 프롬프트 전달
        raw_report_text = run_report_model(
            api_key=API_KEY,
            model_name=REPORT_MODEL, # 리포트는 Gemini-2.5-flash 사용
            image_path=img_path[maxidx],  # 1단계에서 선택된 이미지 사용
            prompt=report_prompt,
        )

        # Gemini 응답 대기 시간
        time.sleep(1)

        # 전체 리포트 파싱
        parsed_data = parse_report_output(raw_report_text)

        # 2-1) 리포트 원본 txt 저장
        report_output_path = "report_analysis_result.txt"
        with open(report_output_path, "w", encoding="utf-8") as f:
            f.write(raw_report_text)

        # 2-2) 파싱된 전체 데이터를 JSON으로 저장
        parsed_json_path = "parsed_report.json"
        with open(parsed_json_path, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=4)

        # --------------------------------------------------

    except Exception as e:
        print(f"2단계 (리포트 분석) 중 에러 발생: {e}")
        # 에러 시 그냥 종료
        return


if __name__ == "__main__":
    main_report()
