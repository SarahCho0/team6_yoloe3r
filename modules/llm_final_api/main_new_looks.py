"""
드롭박스로 선택한 인테리어 스타일(또는 AI 추천 스타일)을 바탕으로

1) 방 전체 스타일을 변경한 새 이미지를 1장 생성하고
2) 그 이미지를 img4new3r_org.png 로 저장한 뒤
3) 그 이미지를 기준으로 좌/우 각도 이미지를 1장씩 생성하는 스크립트.

전제:
- 1~2단계를 거쳐 parsed_report.json 이 생성되어 있음
- 프론트에서 드롭박스 선택 결과를 style_choice.json 으로 저장해 둠
"""
import os
import json
import shutil

from config import (
    API_KEY,
    STYLE_MODEL,
    SELECTED_IMAGE_PATH,
)

from style.style_client import run_style_model
from style.style_prompt import generate_style_prompt
from main_1img23 import make_one_image_to_three   

PARSED_REPORT_PATH = "parsed_report.json"
STYLE_CHOICE_PATH = "style_choice.json"
ORG_IMAGE_PATH = "img4new3r_org.png"  # 최종 결과물 이름


def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def decide_target_style(parsed_report: dict, style_choice: dict) -> str:
    pass

    selected = (style_choice.get("selected_style") or "").strip()

    # AI 추천 모드
    if selected in ("AI 추천", "AI추천"):
        print("\n드롭박스 선택: AI 추천")

        # recommend_styles가 있는 경우
        rec_list = parsed_report.get("recommended_styles") or []
        if rec_list:
            style_from_ai = (rec_list[0].get("style") or "").strip()
            if style_from_ai:
                print(f"  → recommended_styles[0].style 사용: {style_from_ai}")
                return style_from_ai

        # general_style 사용
        general = (parsed_report.get("general_style") or "").strip()
        if general:
            print(f"  → recommended_styles 없음, general_style 사용: {general}")
            return general

        fallback = "모던 (Modern Interior)"
        print(f"  → fallback: {fallback}")
        return fallback

    # 선택값이 없음
    if not selected:
        fallback = (parsed_report.get("general_style") or "모던 (Modern Interior)").strip()
        print(f"  → {fallback}")
        return fallback

    print(f"\n드롭박스 선택: {selected}")
    return selected


# 메인 실행
def main():

    # 1. 입력 데이터
    try:
        parsed_report = load_json(PARSED_REPORT_PATH)
    except Exception as e:
        print(f"parsed_report.json 로드 실패: {e}")
        return

    try:
        style_choice = load_json(STYLE_CHOICE_PATH)
    except Exception as e:
        print(f"style_choice.json 로드 실패: {e}")
        return

    # 2. 기준 이미지 선택 
    if os.path.exists(ORG_IMAGE_PATH):
        base_image_path = ORG_IMAGE_PATH
        print(f"\n기준 이미지: {ORG_IMAGE_PATH} (이전에 생성된 최종본 사용)")
    elif os.path.exists(SELECTED_IMAGE_PATH):
        base_image_path = SELECTED_IMAGE_PATH
        print(f"\n기준 이미지: {SELECTED_IMAGE_PATH} (최초 선택 이미지 사용)")
    else:
        print("사용할 입력 이미지가 없습니다. SELECTED_IMAGE_PATH 또는 img4new3r_org.png 중 하나는 있어야 합니다.")
        return

    print(f"리포트 파싱 파일: {PARSED_REPORT_PATH}")
    print(f"스타일 선택 파일: {STYLE_CHOICE_PATH}")

    # 3. 최종 target_style 결정
    target_style = decide_target_style(parsed_report, style_choice)
    print(f"\n최종 적용할 스타일: {target_style}")

    # 모든 가구 선택
    target_objects = "모든 가구와 데코 요소"

    try:
        style_prompt = generate_style_prompt(
            target_style=target_style,
            target_objects=target_objects,
        )

        image_bytes = run_style_model(
            api_key=API_KEY,
            model_name=STYLE_MODEL,
            image_path=base_image_path,
            prompt=style_prompt,
        )

        temp_output = "styled_new_look_tmp.jpg"
        with open(temp_output, "wb") as f:
            f.write(image_bytes)

        # 최종본은 항상 ORG_IMAGE_PATH 로 통일
        shutil.copyfile(temp_output, ORG_IMAGE_PATH)
        styled_image_path = ORG_IMAGE_PATH

        print(f"스타일 변경 이미지 저장 완료: {styled_image_path}")

    except Exception as e:
        print(f"스타일 변경(3단계) 중 에러 발생: {e}")
        return

    # 4. 좌&우 각도 이미지 2장 생성
    print("\n 4단계: 좌/우 각도 이미지 생성 시작 ---")

    try:
        make_one_image_to_three(
            api_key=API_KEY,
            model_name=STYLE_MODEL,
            input_image_path=styled_image_path,
        )
        print("\n 좌/우 각도 이미지 생성 완료!")
        print("   - img4new3r_left.png")
        print("   - img4new3r_right.png")
    except Exception as e:
        print(f"좌/우 각도 생성(4단계) 중 에러 발생: {e}")


if __name__ == "__main__":
    main()
