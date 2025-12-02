import os
import json
import shutil

from config import (
    API_KEY,
    STYLE_MODEL,
    SELECTED_IMAGE_PATH,
)

from edit.image_edit import run_image_edit
from main_1img23 import make_one_image_to_three  

PARSED_REPORT_PATH = "parsed_report.json" # main_report.py에서 생성
USER_CHOICE_PATH = "user_choice.json" # 사용자 선택값 저장
ORG_IMAGE_PATH = "img4new3r_org.png"  # 최종 결과물 이름

def load_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # ------ 1. 입력 파일/경로 로드 ------
    try:
        parsed_report = load_json(PARSED_REPORT_PATH)
    except Exception as e:
        print(f"parsed_report.json 로드 실패: {e}")
        return

    try:
        user_choice = load_json(USER_CHOICE_PATH)
    except Exception as e:
        print(f"user_choice.json 로드 실패: {e}")
        return

    # 기준 이미지 결정
    if os.path.exists(ORG_IMAGE_PATH):
        # 이미 수정본이 있는 경우
        base_image_path = ORG_IMAGE_PATH
        print(f"\n기준 이미지: {ORG_IMAGE_PATH} (이전에 생성된 최종본 사용)")
    elif os.path.exists(SELECTED_IMAGE_PATH):
        # 수정본이 없는 경우, main_report에서 선택된 최적 이미지 사용
        base_image_path = SELECTED_IMAGE_PATH
        print(f"\n기준 이미지: {SELECTED_IMAGE_PATH} (최초 선택 이미지 사용)")
    else:
        print("사용할 입력 이미지가 없습니다. SELECTED_IMAGE_PATH 또는 img4new3r_org.png 중 하나는 있어야 합니다.")
        return

    print(f"리포트 파싱 파일: {PARSED_REPORT_PATH}")
    print(f"사용자 선택 파일: {USER_CHOICE_PATH}")

    # ------ 2. 리포트 분석 정보 해석 ------
    # 기본 스타일 : "모던"
    base_style = parsed_report.get("general_style", "모던")

    # 추천 항목들
    rec_add_list = parsed_report.get("recommendations_add", []) or []
    rec_remove_list = parsed_report.get("recommendations_remove", []) or []
    rec_change_list = parsed_report.get("recommendations_change", []) or []

    # 추천 항목 중 첫 번째만 사용
    rec_add = rec_add_list[0] if rec_add_list else None
    rec_remove = rec_remove_list[0] if rec_remove_list else None
    rec_change = rec_change_list[0] if rec_change_list else None

    # 사용자 선택값
    use_add = bool(user_choice.get("use_add", False))
    use_remove = bool(user_choice.get("use_remove", False))
    use_change = bool(user_choice.get("use_change", False))

    print("\n사용자 선택 상태:")
    print(f"  - 추가(add) 적용 여부: {use_add}")
    print(f"  - 제거(remove) 적용 여부: {use_remove}")
    print(f"  - 변경(change) 적용 여부: {use_change}")

    # 현재 이미지 경로 
    current_image_path = base_image_path

    # ------ 3. 추가(add) 단계 ------
    if use_add and rec_add is not None:
        add_item = rec_add.get("item", "")
        add_reason = rec_add.get("reason", "")
        edit_instruction_add = (
            f"현재 공간의 분위기를 유지하면서, '{add_item}'를(을) 자연스럽게 추가하세요. "
            f"{add_reason} "
            f"추가되는 가구는 방의 크기와 기존 동선을 해치지 않도록 적절한 위치와 크기로 배치하세요."
        )

        print(f"대상: {add_item}")
        current_image_path = run_image_edit(
            api_key=API_KEY,
            model_name=STYLE_MODEL,
            input_image_path=current_image_path,
            base_style=base_style,
            edit_instruction=edit_instruction_add,
            step_name="add",
        )
    else:
        pass

    # ------ 4. 제거(remove) 단계 -------
    if use_remove and rec_remove is not None:
        remove_item = rec_remove.get("item", "")
        remove_reason = rec_remove.get("reason", "")
        edit_instruction_remove = (
            f"현재 공간에서 '{remove_item}'를(을) 제거하세요. "
            f"{remove_reason} "
            f"제거 후 생기는 빈 공간은 자연스럽게 보이도록 주변 가구와 조화를 이루게 하되, "
            f"새로운 큰 가구를 추가하지는 마세요."
        )

        print(f"대상: {remove_item}")
        current_image_path = run_image_edit(
            api_key=API_KEY,
            model_name=STYLE_MODEL,
            input_image_path=current_image_path,
            base_style=base_style,
            edit_instruction=edit_instruction_remove,
            step_name="remove",
        )
    else:
        pass

    # ------ 4. 변경(change) 단계 ------
    if use_change and rec_change is not None:
        from_item = rec_change.get("from_item", "")
        to_item = rec_change.get("to_item", "")
        change_reason = rec_change.get("reason", "")
        edit_instruction_change = (
            f"현재 공간에서 '{from_item}'를(을) '{to_item}'로 교체하세요. "
            f"{change_reason} "
            f"교체된 가구의 위치와 대략적인 크기는 기존과 비슷하게 유지하며, "
            f"방의 전체 구조와 다른 가구, 소품은 변경하지 마세요."
        )

        print(f"대상: {from_item} -> {to_item}")
        current_image_path = run_image_edit(
            api_key=API_KEY,
            model_name=STYLE_MODEL,
            input_image_path=current_image_path,
            base_style=base_style,
            edit_instruction=edit_instruction_change,
            step_name="change",
        )
    else:
        pass

    # ------ 5. 최종 결과물 저장 -------
    final_image_path = current_image_path

    # 최종 결과를 항상 img4new3r_org.png 로 통일
    if os.path.exists(final_image_path) and final_image_path != ORG_IMAGE_PATH:
        shutil.copyfile(final_image_path, ORG_IMAGE_PATH)
        final_image_path = ORG_IMAGE_PATH
    else:
        # 이미 ORG_IMAGE_PATH 를 쓰고 있었던 경우 
        final_image_path = ORG_IMAGE_PATH

    print(f"3단계(추가/제거/변경)까지 완료된 최종 이미지: {final_image_path}")

    # ------ 6. 좌&우 각도 이미지 생성 ------
    print("\n4단계: 좌&우 각도 이미지 생성")

    try:
        make_one_image_to_three(
            api_key=API_KEY,
            model_name=STYLE_MODEL,
            input_image_path=final_image_path,
        )
        print("   - img4new3r_left.png")
        print("   - img4new3r_right.png")
    except Exception as e:
        print(f"4단계(좌/우 각도 생성) 중 에러 발생: {e}")

if __name__ == "__main__":
    main()
