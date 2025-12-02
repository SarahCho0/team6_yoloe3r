import os

from style.style_client import run_style_model  # Gemini 호출 함수.
from style.style_prompt import generate_style_prompt  # 스타일 프롬프트 재사용.


def run_image_edit(
    api_key: str,
    model_name: str,
    input_image_path: str,
    base_style: str,
    edit_instruction: str,
    step_name: str,
) -> str:
    """
    한 번의 편집(추가/제거/변경)을 수행하고 새로운 이미지를 저장한 뒤 경로를 반환한다.

    - base_style: 공간의 기본 스타일 설명 (예: "차분하고 따뜻한 북유럽")
    - edit_instruction: 이번 단계에서 수행할 변경에 대한 자연어 설명
    - step_name: "add" / "remove" / "change" 등, 파일 이름에 사용
    """

    if not os.path.exists(input_image_path):
        print(f" run_image_edit: 입력 이미지가 존재하지 않습니다: {input_image_path}")
        return input_image_path

    target_style = base_style or "모던"
    # generate_style_prompt의 target_objects 자리에, 이번에 수행할 변경 내용을 그대로 넣어준다.
    target_objects = (
        f"이번 단계에서 수행해야 할 변경 사항:\n"
        f"- {edit_instruction}\n\n"
        "이미 현재 이미지가 위 요청을 충분히 만족하고 있다면, 그 부분은 변경하지 마세요.\n"
        f"위에서 요청한 변경 사항만 적용하고, 그 외의 구조와 가구 배치, 카메라 구도는 그대로 유지하세요."
    )

    try:
        prompt = generate_style_prompt(
            target_style=target_style,
            target_objects=target_objects,
        )

        # Gemini 스타일 모델 호출 -> 이미지 바이트 획득.
        image_bytes = run_style_model(
            api_key=api_key,
            model_name=model_name,
            image_path=input_image_path,
            prompt=prompt,
        )

        output_path = f"modified_{step_name}.jpg"
        with open(output_path, "wb") as f:
            f.write(image_bytes)

        print(f"  '{step_name}' 단계 편집 완료 → {output_path}")
        return output_path

    except Exception as e:
        print(f"  run_image_edit('{step_name}') 중 에러 발생: {e}")
        # 실패해도 파이프라인이 완전히 멈추지 않도록, 이전 이미지를 그대로 반환.
        return input_image_path
