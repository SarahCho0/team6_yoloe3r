from google import genai
from google.genai import types
import mimetypes

# 보고서 모델을 실행하는 함수
def run_report_model(api_key, model_name, image_path, prompt):
    client = genai.Client(api_key=api_key) # 제공된 API 키로 client 초기화

    with open(image_path, "rb") as f:
        img_bytes = f.read()
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/jpeg"

    # Gemini 모델에 콘텐츠를 생성하도록 요청
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_bytes(
                data=img_bytes,
                mime_type=mime_type
            ),
            prompt
        ]
    )

    # 모델 응답의 텍스트 부분 반환
    return response.text
