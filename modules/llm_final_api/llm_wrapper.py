"""
IN3R LLM API Wrapper
Gemini / Claude / OpenAI 를 동일한 인터페이스로 사용

사용법:
    from modules.llm_final_api.llm_wrapper import call_llm, call_image_edit
    result = call_llm(prompt, image_path="path/to/image.jpg")

주의:
    - Gemini 이미지 편집(call_image_edit)은 텍스트 묘사를 반환함 (이미지 바이트 아님)
    - 실제 이미지 생성은 기존 style_client.py::run_style_model() 사용 (Gemini 전용)
    - Claude/OpenAI로 전환 시 report 생성은 지원되나, 스타일 이미지 생성은 Gemini 필수
"""

import base64
import os
from pathlib import Path

try:
    from .config import (
        API_PROVIDER, GEMINI_API_KEY, CLAUDE_API_KEY, OPENAI_API_KEY,
        REPORT_MODEL, STYLE_MODEL, SELECTED_IMAGE_PATH
    )
except ImportError:
    from modules.llm_final_api.config import (
        API_PROVIDER, GEMINI_API_KEY, CLAUDE_API_KEY, OPENAI_API_KEY,
        REPORT_MODEL, STYLE_MODEL, SELECTED_IMAGE_PATH
    )


# ──────────────────────────────────────────────
# 유틸: 이미지 → base64
# ──────────────────────────────────────────────
def _image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _get_media_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")


# ──────────────────────────────────────────────
# Gemini (google-generativeai SDK)
# ──────────────────────────────────────────────
def _call_gemini(prompt: str, image_path: str = None, model: str = None) -> str:
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=GEMINI_API_KEY)
    model_name = model or REPORT_MODEL
    m = genai.GenerativeModel(model_name)

    if image_path and os.path.exists(image_path):
        img = Image.open(image_path)
        response = m.generate_content([prompt, img])
    else:
        response = m.generate_content(prompt)

    return response.text


def _call_gemini_image_edit(prompt: str, image_path: str) -> str:
    """Gemini 이미지 편집 - 텍스트 묘사 반환 (실제 이미지 바이트는 style_client.py 사용)"""
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=GEMINI_API_KEY)
    m = genai.GenerativeModel(STYLE_MODEL)
    img = Image.open(image_path)
    response = m.generate_content([
        f"다음 실내 이미지를 아래 지시에 따라 편집하세요. 편집된 결과를 상세히 묘사하세요:\n{prompt}",
        img
    ])
    return response.text


# ──────────────────────────────────────────────
# Claude (Anthropic)
# ──────────────────────────────────────────────
def _call_claude(prompt: str, image_path: str = None, model: str = None) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    model_name = model or REPORT_MODEL

    if image_path and os.path.exists(image_path):
        img_b64 = _image_to_base64(image_path)
        media_type = _get_media_type(image_path)
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_b64,
                },
            },
            {"type": "text", "text": prompt},
        ]
    else:
        content = [{"type": "text", "text": prompt}]

    message = client.messages.create(
        model=model_name,
        max_tokens=4096,
        messages=[{"role": "user", "content": content}],
    )
    return message.content[0].text


# ──────────────────────────────────────────────
# OpenAI
# ──────────────────────────────────────────────
def _call_openai(prompt: str, image_path: str = None, model: str = None) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    model_name = model or REPORT_MODEL

    if image_path and os.path.exists(image_path):
        img_b64 = _image_to_base64(image_path)
        media_type = _get_media_type(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{img_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def _call_openai_image_edit(prompt: str, image_path: str) -> str:
    """DALL-E 3 이미지 생성 - URL 반환"""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url


# ──────────────────────────────────────────────
# 통합 인터페이스
# ──────────────────────────────────────────────
def call_llm(prompt: str, image_path: str = None, model: str = None) -> str:
    """
    설정된 API_PROVIDER에 따라 자동으로 LLM을 호출합니다.

    Args:
        prompt: 텍스트 프롬프트
        image_path: 이미지 파일 경로 (선택)
        model: 특정 모델 지정 (None이면 config의 REPORT_MODEL 사용)

    Returns:
        LLM 응답 텍스트
    """
    if API_PROVIDER == "gemini":
        return _call_gemini(prompt, image_path, model)
    elif API_PROVIDER == "claude":
        return _call_claude(prompt, image_path, model)
    elif API_PROVIDER == "openai":
        return _call_openai(prompt, image_path, model)
    else:
        raise ValueError(
            f"지원하지 않는 API_PROVIDER: '{API_PROVIDER}'. "
            "'gemini', 'claude', 'openai' 중 선택하세요."
        )


def call_image_edit(prompt: str, image_path: str) -> str:
    """
    이미지 편집/생성 API 호출 (텍스트 or URL 반환).

    Returns:
        Gemini/Claude: 편집 묘사 텍스트
        OpenAI: 생성된 이미지 URL

    주의: 실제 이미지 바이트가 필요한 경우(스타일 변경 파이프라인)는
          style_client.py::run_style_model() 을 사용하세요 (Gemini 전용).
    """
    if API_PROVIDER == "gemini":
        return _call_gemini_image_edit(prompt, image_path)
    elif API_PROVIDER == "claude":
        return _call_claude(
            f"실내 인테리어 편집 전문가로서, 다음 이미지를 아래 지시에 따라 편집한 결과를 "
            f"매우 상세하게 묘사하세요 (후속 이미지 생성 모델에 전달될 프롬프트로 사용됩니다):\n{prompt}",
            image_path
        )
    elif API_PROVIDER == "openai":
        return _call_openai_image_edit(prompt, image_path)
    else:
        raise ValueError(f"지원하지 않는 API_PROVIDER: '{API_PROVIDER}'")


# ──────────────────────────────────────────────
# 직접 실행 시 연결 테스트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[테스트] API Provider: {API_PROVIDER}")
    print(f"[테스트] Report Model: {REPORT_MODEL}")
    result = call_llm("안녕하세요. API 연결 테스트입니다. 한 문장으로 답해주세요.")
    print(f"[결과] {result}")
