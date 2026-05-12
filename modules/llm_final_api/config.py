"""
IN3R Multi-API Config
지원 API: Gemini, Claude (Anthropic), OpenAI

사용할 API를 API_PROVIDER 변수 한 줄만 바꿔서 전환 가능.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# ✅ 사용할 API 선택: "gemini" | "claude" | "openai"
# ============================================================
API_PROVIDER = "gemini"

# ============================================================
# API 키 설정 (사용할 provider의 키만 입력)
# .env 파일에 GEMINI_API_KEY="..." 형태로 저장하거나 아래에 직접 입력
# ============================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "여기에 Gemini API 키 입력")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "여기에 Claude API 키 입력")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "여기에 OpenAI API 키 입력")

# 기존 코드 호환성: API_KEY = 현재 선택된 provider의 키
_KEY_MAP = {
    "gemini": GEMINI_API_KEY,
    "claude": CLAUDE_API_KEY,
    "openai": OPENAI_API_KEY,
}
API_KEY = _KEY_MAP.get(API_PROVIDER, GEMINI_API_KEY)

if not API_KEY or API_KEY.startswith("여기에"):
    print("🚨 경고: API 키가 설정되지 않았습니다. config.py 또는 .env 파일을 확인하세요.")

# ============================================================
# 모델 설정 (provider별)
# ============================================================
MODEL_CONFIG = {
    "gemini": {
        # gemini-2.5-flash-image: 이미지 생성/편집 지원 모델 (style_client.py 필수)
        "report_model": "gemini-2.5-flash",
        "style_model":  "gemini-2.5-flash-image",
    },
    "claude": {
        "report_model": "claude-opus-4-6",
        "style_model":  "claude-opus-4-6",
    },
    "openai": {
        "report_model": "gpt-4o",
        "style_model":  "gpt-4o",
    },
}

REPORT_MODEL = MODEL_CONFIG[API_PROVIDER]["report_model"]
STYLE_MODEL  = MODEL_CONFIG[API_PROVIDER]["style_model"]

# ============================================================
# 이미지 경로 설정
# ============================================================
INITIAL_IMAGE_PATHS = [
    "modules/llm_final_api/input_image/test1.png",
    "modules/llm_final_api/input_image/test2.png",
    "modules/llm_final_api/input_image/test3.png",
]

# 3장 중 AI가 선택한 최적 이미지가 저장될 경로
# 이후 Report, Style 모든 단계에서 이 경로를 사용
SELECTED_IMAGE_PATH = "selected_input_image.jpg"
>>>>>>> 28294bf (feat: 유진홈센터 경량화 버전 - LLM 제거, VGGT 재구성 추가)
