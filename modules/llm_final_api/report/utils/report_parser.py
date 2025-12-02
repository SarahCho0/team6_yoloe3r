import re
from typing import Dict, Any, List, Union

def parse_report_output(result_text: str) -> Dict[str, Union[str, Dict, List]]:
    llm_output = result_text
    parsed_data: Dict[str, Any] = {}

    # ------ 전체적인 분위기 한 줄 ------
    match_style = re.search(
        r"#\s*전체적인 분위기는\s*\*\*(.*?)\s*스타일\*\*",
        llm_output,
        re.DOTALL, 
    )

    # 일치하는 경우에만 처리
    if match_style: 
        general = match_style.group(1).strip()
        parsed_data["general_style"] = general

        # {분위기1}, {분위기2} ,{분위기3} 추출 
        moods = re.findall(r"([가-힣\s]+?)(?:하고|한|\s*$)", general)
        parsed_data["mood_words"] = [m.strip() for m in moods if m.strip()] # 추출된 단어 리스트는 'mood_words' 저장

    # ------ ## 1. 분위기 정의 및 유형별 확률 ------
    mood_section_match = re.search(
        r"##\s*1\. 분위기 정의 및 유형별 확률(.*?)(?=##\s*2\. 분위기 판단 근거)",
        llm_output,
        re.DOTALL,
    )

    if mood_section_match:
        mood_section = mood_section_match.group(1)

        # {분위기}({확률}%): {설명} 패턴을 정의
        PATTERN_MOOD_DETAIL = r"-\s*(.*?)\s*\((\d+)%\):\s*(.*)"

        # PATTERN_MOOD_DETAIL에 매칭되는 항목을 찾아 리스트로 변환
        mood_matches = re.findall(PATTERN_MOOD_DETAIL, mood_section)

        parsed_data["mood_details"] = []

        # 찾은 모든 항목(튜플)을 순회하며, 구조화
        for mood, pct, desc in mood_matches:
            parsed_data["mood_details"].append(
                {
                    "word": mood.strip(),
                    "percentage": int(pct),
                    "description": desc.strip(),
                }
            )

    
    # ------ ## 2. 분위기 판단 근거 -------
    basis_section_match = re.search(
        r"##\s*2\. 분위기 판단 근거(.*?)(?=##\s*3-1\. 현재 분위기에 맞춰 추가하면 좋을 가구 추천)",
        llm_output,
        re.DOTALL,
    )
    if basis_section_match:
        basis_section = basis_section_match.group(1)

        PATTERN_BASIS = r"-\s*(.*?):\s*(.*)"
        basis_matches = re.findall(PATTERN_BASIS, basis_section)

        parsed_data["basis"] = {}
        key_mapping = {
            "가구 배치 및 공간 분석": "furniture_layout",
            "색감 및 질감": "color_texture",
            "소재": "material",
        }

        # 찾은 모든 (key, value) 쌍
        for key, value in basis_matches:
            k = key.strip()
            v = value.strip()
            if k in key_mapping:
                parsed_data["basis"][key_mapping[k]] = v
            else:
                # 매핑에 없는 키는 원문 그대로도 보존
                parsed_data["basis"][k] = v

    # ------ ## 3-1. 현재 분위기에 맞춰 추가하면 좋을 가구 추천 ------
    add_section_match = re.search(
        r"##\s*3-1\. 현재 분위기에 맞춰 추가하면 좋을 가구 추천(.*?)(?=##\s*3-2\. 제거하면 좋을 가구 추천)",
        llm_output,
        re.DOTALL,
    )
    if add_section_match:
        add_section = add_section_match.group(1)

        # {추가 가구} : {근거}
        PATTERN_ADD = r"-\s*(.*?):\s*(.*)"
        add_matches = re.findall(PATTERN_ADD, add_section)

        parsed_data["recommendations_add"] = []

        # 찾은 항목(튜플)을 구조화
        for item, reason in add_matches:
            parsed_data["recommendations_add"].append(
                {
                    "item": item.strip(),
                    "reason": reason.strip(),
                }
            )

    # ------ ## 3-2. 제거하면 좋을 가구 추천 ------
    rem_section_match = re.search(
        r"##\s*3-2\. 제거하면 좋을 가구 추천(.*?)(?=##\s*3-3\. 분위기별 바꿨으면 하는 가구 추천)",
        llm_output,
        re.DOTALL,
    )
    if rem_section_match:
        rem_section = rem_section_match.group(1)

        # {제거 가구} : {근거}
        PATTERN_REM = r"-\s*(.*?):\s*(.*)"
        rem_matches = re.findall(PATTERN_REM, rem_section)

        # 템플릿 상 한 줄만 나오지만, 혹시 모를 확장을 고려해 리스트로 저장
        parsed_data["recommendations_remove"] = []
        for item, reason in rem_matches:
            parsed_data["recommendations_remove"].append(
                {
                    "item": item.strip(),
                    "reason": reason.strip(),
                }
            )

    # ------ ## 3-3. 분위기별 바꿨으면 하는 가구 추천 ------
    change_section_match = re.search(
        r"##\s*3-3\. 분위기별 바꿨으면 하는 가구 추천(.*?)(?=#\s*6\. 이런 스타일 어떠세요\?|##\s*정리|$)",
        llm_output,
        re.DOTALL,
    )
    if change_section_match:
        change_section = change_section_match.group(1)

        # {변경 가구} -> {추천 가구} : {근거}
        PATTERN_CHANGE = r"-\s*(.*?)\s*->\s*(.*?)\s*:\s*(.*)"
        change_matches = re.findall(PATTERN_CHANGE, change_section)

        parsed_data["recommendations_change"] = []
        for src, dst, reason in change_matches:
            parsed_data["recommendations_change"].append(
                {
                    "from_item": src.strip(),
                    "to_item": dst.strip(),
                    "reason": reason.strip(),
                }
            )

    # ------ ## 4. 이런 스타일 어떠세요? ------
    section_pattern = re.compile(
        r"^##\s*4\.\s*이런 스타일 어떠세요\?\s*$"
        r"(?P<body>.*?)(?=^##\s*정리|\Z)", 
        re.MULTILINE | re.DOTALL,
    )

    m = section_pattern.search(llm_output)
    
    parsed_data["recommended_styles"] = [] # 결과 리스트 초기화
    
    if m:
        body = m.group("body").strip()
        
        if body:
            # "{- 스타일} : {이유}" 형식 한 줄씩 파싱
            bullet_pattern = re.compile(
                r"^\s*-\s*(?P<style>[^:]+?)\s*:\s*(?P<reason>.+)$",
                re.MULTILINE,
            )

            # 본문(body) 내에서 모든 일치 항목을 반복해서 찾음
            for b in bullet_pattern.finditer(body):
                style = b.group("style").strip()
                reason = b.group("reason").strip()
                parsed_data["recommended_styles"].append(
                    {
                        "style": style,
                        "reason": reason,
                    }
                )

    # ------ 정리 ------
    sum_section_match = re.search(r"##\s*정리(.*)", llm_output, re.DOTALL)
    if sum_section_match:
        sum_section = sum_section_match.group(1)

        # " {- 문장}" 형태의 모든 줄을 뽑아온다.
        lines = re.findall(r"-\s*(.*)", sum_section)

        parsed_data["summary"] = {}
        for idx, sentence in enumerate(lines):
            key = f"summary{idx + 1}"
            parsed_data["summary"][key] = sentence.strip()

    # 최종 결과 반환
    return parsed_data


