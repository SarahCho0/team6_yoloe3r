import re
from typing import Dict, Any, List, Union

def parse_report_output(result_text: str) -> Dict[str, Union[str, Dict, List]]:
    llm_output = result_text
    parsed_data: Dict[str, Any] = {}

    # --- # 전체적인 분위기 한 줄 ---
    # 패턴: # 전체적인 분위기는 **{...} 스타일**입니다.
    match_style = re.search(
        r"#\s*전체적인 분위기는\s*\*\*(.*?)\s*스타일\*\*",
        llm_output,
        re.DOTALL, 
    )

    if match_style: 
        general = match_style.group(1).strip()
        parsed_data["general_style"] = general

        # {분위기1}, {분위기2} ,{분위기3} 추출
        moods = re.findall(r"([가-힣\s]+?)(?:하고|한|\s*$)", general)
        parsed_data["mood_words"] = [m.strip() for m in moods if m.strip()]

    # --- ## 1. 분위기 정의 및 유형별 확률 ---
    # 섹션 탐지: 다음 헤딩인 '## 2.'까지
    mood_section_match = re.search(
        r"##\s*1\. 분위기 정의 및 유형별 확률(.*?)(?=##\s*2\. 분위기 판단 근거)",
        llm_output,
        re.DOTALL,
    )

    if mood_section_match:
        mood_section = mood_section_match.group(1).strip()

        # 패턴: - **{"{분위기}"}({확률}%)**: \n {설명}
        PATTERN_MOOD_DETAIL = re.compile(
            # - **{"{분위기}"}({확률}%)**: \s*\n\s*{설명}
            r'-\s*\*\*\{\"([^\"]+)\"\}\s*\((\d+)%\)\*\*\s*:\s*\n\s*(.+?)',
            re.DOTALL
        )

        mood_matches = PATTERN_MOOD_DETAIL.findall(mood_section)

        parsed_data["mood_details"] = []

        for mood, pct, desc in mood_matches:
            parsed_data["mood_details"].append(
                {
                    "word": mood.strip(),
                    "percentage": int(pct),
                    "description": desc.strip(),
                }
            )

    
    # --- ## 2. 분위기 판단 근거 ---
    # 섹션 탐지: 다음 헤딩인 '## 3.'까지
    basis_section_match = re.search(
        r"##\s*2\. 분위기 판단 근거(.*?)(?=##\s*3\. 가구 추가 / 제거 / 변경 추천)",
        llm_output,
        re.DOTALL,
    )
    if basis_section_match:
        basis_section = basis_section_match.group(1).strip()

        # 패턴: - **{키}** : \n {값}
        PATTERN_BASIS = re.compile(
            r"-\s*\*\*(.*?)\*\*\s*:\s*\n\s*(.+?)",
            re.DOTALL
        )
        basis_matches = PATTERN_BASIS.findall(basis_section)

        parsed_data["basis"] = {}
        key_mapping = {
            "가구 배치 및 공간 분석": "furniture_layout",
            "색감 및 질감": "color_texture",
            "소재": "material",
        }

        for key, value in basis_matches:
            k = key.strip()
            v = value.strip()
            if k in key_mapping:
                parsed_data["basis"][key_mapping[k]] = v
            else:
                parsed_data["basis"][k] = v

    # --- ## 3. 가구 추가 / 제거 / 변경 추천 (복합 섹션) ---
    # 섹션 탐지: 다음 헤딩인 '## 4.'까지
    rec_section_match = re.search(
        r"##\s*3\. 가구 추가 / 제거 / 변경 추천(.*?)(?=##\s*4\. 이런 스타일 어떠세요\?)",
        llm_output,
        re.DOTALL,
    )
    
    if rec_section_match:
        rec_section = rec_section_match.group(1).strip()
        
        # 3-1: 추가 추천: 3-1 **제목**\n- **가구** :\n근거
        add_match = re.search(
            r"3-1\s*\*\*현재 분위기에 맞춰 추가하면 좋을 가구 추천\*\*\s*\n-\s*\*\*(.*?)\*\*\s*:\s*\n\s*(.*?)",
            rec_section,
            re.DOTALL
        )
        if add_match:
            item, reason = add_match.groups()
            parsed_data["recommendations_add"] = [
                {"item": item.strip(), "reason": reason.strip()}
            ]
        else:
            parsed_data["recommendations_add"] = []

        # 3-2: 제거 추천: 3-2 **제목**\n- **가구** :\n근거
        rem_match = re.search(
            r"3-2\s*\*\*제거하면 좋을 가구 추천\*\*\s*\n-\s*\*\*(.*?)\*\*\s*:\s*\n\s*(.*?)",
            rec_section,
            re.DOTALL
        )
        if rem_match:
            item, reason = rem_match.groups()
            parsed_data["recommendations_remove"] = [
                {"item": item.strip(), "reason": reason.strip()}
            ]
        else:
            parsed_data["recommendations_remove"] = []

        # 3-3: 변경 추천: 3-3 **제목**\n- **변경 -> 추천** :\n근거
        change_match = re.search(
            r"3-3\s*\*\*분위기별 바꿨으면 하는 가구 추천\*\*\s*\n-\s*\*\*(.*?)\s*->\s*(.*?)\*\*\s*:\s*\n\s*(.*)",
            rec_section,
            re.DOTALL
        )
        if change_match:
            src, dst, reason = change_match.groups()
            parsed_data["recommendations_change"] = [
                {
                    "from_item": src.strip(),
                    "to_item": dst.strip(),
                    "reason": reason.strip()
                }
            ]
        else:
            parsed_data["recommendations_change"] = []

    # --- ## 4. 이런 스타일 어떠세요? ---
    # 섹션 탐지: 다음 헤딩인 '## 정리'까지
    section_pattern = re.compile(
        r"^##\s*4\.\s*이런 스타일 어떠세요\?\s*$"
        r"(?P<body>.*?)(?=^##\s*정리|\Z)", 
        re.MULTILINE | re.DOTALL,
    )

    m = section_pattern.search(llm_output)
    
    parsed_data["recommended_styles"] = []
    
    if m:
        body = m.group("body").strip()
        
        if body:
            # 패턴: **{스타일}** :\n {이유}
            bullet_pattern = re.compile(
                r"^\s*\*\*(?P<style>[^:]+?)\*\*\s*:\s*\n\s*(?P<reason>.+)$",
                re.MULTILINE | re.DOTALL,
            )

            for b in bullet_pattern.finditer(body):
                style = b.group("style").strip()
                reason = b.group("reason").strip()
                parsed_data["recommended_styles"].append(
                    {
                        "style": style,
                        "reason": reason,
                    }
                )

    # --- ## 정리 ---
    sum_section_match = re.search(r"##\s*정리(.*)", llm_output, re.DOTALL)
    if sum_section_match:
        sum_section = sum_section_match.group(1)

        # 패턴: - {문장}
        lines = re.findall(r"-\s*(.*)", sum_section)

        parsed_data["summary"] = {}
        for idx, sentence in enumerate(lines):
            key = f"summary{idx + 1}"
            parsed_data["summary"][key] = sentence.strip()

    return parsed_data
