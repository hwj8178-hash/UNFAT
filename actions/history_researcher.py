"""
history_researcher.py — 역사학 연구 분석 핵심 모듈

기능:
  - 학술 논문 및 역사 자료 심층 분석
  - 주요 주장, 문제의식, 방법론 추출
  - 페이지 번호 포함 원문 인용 각주 생성
  - 연구사 공백 및 새로운 문제의식 도출
  - 옵시디안 연동 저장 및 지식 그래프 업데이트
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional

from .document_extractor import extract_document, ExtractedDocument
from .obsidian_bridge import save_research_note, analyze_research_landscape, get_knowledge_summary


def _get_gemini_client():
    config_path = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"
    with open(config_path, "r", encoding="utf-8") as f:
        api_key = json.load(f)["gemini_api_key"]
    from google import genai
    return genai.Client(api_key=api_key)


def analyze_document(file_path: str, save_to_obsidian: bool = True) -> str:
    """
    문서를 분석하여 역사학 연구 관점에서 핵심 내용을 추출합니다.
    결과를 옵시디안에 저장하고 요약을 반환합니다.
    """
    # 1. 텍스트 추출
    try:
        doc = extract_document(file_path)
    except FileNotFoundError:
        return f"❌ 파일을 찾을 수 없습니다: {file_path}"
    except ValueError as e:
        return f"❌ {e}"
    except Exception as e:
        return f"❌ 문서 추출 오류: {e}"

    if not doc.full_text.strip():
        warnings = "\n".join(doc.extraction_warnings)
        return (
            f"❌ 문서에서 텍스트를 추출하지 못했습니다.\n"
            f"파일: {file_path}\n"
            f"⚠️ 자세한 확인이 필요합니다:\n{warnings}"
        )

    # 2. LLM으로 역사학적 분석
    analysis = _analyze_with_llm(doc)

    # 3. 추출 경고 추가
    if doc.extraction_warnings:
        analysis.setdefault("uncertainty_notes", [])
        analysis["uncertainty_notes"].extend(doc.extraction_warnings)

    # 4. 옵시디안 저장
    obsidian_result = ""
    if save_to_obsidian:
        obsidian_result = save_research_note(analysis)

    # 5. 음성 응답용 요약 생성
    return _format_voice_summary(analysis, obsidian_result)


def _analyze_with_llm(doc: ExtractedDocument) -> dict:
    """Gemini를 사용해 역사학 논문 분석"""
    text_with_pages = doc.get_text_with_pages()

    # 현재 지식 그래프 요약 (맥락 제공)
    knowledge_context = get_knowledge_summary()

    system_prompt = """당신은 역사학 전문 연구 보조 AI입니다.
학술 논문과 역사 자료를 분석하여 다음을 추출하는 것이 임무입니다:
1. 핵심 주장과 테제
2. 주요 논거와 근거
3. 연구 방법론과 사료
4. 기존 연구와의 관계 및 새로운 문제의식
5. 정확한 페이지 번호가 포함된 핵심 인용문

반드시 JSON 형식으로 응답하세요."""

    user_prompt = f"""다음 역사학 문서를 분석해주세요.
각 페이지의 텍스트는 [p.숫자] 형식으로 표시되어 있습니다.
⚠️ 표시가 있는 부분은 추출이 불확실합니다.

{knowledge_context}

=== 문서 텍스트 (파일: {doc.file_path}) ===
{text_with_pages[:12000]}

다음 JSON 형식으로 분석 결과를 반환하세요:
{{
  "title": "논문/자료 제목",
  "authors": ["저자1", "저자2"],
  "year": "출판연도 (숫자 또는 문자열)",
  "journal": "학술지명 또는 출처",
  "main_thesis": "핵심 주장 및 테제 (2-3문장)",
  "key_arguments": [
    "주요 논거 1",
    "주요 논거 2"
  ],
  "methodology": "연구 방법론 설명",
  "primary_sources": [
    "주요 사료 또는 참고문헌 1",
    "주요 사료 또는 참고문헌 2"
  ],
  "footnotes": [
    {{
      "page": 페이지번호,
      "text": "원문에서 그대로 추출한 핵심 인용 문장",
      "context": "이 인용문의 논문 내 맥락",
      "uncertain": false
    }}
  ],
  "keywords": ["핵심키워드1", "핵심키워드2", "핵심키워드3"],
  "related_works": ["관련 선행 연구 제목 또는 저자"],
  "research_gaps": [
    "이 연구가 드러내는 기존 연구의 공백 또는 새로운 문제의식 1",
    "새로운 연구 방향 제안 2"
  ],
  "uncertainty_notes": [
    "텍스트 추출이 불확실한 부분에 대한 경고 메시지"
  ]
}}

중요 지침:
- footnotes의 text는 반드시 원문 텍스트를 그대로 사용하세요 (요약 금지)
- 정보가 불분명할 경우 해당 필드에 "⚠️ 확인 필요" 표시
- 텍스트에 ⚠️[불확실] 표시가 있으면 uncertain: true 설정
- 페이지 번호를 반드시 포함하세요"""

    try:
        client = _get_gemini_client()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config={"system_instruction": system_prompt}
        )
        raw = response.text.strip()

        # JSON 파싱
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            analysis = json.loads(raw)

        analysis["file_path"] = doc.file_path
        analysis["file_type"] = doc.file_type
        return analysis

    except json.JSONDecodeError:
        # JSON 파싱 실패 시 기본 구조로 텍스트 저장
        return _fallback_analysis(doc, raw if "raw" in dir() else "분석 실패")
    except Exception as e:
        return _fallback_analysis(doc, str(e))


def _fallback_analysis(doc: ExtractedDocument, error_msg: str) -> dict:
    """LLM 분석 실패 시 기본 구조 반환"""
    return {
        "title": doc.title,
        "authors": ["⚠️ 확인 필요"],
        "year": "⚠️ 확인 필요",
        "journal": "⚠️ 확인 필요",
        "main_thesis": f"⚠️ AI 분석 실패 — 원본 확인 필요. 오류: {error_msg[:200]}",
        "key_arguments": [],
        "methodology": "⚠️ 확인 필요",
        "primary_sources": [],
        "footnotes": [],
        "keywords": [],
        "related_works": [],
        "research_gaps": [],
        "uncertainty_notes": [f"AI 분석 오류 발생: {error_msg[:300]}"],
        "file_path": doc.file_path,
        "file_type": doc.file_type,
    }


def _format_voice_summary(analysis: dict, obsidian_result: str) -> str:
    """음성 비서 응답용 요약 포맷"""
    title = analysis.get("title", "제목미상")
    authors = ", ".join(analysis.get("authors", ["저자미상"]))
    year = analysis.get("year", "연도미상")
    main_thesis = analysis.get("main_thesis", "")[:200]
    gaps = analysis.get("research_gaps", [])
    warnings = analysis.get("uncertainty_notes", [])
    footnotes = analysis.get("footnotes", [])

    summary_parts = [
        f"📚 분석 완료: {title} ({authors}, {year})",
        f"핵심 주장: {main_thesis}",
    ]

    if footnotes:
        summary_parts.append(f"각주 추출: {len(footnotes)}개의 원문 인용 추출 완료")

    if gaps:
        summary_parts.append(f"연구사 공백: {gaps[0][:100]}")

    if warnings:
        summary_parts.append(f"⚠️ 주의: {len(warnings)}개 항목 원본 확인 필요")

    if obsidian_result:
        summary_parts.append(obsidian_result)

    return "\n\n".join(summary_parts)


# ─── 각주 생성 도우미 ────────────────────────────────────────────────────────

def generate_footnote(file_path: str, page: int, quote_hint: str) -> str:
    """
    특정 파일의 특정 페이지에서 인용문을 찾아 각주 형식으로 반환합니다.
    quote_hint: 찾을 문장의 키워드나 일부
    """
    try:
        doc = extract_document(file_path)
    except Exception as e:
        return f"❌ 파일 추출 실패: {e}"

    target_pages = [p for p in doc.pages if p.page_number == page]
    if not target_pages:
        return f"❌ p.{page}를 찾을 수 없습니다. 총 {doc.total_pages}페이지."

    page_content = target_pages[0]
    text = page_content.text

    # 힌트로 관련 문장 찾기
    sentences = re.split(r'(?<=[.!?。])\s+', text)
    matches = [s for s in sentences if any(word in s for word in quote_hint.split())]

    if not matches:
        if page_content.is_uncertain:
            return (
                f"⚠️ p.{page} 텍스트 추출 불확실 ({page_content.uncertainty_reason}). "
                f"원본 파일에서 직접 확인이 필요합니다."
            )
        return f"p.{page}에서 '{quote_hint}' 관련 문장을 찾지 못했습니다."

    best_match = max(matches, key=len)
    uncertain_note = ""
    if page_content.is_uncertain:
        uncertain_note = " ⚠️[원본 확인 필요]"

    return f'각주: "{best_match.strip()}" (p.{page}){uncertain_note}'


# ─── 연구사 공백 분석 래퍼 ──────────────────────────────────────────────────

def find_research_gaps() -> str:
    """저장된 모든 논문을 바탕으로 연구사 공백을 분석합니다."""
    return analyze_research_landscape()


# ─── 메인 액션 진입점 (main.py에서 호출) ─────────────────────────────────────

def history_research_action(command: str, parameters: dict) -> str:
    """
    음성 비서에서 호출되는 역사학 연구 액션 핸들러.

    지원 명령:
      analyze_document    - 문서 분석 및 옵시디안 저장
      find_research_gaps  - 연구사 공백 분석
      generate_footnote   - 특정 페이지 각주 생성
      set_vault_path      - 옵시디안 볼트 경로 설정
    """
    from .obsidian_bridge import set_vault_path

    if command == "analyze_document":
        file_path = parameters.get("file_path", "")
        if not file_path:
            return "❌ 파일 경로가 필요합니다."
        save = parameters.get("save_to_obsidian", True)
        return analyze_document(file_path, save_to_obsidian=save)

    elif command == "find_research_gaps":
        return find_research_gaps()

    elif command == "generate_footnote":
        file_path = parameters.get("file_path", "")
        page = int(parameters.get("page", 1))
        quote_hint = parameters.get("quote_hint", "")
        return generate_footnote(file_path, page, quote_hint)

    elif command == "set_vault_path":
        vault_path = parameters.get("vault_path", "")
        if not vault_path:
            return "❌ 볼트 경로가 필요합니다."
        return set_vault_path(vault_path)

    else:
        return f"❌ 알 수 없는 명령: {command}"
