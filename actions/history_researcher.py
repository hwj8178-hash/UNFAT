"""
history_researcher.py — 역사학 연구 분석 핵심 모듈 (멀티 AI 파이프라인)

AI 역할 분담:
  Gemini  → 음성 인식 + 명령 처리 + 컴퓨터 제어 (main.py)
  Claude  → 로컬 문서(PDF/HWP/DOCX/이미지) 심층 분석, 각주, 연구사 정리
  Liner   → 웹 URL 기반 자료 수집 및 AI 하이라이팅

파이프라인:
  로컬 파일 → document_extractor → Claude 분석 → Obsidian 저장
  웹 URL    → Liner 웹앱 (브라우저 자동화) → Claude 병합 분석 → Obsidian 저장
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional

from .document_extractor import extract_document, ExtractedDocument
from .obsidian_bridge import (
    save_research_note, analyze_research_landscape,
    get_knowledge_summary, get_vault_path,
)


# ─── Claude/Liner 가용성 확인 ────────────────────────────────────────────────

def _claude_available() -> bool:
    try:
        from core.claude_client import is_claude_available
        return is_claude_available()
    except Exception:
        return False


def _liner_available() -> bool:
    try:
        from actions.liner_bridge import open_liner_for_manual_review  # noqa
        return True
    except Exception:
        return False


# ─── 로컬 파일 분석 (Claude 주도) ───────────────────────────────────────────

def analyze_document(
    file_path: str,
    save_to_obsidian: bool = True,
    player=None,
) -> str:
    """
    로컬 문서를 분석합니다.
    1단계: document_extractor로 텍스트·페이지 번호 추출
    2단계: Claude API로 심층 역사학 분석
    3단계: Obsidian에 결과 저장
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

    # 2. Claude 분석 시도 → 실패 시 Gemini 폴백
    analysis, ai_used = _analyze_with_best_ai(doc)

    # 3. 추출 경고 병합
    if doc.extraction_warnings:
        analysis.setdefault("uncertainty_notes", [])
        analysis["uncertainty_notes"].extend(doc.extraction_warnings)

    # 4. Obsidian 저장
    obsidian_result = ""
    if save_to_obsidian:
        try:
            obsidian_result = save_research_note(analysis)
        except Exception as e:
            obsidian_result = f"⚠️ Obsidian 저장 실패: {e}"

    return _format_voice_summary(analysis, obsidian_result, ai_used)


def _analyze_with_best_ai(doc: ExtractedDocument) -> tuple[dict, str]:
    """
    Claude → Gemini 순서로 분석을 시도합니다.
    성공한 AI 이름도 함께 반환합니다.
    """
    text_with_pages = doc.get_text_with_pages()
    knowledge_context = get_knowledge_summary()

    # Claude 우선 시도
    if _claude_available():
        try:
            from core.claude_client import analyze_document_with_claude
            analysis = analyze_document_with_claude(
                text_with_pages=text_with_pages,
                file_path=doc.file_path,
                knowledge_context=knowledge_context,
            )
            analysis["file_path"] = doc.file_path
            analysis["file_type"] = doc.file_type
            return analysis, "Claude"
        except Exception as e:
            print(f"[Research] Claude 분석 실패, Gemini로 폴백: {e}")

    # Gemini 폴백
    try:
        analysis = _analyze_with_gemini(doc, text_with_pages, knowledge_context)
        return analysis, "Gemini"
    except Exception as e:
        return _fallback_analysis(doc, str(e)), "없음(오류)"


def _analyze_with_gemini(
    doc: ExtractedDocument,
    text_with_pages: str,
    knowledge_context: str,
) -> dict:
    """Gemini 폴백 분석 (Claude 사용 불가 시)"""
    config_path = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"
    with open(config_path, "r", encoding="utf-8") as f:
        api_key = json.load(f)["gemini_api_key"]
    from google import genai

    client = genai.Client(api_key=api_key)
    system_prompt = (
        "당신은 역사학 전문 연구 보조 AI입니다. "
        "학술 논문과 역사 자료를 분석하여 핵심 주장, 논거, 방법론, 사료, "
        "연구 공백, 원문 각주(페이지 번호 필수)를 JSON으로 추출합니다."
    )
    user_prompt = _build_analysis_prompt(text_with_pages, doc.file_path, knowledge_context)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config={"system_instruction": system_prompt}
    )
    raw = response.text.strip()
    json_match = re.search(r'\{[\s\S]*\}', raw)
    result = json.loads(json_match.group() if json_match else raw)
    result["file_path"] = doc.file_path
    result["file_type"] = doc.file_type
    return result


def _build_analysis_prompt(text: str, file_path: str, knowledge_context: str) -> str:
    return f"""다음 역사학 문서를 분석해주세요. [p.숫자]가 페이지 번호입니다. ⚠️는 불확실 구간입니다.

{knowledge_context}

=== 문서 텍스트 (파일: {file_path}) ===
{text[:12000]}

JSON으로만 응답:
{{
  "title": "제목",
  "authors": ["저자"],
  "year": "연도",
  "journal": "학술지",
  "main_thesis": "핵심 주장 2-3문장",
  "key_arguments": ["논거1", "논거2"],
  "methodology": "방법론",
  "primary_sources": ["사료1"],
  "footnotes": [{{"page": 번호, "text": "원문 그대로", "context": "맥락", "uncertain": false}}],
  "keywords": ["키워드"],
  "related_works": ["관련 연구"],
  "research_gaps": ["연구 공백"],
  "liner_highlights": ["핵심 문장"],
  "uncertainty_notes": ["경고"]
}}"""


# ─── 웹 URL 분석 (Liner + Claude 협업) ──────────────────────────────────────

def analyze_url_with_liner_and_claude(
    url: str,
    question: str = "",
    save_to_obsidian: bool = True,
    player=None,
) -> str:
    """
    웹 URL 자료를 Liner로 수집한 뒤 Claude로 심층 분석합니다.

    Liner: URL 저장 + AI 요약 + 하이라이트 추출
    Claude: Liner 결과 기반 역사학적 심층 분석 + 각주 생성
    """
    # 1. Liner로 URL 분석
    from actions.liner_bridge import (
        analyze_url_with_liner, format_liner_context_for_claude,
        liner_result_to_analysis_dict
    )

    liner_result = analyze_url_with_liner(url=url, question=question, player=player)

    if not liner_result.get("success"):
        err = liner_result.get("error", "알 수 없는 오류")
        # Liner 실패 시 Liner를 수동으로 열고 Claude에게만 URL 정보 전달
        open_msg = ""
        try:
            from actions.liner_bridge import open_liner_for_manual_review
            open_liner_for_manual_review(url=url, player=player)
            open_msg = f" Liner를 열었습니다 — 직접 확인하세요."
        except Exception:
            pass
        return (
            f"⚠️ Liner 자동 분석 실패: {err}{open_msg}\n"
            f"URL: {url}\n"
            f"Liner에서 직접 이 URL을 열어 분석하거나, "
            f"파일을 다운로드 후 '파일 분석'을 요청해주세요."
        )

    liner_context = format_liner_context_for_claude(liner_result)

    # 2. Claude로 Liner 결과 심층 분석
    analysis = None
    ai_used = "Liner"

    if _claude_available() and liner_context:
        try:
            from core.claude_client import analyze_document_with_claude
            analysis = analyze_document_with_claude(
                text_with_pages=liner_context,
                file_path=url,
                knowledge_context=get_knowledge_summary(),
                liner_context="",
            )
            analysis["file_path"] = url
            analysis["file_type"] = "WEB_URL"
            ai_used = "Liner + Claude"
        except Exception as e:
            print(f"[Research] URL Claude 분석 실패: {e}")

    if analysis is None:
        analysis = liner_result_to_analysis_dict(liner_result, question)

    # 3. Obsidian 저장
    obsidian_result = ""
    if save_to_obsidian:
        try:
            obsidian_result = save_research_note(analysis)
        except Exception as e:
            obsidian_result = f"⚠️ Obsidian 저장 실패: {e}"

    return _format_voice_summary(analysis, obsidian_result, ai_used)


# ─── 각주 생성 ───────────────────────────────────────────────────────────────

def generate_footnote(file_path: str, page: int, quote_hint: str) -> str:
    """
    특정 파일의 특정 페이지에서 인용문을 찾아 각주 형식으로 반환합니다.
    Claude가 가능하면 지능적으로 검색, 불가능하면 키워드 매칭 폴백.
    """
    try:
        doc = extract_document(file_path)
    except Exception as e:
        return f"❌ 파일 추출 실패: {e}"

    target_pages = [p for p in doc.pages if p.page_number == page]
    if not target_pages:
        return f"❌ p.{page}를 찾을 수 없습니다. 총 {doc.total_pages}페이지."

    page_content = target_pages[0]

    # Claude로 지능적 각주 추출 시도
    if _claude_available():
        try:
            from core.claude_client import extract_footnote_with_claude
            return extract_footnote_with_claude(
                page_text=page_content.text,
                page_number=page,
                quote_hint=quote_hint,
                file_path=file_path,
                is_uncertain=page_content.is_uncertain,
            )
        except Exception as e:
            print(f"[Research] Claude 각주 추출 실패, 키워드 검색으로 폴백: {e}")

    # 키워드 매칭 폴백
    return _keyword_footnote_search(page_content, page, quote_hint)


def _keyword_footnote_search(page_content, page: int, quote_hint: str) -> str:
    """키워드 기반 각주 검색 (Claude 없을 때 폴백)"""
    sentences = re.split(r'(?<=[.!?。])\s+', page_content.text)
    matches = [s for s in sentences if any(w in s for w in quote_hint.split())]

    if not matches:
        if page_content.is_uncertain:
            return (
                f"⚠️ p.{page} 텍스트 추출 불확실 ({page_content.uncertainty_reason}). "
                f"원본 파일에서 직접 확인이 필요합니다."
            )
        return f"p.{page}에서 '{quote_hint}' 관련 문장을 찾지 못했습니다."

    best = max(matches, key=len)
    uncertain = " ⚠️[원본 확인 필요]" if page_content.is_uncertain else ""
    return f'각주: "{best.strip()}" (p.{page}){uncertain}'


# ─── 연구사 공백 종합 분석 (Claude 주도) ────────────────────────────────────

def find_research_gaps() -> str:
    """
    저장된 모든 논문을 바탕으로 연구사 공백을 분석합니다.
    Claude가 가능하면 지식 그래프를 종합 추론, 불가능하면 기본 통계 분석.
    """
    # 기본 통계 분석 (항상 실행)
    base_analysis = analyze_research_landscape()

    if not _claude_available():
        return base_analysis

    # Claude 심층 종합 분석
    vault = get_vault_path()
    if not vault:
        return base_analysis

    graph_path = vault / "역사학연구" / ".knowledge_graph.json"
    if not graph_path.exists():
        return base_analysis

    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        papers = [
            {"title": k, **v}
            for k, v in graph.get("nodes", {}).items()
            if v.get("type") == "paper"
        ]
        if not papers:
            return base_analysis

        from core.claude_client import synthesize_research_gaps_with_claude
        claude_synthesis = synthesize_research_gaps_with_claude(papers)

        # Claude 결과를 Obsidian에도 저장
        if vault:
            synthesis_path = vault / "역사학연구" / "Claude_연구사종합.md"
            synthesis_path.write_text(
                f"# Claude AI 연구사 종합 분석\n\n{claude_synthesis}\n\n---\n\n{base_analysis}",
                encoding="utf-8"
            )

        return f"## Claude 연구사 종합 분석\n\n{claude_synthesis}\n\n---\n\n{base_analysis}"

    except Exception as e:
        return f"⚠️ Claude 종합 분석 실패 ({e})\n\n{base_analysis}"


# ─── 응답 포맷 ───────────────────────────────────────────────────────────────

def _format_voice_summary(analysis: dict, obsidian_result: str, ai_used: str) -> str:
    """음성 비서 응답용 요약 포맷"""
    title = analysis.get("title", "제목미상")
    authors = ", ".join(analysis.get("authors", ["저자미상"]))
    year = analysis.get("year", "연도미상")
    main_thesis = analysis.get("main_thesis", "")[:200]
    gaps = analysis.get("research_gaps", [])
    warnings = analysis.get("uncertainty_notes", [])
    footnotes = analysis.get("footnotes", [])

    parts = [
        f"📚 분석 완료 [{ai_used}]: {title} ({authors}, {year})",
        f"핵심 주장: {main_thesis}",
    ]

    if footnotes:
        certain = sum(1 for f in footnotes if not f.get("uncertain"))
        uncertain = len(footnotes) - certain
        fn_note = f"{len(footnotes)}개 추출"
        if uncertain:
            fn_note += f" (이 중 {uncertain}개 ⚠️ 원본 확인 필요)"
        parts.append(f"각주 후보: {fn_note}")

    if gaps:
        parts.append(f"연구사 공백: {gaps[0][:120]}")

    if warnings:
        parts.append(f"⚠️ {len(warnings)}개 항목 원본 확인 필요")

    if obsidian_result:
        parts.append(obsidian_result)

    return "\n\n".join(parts)


def _fallback_analysis(doc: ExtractedDocument, error_msg: str) -> dict:
    return {
        "title": doc.title,
        "authors": ["⚠️ 확인 필요"],
        "year": "⚠️ 확인 필요",
        "journal": "⚠️ 확인 필요",
        "main_thesis": f"⚠️ AI 분석 실패 — 원본 확인 필요. 오류: {error_msg[:200]}",
        "key_arguments": [], "methodology": "⚠️ 확인 필요",
        "primary_sources": [], "footnotes": [],
        "keywords": [], "related_works": [], "research_gaps": [],
        "liner_highlights": [],
        "uncertainty_notes": [f"AI 분석 오류: {error_msg[:300]}"],
        "file_path": doc.file_path,
        "file_type": doc.file_type,
    }


# ─── 메인 액션 진입점 (main.py에서 호출) ─────────────────────────────────────

def history_research_action(command: str, parameters: dict, player=None) -> str:
    """
    Gemini가 도구 호출 시 진입하는 핸들러.

    지원 명령:
      analyze_document       - 로컬 파일 → Claude 분석 → Obsidian
      analyze_url            - 웹 URL → Liner → Claude → Obsidian
      find_research_gaps     - Claude 기반 연구사 공백 종합 분석
      generate_footnote      - Claude 기반 각주 생성
      set_vault_path         - Obsidian 볼트 경로 설정
    """
    from .obsidian_bridge import set_vault_path

    if command == "analyze_document":
        fp = parameters.get("file_path", "")
        if not fp:
            return "❌ 파일 경로가 필요합니다."
        return analyze_document(fp, parameters.get("save_to_obsidian", True), player=player)

    elif command == "analyze_url":
        url = parameters.get("url", "")
        if not url:
            return "❌ URL이 필요합니다."
        return analyze_url_with_liner_and_claude(
            url=url,
            question=parameters.get("question", ""),
            save_to_obsidian=parameters.get("save_to_obsidian", True),
            player=player,
        )

    elif command == "find_research_gaps":
        return find_research_gaps()

    elif command == "generate_footnote":
        fp = parameters.get("file_path", "")
        page = int(parameters.get("page", 1))
        hint = parameters.get("quote_hint", "")
        return generate_footnote(fp, page, hint)

    elif command == "set_vault_path":
        vp = parameters.get("vault_path", "")
        if not vp:
            return "❌ 볼트 경로가 필요합니다."
        return set_vault_path(vp)

    else:
        return f"❌ 알 수 없는 명령: {command}"
