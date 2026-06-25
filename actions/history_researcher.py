"""
history_researcher.py — 역사학 연구 분석 핵심 모듈 v2 (멀티 AI 파이프라인)

AI 역할 분담:
  Gemini  → 음성 인식 + 명령 처리 + 컴퓨터 제어 (main.py)
  Claude  → 로컬 문서(PDF/HWP/DOCX/이미지) 심층 분석, 각주, 연구사 정리
  Liner   → 웹 URL 기반 자료 수집 및 AI 하이라이팅

v2 신기능:
  - 목차 기반 장/절/페이지 정확 출처 (제2장 3절 p.45)
  - 미국 자료 날짜·기관·수신발신자·기밀등급 특화 추출
  - document_type 자동 감지 및 유형별 음성 요약 포맷
  - 각주에 chapter/section/location_string 귀속
  - ExtractedDocument 구조 정보를 Claude 프롬프트에 전달
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
    record_group: str = "",
    entry: str = "",
    box: str = "",
    folder: str = "",
) -> str:
    """
    로컬 문서를 분석합니다.
    1단계: document_extractor로 텍스트·페이지·목차·날짜 추출
    2단계: Claude API로 심층 역사학 분석 (장/절 귀속 포함)
    3단계: Obsidian에 결과 저장 (NARA RG/Entry 포함)

    record_group: NARA Record Group 번호 (예: "59", "RG 59")
    entry:        NARA Entry 번호 (예: "1234", "A1 1234")
    box:          Box 번호
    folder:       Folder명
    """
    def _log(msg: str):
        print(f"[Research] {msg}")
        if player and hasattr(player, "write_log"):
            player.write_log(f"SYS: {msg}")

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

    # 구조 감지 결과 로그
    if doc.toc:
        _log(f"목차 감지: {len(doc.toc)}개 항목")
    if doc.document_date:
        _log(f"날짜 감지: {doc.document_date}")
    if doc.document_language:
        _log(f"언어 감지: {doc.document_language}")
    if record_group or entry:
        rg_str = f"RG {record_group}" if record_group else ""
        en_str = f"Entry {entry}" if entry else ""
        _log(f"아카이브 식별자: {' / '.join(filter(None, [rg_str, en_str]))}")

    # 2. Claude 분석 시도 → 실패 시 Gemini 폴백
    analysis, ai_used = _analyze_with_best_ai(doc, _log)

    # 3. 추출 경고 병합
    if doc.extraction_warnings:
        analysis.setdefault("uncertainty_notes", [])
        analysis["uncertainty_notes"].extend(doc.extraction_warnings)

    # 4. NARA 아카이브 식별자 삽입 (사용자가 제공한 경우)
    archive_info = _build_archive_info(record_group, entry, box, folder)
    if archive_info:
        analysis["archive_info"] = archive_info

    # 5. Obsidian 저장
    obsidian_result = ""
    if save_to_obsidian:
        try:
            obsidian_result = save_research_note(analysis)
        except Exception as e:
            obsidian_result = f"⚠️ Obsidian 저장 실패: {e}"

    return _format_voice_summary(analysis, obsidian_result, ai_used)


def _analyze_with_best_ai(doc: ExtractedDocument, log_fn=None) -> tuple[dict, str]:
    """
    Claude → Gemini 순서로 분석을 시도합니다.
    대용량 문서는 자동으로 청크 분할 후 종합합니다.
    v2: ExtractedDocument의 구조 정보(toc, date, language)를 전달합니다.
    """
    def _log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(f"[Research] {msg}")

    knowledge_context = get_knowledge_summary()
    is_large = doc.total_chars > 160_000

    # v2: 구조화 텍스트 사용 (목차+장절 헤더 포함)
    text_for_analysis = doc.get_text_with_structure()

    # Claude 우선 시도
    if _claude_available():
        try:
            from core.claude_client import (
                analyze_document_with_claude,
                analyze_large_document_with_claude,
                SINGLE_PASS_CHARS,
            )

            common_kwargs = dict(
                knowledge_context=knowledge_context,
                first_pages_text=doc.first_pages_text,
                document_date=doc.document_date,
                document_language=doc.document_language,
                toc_text=doc.toc_text,
            )

            if is_large:
                _log(f"대용량 문서 ({doc.total_chars:,}자) — 청킹 분석 시작")
                analysis = analyze_large_document_with_claude(
                    text_with_pages=text_for_analysis,
                    file_path=doc.file_path,
                    **common_kwargs,
                )
                ai_label = f"Claude (청크×{analysis.get('_chunk_count', '?')})"
            else:
                analysis = analyze_document_with_claude(
                    text_with_pages=text_for_analysis,
                    file_path=doc.file_path,
                    **common_kwargs,
                )
                ai_label = "Claude"

            analysis["file_path"] = doc.file_path
            analysis["file_type"] = doc.file_type

            # 목차 정보가 Claude 응답에 없으면 추출기 TOC 사용
            if not analysis.get("toc") and doc.toc:
                analysis["toc"] = [
                    {"level": e.level, "number": e.number,
                     "title": e.title, "page": e.page_number}
                    for e in doc.toc
                ]

            return analysis, ai_label

        except Exception as e:
            _log(f"Claude 분석 실패, Gemini로 폴백: {e}")

    # Gemini 폴백
    try:
        analysis = _analyze_with_gemini(doc, text_for_analysis, knowledge_context)
        return analysis, "Gemini"
    except Exception as e:
        return _fallback_analysis(doc, str(e)), "없음(오류)"


def _analyze_with_gemini(
    doc: ExtractedDocument,
    text_with_structure: str,
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
        "연구 공백, 원문 각주(페이지·장·절 번호 필수)를 JSON으로 추출합니다."
    )
    user_prompt = _build_gemini_prompt(text_with_structure, doc.file_path, knowledge_context, doc)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config={"system_instruction": system_prompt}
    )
    raw = response.text.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```\s*$', '', raw, flags=re.MULTILINE)
    json_match = re.search(r'\{[\s\S]*\}', raw)
    result = json.loads(json_match.group() if json_match else raw)
    result["file_path"] = doc.file_path
    result["file_type"] = doc.file_type
    return result


def _build_gemini_prompt(
    text: str, file_path: str, knowledge_context: str, doc: ExtractedDocument
) -> str:
    toc_section = f"\n{doc.toc_text}\n" if doc.toc_text else ""
    date_hint = f"\n감지된 날짜: {doc.document_date}" if doc.document_date else ""
    lang_hint = f"\n언어: {doc.document_language}" if doc.document_language else ""

    return f"""다음 역사학 문서를 분석해주세요. [p.숫자]가 페이지 번호입니다. ⚠️는 불확실 구간입니다.
{date_hint}{lang_hint}{toc_section}

{knowledge_context}

=== 문서 텍스트 (파일: {file_path}) ===
{text[:100_000]}

JSON으로만 응답:
{{
  "title": "제목",
  "authors": ["저자"],
  "year": "연도",
  "document_date": "날짜 (YYYY-MM-DD, 미국 자료 우선)",
  "document_type": "academic_paper|monograph|newspaper_article|government_document|other",
  "journal_or_source": "학술지명/신문명/기관명",
  "publisher": "출판사",
  "issuing_body": "발행기관 (미국 자료)",
  "historical_period": "역사적 시기",
  "geographical_scope": "지리적 범위",
  "main_thesis": "핵심 주장 2-3문장",
  "key_arguments": ["논거1 (p.N 근거)", "논거2"],
  "key_events": ["주요 사건 (날짜·장소 포함)"],
  "key_persons": ["주요 인물 (직책 포함)"],
  "methodology": "방법론",
  "primary_sources": ["사료1"],
  "toc": [{{"level": 레벨, "number": "제1장", "title": "제목", "page": 페이지}}],
  "footnotes": [{{
    "page": 번호,
    "chapter": "제N장 제목",
    "section": "제N절 제목",
    "location_string": "제N장 > 제N절 > p.N",
    "text": "원문 그대로",
    "context": "맥락",
    "uncertain": false
  }}],
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
    """웹 URL 자료를 Liner로 수집한 뒤 Claude로 심층 분석합니다."""
    from actions.liner_bridge import (
        analyze_url_with_liner, format_liner_context_for_claude,
        liner_result_to_analysis_dict
    )

    liner_result = analyze_url_with_liner(url=url, question=question, player=player)

    if not liner_result.get("success"):
        err = liner_result.get("error", "알 수 없는 오류")
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

    obsidian_result = ""
    if save_to_obsidian:
        try:
            obsidian_result = save_research_note(analysis)
        except Exception as e:
            obsidian_result = f"⚠️ Obsidian 저장 실패: {e}"

    return _format_voice_summary(analysis, obsidian_result, ai_used)


# ─── 각주 생성 ───────────────────────────────────────────────────────────────

def generate_footnote(file_path: str, page: int, quote_hint: str) -> str:
    """특정 파일의 특정 페이지에서 인용문을 찾아 각주 형식으로 반환합니다."""
    try:
        doc = extract_document(file_path)
    except Exception as e:
        return f"❌ 파일 추출 실패: {e}"

    target_pages = [p for p in doc.pages if p.page_number == page]
    if not target_pages:
        return f"❌ p.{page}를 찾을 수 없습니다. 총 {doc.total_pages}페이지."

    page_content = target_pages[0]
    chapter_info = doc.get_chapter_for_page(page)
    location_str = doc.get_location_string(page)

    if _claude_available():
        try:
            from core.claude_client import extract_footnote_with_claude
            return extract_footnote_with_claude(
                page_text=page_content.text,
                page_number=page,
                quote_hint=quote_hint,
                file_path=file_path,
                is_uncertain=page_content.is_uncertain,
                chapter_info=chapter_info,
            )
        except Exception as e:
            print(f"[Research] Claude 각주 추출 실패, 키워드 검색으로 폴백: {e}")

    return _keyword_footnote_search(page_content, page, quote_hint, location_str)


def _keyword_footnote_search(
    page_content, page: int, quote_hint: str, location_str: str
) -> str:
    """키워드 기반 각주 검색 (Claude 없을 때 폴백)"""
    sentences = re.split(r'(?<=[.!?。])\s+', page_content.text)
    matches = [s for s in sentences if any(w in s for w in quote_hint.split())]

    if not matches:
        if page_content.is_uncertain:
            return (
                f"⚠️ {location_str} 텍스트 추출 불확실 ({page_content.uncertainty_reason}). "
                f"원본 파일에서 직접 확인이 필요합니다."
            )
        return f"{location_str}에서 '{quote_hint}' 관련 문장을 찾지 못했습니다."

    best = max(matches, key=len)
    uncertain = " ⚠️[원본 확인 필요]" if page_content.is_uncertain else ""
    return f'각주: "{best.strip()}" ({location_str}){uncertain}'


# ─── 연구사 공백 종합 분석 (Claude 주도) ─────────────────────────────────────

def find_research_gaps() -> str:
    """저장된 모든 논문을 바탕으로 연구사 공백을 분석합니다."""
    base_analysis = analyze_research_landscape()

    if not _claude_available():
        return base_analysis

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
    """
    음성 비서 응답용 요약 포맷 (v2: 문서 유형별 특화 출력)
    """
    title       = analysis.get("title", "제목미상")
    subtitle    = analysis.get("subtitle", "")
    authors     = ", ".join(analysis.get("authors", ["저자미상"]))
    year        = analysis.get("year", "")
    doc_date    = analysis.get("document_date", "")
    doc_type    = analysis.get("document_type", "")
    source      = analysis.get("journal_or_source", "")
    publisher   = analysis.get("publisher", "")
    pub_place   = analysis.get("publication_place", "")
    issuing     = analysis.get("issuing_body", "")
    recipients  = analysis.get("recipients", [])
    senders     = analysis.get("senders", [])
    classif     = analysis.get("classification", "")
    hist_period = analysis.get("historical_period", "")
    geo_scope   = analysis.get("geographical_scope", "")
    main_thesis = analysis.get("main_thesis", "")[:300]
    doc_struct  = analysis.get("document_structure", "")
    key_events  = analysis.get("key_events", [])
    key_persons = analysis.get("key_persons", [])
    gaps        = analysis.get("research_gaps", [])
    warnings    = analysis.get("uncertainty_notes", [])
    footnotes   = analysis.get("footnotes", [])
    toc         = analysis.get("toc", [])

    parts: list[str] = []

    archive_info = analysis.get("archive_info", {})

    # ── 헤더: 문서 유형에 따라 아이콘 구분 ──────────────────────────────────
    TYPE_ICON = {
        "academic_paper":       "📄",
        "monograph":            "📚",
        "newspaper_article":    "📰",
        "government_document":  "🏛️",
        "diplomatic_cable":     "📡",
        "memorandum":           "📋",
        "report":               "📊",
        "letter":               "✉️",
        "testimony":            "🗣️",
    }
    icon = TYPE_ICON.get(doc_type, "📁")
    full_title = f"{title}{(' — ' + subtitle) if subtitle else ''}"
    parts.append(f"{icon} 분석 완료 [{ai_used}]: {full_title}")

    # ── NARA 아카이브 출처 블록 (최우선 표시) ────────────────────────────────
    if archive_info:
        arc_parts = []
        if archive_info.get("record_group"):
            arc_parts.append(archive_info["record_group"])
        if archive_info.get("entry"):
            arc_parts.append(archive_info["entry"])
        if archive_info.get("box"):
            arc_parts.append(archive_info["box"])
        if archive_info.get("folder"):
            arc_parts.append(f"Folder: {archive_info['folder']}")
        parts.append(f"🏛️ NARA 출처: {' / '.join(arc_parts)}")

    # ── 서지사항 블록 ────────────────────────────────────────────────────────
    bib_parts: list[str] = []

    if doc_date:
        bib_parts.append(f"날짜: {_format_date_ko(doc_date)}")
    elif year:
        bib_parts.append(f"연도: {year}")

    if doc_type:
        TYPE_KO = {
            "academic_paper": "학술논문", "monograph": "단행본",
            "newspaper_article": "신문기사", "government_document": "정부문서",
            "diplomatic_cable": "외교전문", "memorandum": "메모/각서",
            "report": "보고서", "letter": "서한", "testimony": "증언",
        }
        bib_parts.append(f"유형: {TYPE_KO.get(doc_type, doc_type)}")

    if authors and authors != "저자미상":
        bib_parts.append(f"저자: {authors}")

    if issuing:
        bib_parts.append(f"발행: {issuing}")
    elif source:
        bib_parts.append(f"출처: {source}")
    elif publisher:
        bib_parts.append(f"출판: {publisher}")

    if pub_place:
        bib_parts.append(f"발행지: {pub_place}")

    if senders:
        bib_parts.append(f"발신: {', '.join(senders[:3])}")
    if recipients:
        bib_parts.append(f"수신: {', '.join(recipients[:3])}")
    if classif:
        bib_parts.append(f"기밀: {classif}")

    if hist_period:
        bib_parts.append(f"시기: {hist_period}")
    if geo_scope:
        bib_parts.append(f"지역: {geo_scope}")

    if bib_parts:
        parts.append(" | ".join(bib_parts))

    # ── 문서 구조 (목차) ─────────────────────────────────────────────────────
    if toc:
        toc_lines = []
        for entry in toc[:12]:  # 최대 12개만 표시
            indent = "  " * (entry.get("level", 1) - 1)
            num    = entry.get("number", "")
            ttl    = entry.get("title", "")
            pg     = entry.get("page", 0)
            pg_str = f" → p.{pg}" if pg > 0 else ""
            toc_lines.append(f"{indent}{num} {ttl}{pg_str}")
        if toc_lines:
            toc_str = "\n".join(toc_lines)
            parts.append(f"목차 구조:\n{toc_str}")
    elif doc_struct:
        parts.append(f"구조: {doc_struct[:200]}")

    # ── 핵심 내용 ────────────────────────────────────────────────────────────
    if main_thesis:
        parts.append(f"핵심 내용: {main_thesis}")

    # ── 미국/영어 자료 특화: 주요 사건·인물 ──────────────────────────────────
    if key_events:
        events_str = " / ".join(str(e)[:80] for e in key_events[:4])
        parts.append(f"주요 사건: {events_str}")
    if key_persons:
        persons_str = " / ".join(str(p)[:60] for p in key_persons[:5])
        parts.append(f"주요 인물: {persons_str}")

    # ── 각주 요약 (장/절 귀속 포함) ──────────────────────────────────────────
    if footnotes:
        certain_fn   = [f for f in footnotes if not f.get("uncertain")]
        uncertain_fn = [f for f in footnotes if f.get("uncertain")]

        fn_count_str = f"{len(footnotes)}개 (확실 {len(certain_fn)}개"
        if uncertain_fn:
            fn_count_str += f", ⚠️ 확인필요 {len(uncertain_fn)}개"
        fn_count_str += ")"
        parts.append(f"인용 추출: {fn_count_str}")

        # 장별 분포
        chapter_dist: dict[str, int] = {}
        for fn in footnotes:
            ch = fn.get("chapter", "") or fn.get("section", "") or "미분류"
            chapter_dist[ch] = chapter_dist.get(ch, 0) + 1
        if len(chapter_dist) > 1:
            dist_str = " / ".join(
                f"{ch[:20]}: {cnt}개"
                for ch, cnt in sorted(chapter_dist.items(), key=lambda x: -x[1])[:4]
            )
            parts.append(f"장별 분포: {dist_str}")

        # 대표 각주 1개 (가장 긴 것)
        if certain_fn:
            best = max(certain_fn, key=lambda f: len(f.get("text", "")))
            loc  = best.get("location_string", "") or f"p.{best.get('page', '?')}"
            txt  = best.get("text", "")[:120]
            parts.append(f'예시 인용: "{txt}" ({loc})')

    # ── 연구사 공백 ──────────────────────────────────────────────────────────
    if gaps:
        parts.append(f"연구사 공백: {str(gaps[0])[:150]}")
        if len(gaps) > 1:
            parts.append(f"추가 공백: {str(gaps[1])[:120]}")

    # ── 경고 ─────────────────────────────────────────────────────────────────
    if warnings:
        parts.append(f"⚠️ {len(warnings)}개 항목 원본 확인 필요")

    if obsidian_result:
        parts.append(obsidian_result)

    return "\n\n".join(parts)


def _format_date_ko(date_str: str) -> str:
    """
    "1945-03-15" → "1945년 3월 15일"
    "1945-03"    → "1945년 3월"
    "1945"       → "1945년"
    """
    parts = date_str.split("-")
    if len(parts) == 3:
        return f"{parts[0]}년 {int(parts[1])}월 {int(parts[2])}일"
    elif len(parts) == 2:
        return f"{parts[0]}년 {int(parts[1])}월"
    return f"{parts[0]}년"


def _build_archive_info(
    record_group: str, entry: str, box: str, folder: str
) -> dict:
    """NARA 아카이브 식별자 딕셔너리를 구성합니다."""
    if not record_group and not entry:
        return {}

    # "59" → "RG 59", "RG 59" → "RG 59"
    rg = record_group.strip()
    if rg and not rg.upper().startswith("RG"):
        rg = f"RG {rg}"

    en = entry.strip()
    if en and not en.lower().startswith("entry"):
        en = f"Entry {en}"

    bx = box.strip()
    if bx and not bx.lower().startswith("box"):
        bx = f"Box {bx}"

    return {
        "record_group": rg,
        "entry": en,
        "box": bx,
        "folder": folder.strip(),
        "repository": "NARA (National Archives and Records Administration)",
    }


def _fallback_analysis(doc: ExtractedDocument, error_msg: str) -> dict:
    return {
        "title": doc.title,
        "authors": ["⚠️ 확인 필요"],
        "year": "⚠️ 확인 필요",
        "document_date": doc.document_date,
        "document_type": "other",
        "journal_or_source": "⚠️ 확인 필요",
        "main_thesis": f"⚠️ AI 분석 실패 — 원본 확인 필요. 오류: {error_msg[:200]}",
        "key_arguments": [], "methodology": "⚠️ 확인 필요",
        "primary_sources": [], "footnotes": [], "toc": [],
        "key_events": [], "key_persons": [],
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
        return analyze_document(
            file_path=fp,
            save_to_obsidian=parameters.get("save_to_obsidian", True),
            player=player,
            record_group=parameters.get("record_group", ""),
            entry=parameters.get("entry", ""),
            box=parameters.get("box", ""),
            folder=parameters.get("folder", ""),
        )

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
        fp   = parameters.get("file_path", "")
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
