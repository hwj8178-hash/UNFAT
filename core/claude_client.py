"""
claude_client.py — Anthropic Claude API 클라이언트

역할: 역사학 논문 심층 분석, 출처 기반 정리, 연구사 공백 도출
  - 로컬 파일(PDF/HWP/DOCX/이미지)에서 추출된 텍스트 분석
  - 페이지 번호 포함 원문 각주 생성
  - 여러 논문 종합 → 연구사 공백 추론
  - Liner에서 가져온 웹 자료와 병합 분석

설정: config/api_keys.json 에 "claude_api_key" 키 추가 필요
"""

from __future__ import annotations
import json
import re
from pathlib import Path

# 단일 패스 한도: ~40K 토큰, 대부분 학술 논문 커버
SINGLE_PASS_CHARS = 160_000
# 청킹 임계값 초과 시 청크 크기 및 중첩
CHUNK_SIZE_CHARS  = 130_000
CHUNK_OVERLAP_CHARS = 8_000


HISTORY_SYSTEM_PROMPT = """당신은 역사학 전문 연구 분석 AI입니다. 주어진 학술 텍스트를 엄밀하게 분석합니다.

분석 원칙:
1. 인용문은 원문 그대로 추출 — 단 한 글자도 수정하지 않습니다
2. 페이지 번호 [p.숫자]가 명시된 경우만 해당 번호를 사용합니다
3. 불확실하거나 확인되지 않은 정보는 반드시 "⚠️ 확인 필요"로 표시합니다
4. ⚠️[불확실] 표시 구간의 인용은 uncertain: true로 설정합니다
5. 역사학의 핵심은 출처이므로, 근거 없는 추론은 하지 않습니다
6. JSON 형식으로만 응답합니다"""

SYNTHESIS_SYSTEM_PROMPT = """당신은 역사학 연구사 분석 전문가입니다.
여러 논문과 자료의 분석 결과를 종합하여 연구사의 흐름, 논쟁 구도, 공백을 파악합니다.
구체적인 시기·지역·계층·주제를 중심으로 미개척 분야를 제안합니다.
한국어로 답변하세요."""

FOOTNOTE_SYSTEM_PROMPT = """당신은 역사학 논문 각주 작성 보조 AI입니다.
제공된 페이지 텍스트에서 요청한 내용과 관련된 문장을 찾아 정확히 인용합니다.
- 원문 문장을 그대로 인용 (수정 절대 금지)
- 여러 후보가 있으면 가장 핵심적인 문장 선택
- 텍스트가 불확실하면 반드시 경고 표시"""


def _get_api_key() -> str:
    config_path = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        key = cfg.get("claude_api_key", "")
        if not key:
            raise ValueError("claude_api_key가 설정되지 않았습니다.")
        return key
    except FileNotFoundError:
        raise RuntimeError(f"API 설정 파일을 찾을 수 없습니다: {config_path}")


def _get_client():
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "anthropic 패키지가 설치되지 않았습니다. pip install anthropic 실행하세요."
        )
    return anthropic.Anthropic(api_key=_get_api_key())


# ─── 메인 분석 함수 ──────────────────────────────────────────────────────────

def analyze_document_with_claude(
    text_with_pages: str,
    file_path: str,
    knowledge_context: str = "",
    liner_context: str = "",
    model: str = "claude-opus-4-8",
) -> dict:
    """
    Claude API로 역사학 문서를 심층 분석합니다.

    text_with_pages: [p.N] 형식으로 페이지 번호가 포함된 텍스트
    knowledge_context: 기존 지식 그래프 요약 (맥락 제공)
    liner_context: Liner AI가 수집한 보조 자료 (있는 경우)
    """
    client = _get_client()

    liner_section = f"\n[Liner AI 보조 자료]\n{liner_context}\n" if liner_context else ""

    user_prompt = f"""다음 역사학 문서를 분석해주세요.
각 페이지의 텍스트는 [p.숫자] 형식으로 표시되어 있습니다.
⚠️[불확실] 표시 구간은 OCR/추출 오류 가능성이 있습니다.

{knowledge_context}{liner_section}
=== 문서 텍스트 (파일: {file_path}) ===
{text_with_pages[:SINGLE_PASS_CHARS]}

다음 JSON 형식으로만 응답하세요:
{{
  "title": "논문/자료 제목",
  "authors": ["저자1", "저자2"],
  "year": "출판연도",
  "journal": "학술지명 또는 출처",
  "main_thesis": "핵심 주장 및 테제 — 역사적 맥락 포함 2-3문장",
  "key_arguments": [
    "주요 논거 1 (근거 포함)",
    "주요 논거 2 (근거 포함)"
  ],
  "methodology": "연구 방법론, 사료 비판 방법, 분석 틀",
  "primary_sources": ["활용 주요 사료 또는 1차 문헌"],
  "footnotes": [
    {{
      "page": 페이지번호_정수,
      "text": "원문에서 그대로 추출한 핵심 인용 문장 — 수정 절대 금지",
      "context": "이 인용문이 논문 내에서 담당하는 논증 역할",
      "uncertain": false
    }}
  ],
  "keywords": ["핵심키워드1", "핵심키워드2", "시기", "지역", "주제"],
  "related_works": ["이 논문이 언급하거나 비판하는 선행 연구"],
  "research_gaps": [
    "이 연구가 드러내는 기존 연구의 구체적 공백 (시기/지역/계층 명시)",
    "이 논문에서 파생될 수 있는 새로운 연구 문제"
  ],
  "liner_highlights": [
    "라이너에 하이라이트로 저장할 핵심 문장 (최대 5개)"
  ],
  "uncertainty_notes": [
    "⚠️[불확실] 구간 경고 또는 추출 오류 의심 항목"
  ]
}}"""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            system=HISTORY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        )
        raw = response.content[0].text.strip()
        json_match = re.search(r'\{[\s\S]*\}', raw)
        result = json.loads(json_match.group() if json_match else raw)
        result["_analyzed_by"] = "claude"
        result["_model"] = model
        return result
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Claude 응답 JSON 파싱 실패: {e}")


def synthesize_research_gaps_with_claude(
    papers_json_list: list[dict],
    model: str = "claude-opus-4-8",
) -> str:
    """
    여러 논문 분석 결과를 종합하여 연구사 공백과 새로운 문제의식을 도출합니다.
    """
    client = _get_client()

    papers_text = json.dumps(papers_json_list, ensure_ascii=False, indent=2)[:40000]

    response = client.messages.create(
        model=model,
        max_tokens=3000,
        system=SYNTHESIS_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                "다음 논문들의 분석 데이터를 종합하여 연구사 공백과 "
                "새로운 문제의식을 구체적으로 도출해주세요. "
                "시기·지역·계층·방법론의 공백을 명확하게 구분하여 제시하세요.\n\n"
                f"{papers_text}"
            )
        }]
    )
    return response.content[0].text.strip()


def extract_footnote_with_claude(
    page_text: str,
    page_number: int,
    quote_hint: str,
    file_path: str,
    is_uncertain: bool = False,
    model: str = "claude-sonnet-4-6",
) -> str:
    """
    특정 페이지 텍스트에서 요청 내용과 가장 관련된 문장을 찾아 각주 형식으로 반환합니다.
    Sonnet을 사용해 빠르게 처리합니다.
    """
    client = _get_client()

    uncertain_note = "⚠️ 이 페이지의 텍스트는 OCR/추출이 불확실합니다. " if is_uncertain else ""

    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=FOOTNOTE_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"{uncertain_note}아래는 '{file_path}'의 p.{page_number} 텍스트입니다.\n"
                f"'{quote_hint}'와 관련된 핵심 문장을 원문 그대로 찾아 각주 형식으로 제시해주세요.\n\n"
                f"=== p.{page_number} 텍스트 ===\n{page_text[:3000]}"
            )
        }]
    )
    result = response.content[0].text.strip()
    if is_uncertain and "⚠️" not in result:
        result += "\n⚠️ [이 인용은 텍스트 추출이 불확실합니다 — 원본 파일 직접 확인 필요]"
    return result


def analyze_large_document_with_claude(
    text_with_pages: str,
    file_path: str,
    knowledge_context: str = "",
    model: str = "claude-opus-4-8",
) -> dict:
    """
    대용량 문서를 청크 단위로 분석한 뒤 종합합니다.
    SINGLE_PASS_CHARS 이하이면 단일 패스로 처리하고,
    초과하면 CHUNK_SIZE_CHARS 크기로 나눠 각 청크를 분석한 뒤 병합합니다.
    """
    if len(text_with_pages) <= SINGLE_PASS_CHARS:
        return analyze_document_with_claude(
            text_with_pages, file_path, knowledge_context, model=model
        )

    chunks = _split_into_page_chunks(text_with_pages, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
    print(f"[Claude] 대용량 문서 청킹: {len(text_with_pages):,}자 → {len(chunks)}청크")

    chunk_analyses = []
    for i, chunk in enumerate(chunks, 1):
        print(f"[Claude] 청크 {i}/{len(chunks)} 분석 중...")
        try:
            analysis = analyze_document_with_claude(
                text_with_pages=chunk,
                file_path=f"{file_path} [청크 {i}/{len(chunks)}]",
                knowledge_context=knowledge_context,
                model=model,
            )
            chunk_analyses.append(analysis)
        except Exception as e:
            print(f"[Claude] 청크 {i} 분석 실패: {e}")

    if not chunk_analyses:
        raise RuntimeError("모든 청크 분석에 실패했습니다.")

    return _merge_chunk_analyses(chunk_analyses, file_path, model)


def _split_into_page_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """[p.N] 페이지 경계를 기준으로 대용량 텍스트를 청크로 분할합니다."""
    if len(text) <= chunk_size:
        return [text]

    # [p.숫자] 마커 앞에서 분할
    sections = re.split(r'(?=\[p\.\d+\])', text)
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for section in sections:
        sec_size = len(section)
        if current_size + sec_size > chunk_size and current:
            chunks.append("".join(current))
            # 이전 청크 끝부분을 중첩으로 가져와 문맥 유지
            tail: list[str] = []
            tail_size = 0
            for s in reversed(current):
                if tail_size + len(s) > overlap:
                    break
                tail.insert(0, s)
                tail_size += len(s)
            current = tail + [section]
            current_size = tail_size + sec_size
        else:
            current.append(section)
            current_size += sec_size

    if current:
        chunks.append("".join(current))

    return chunks


def _merge_chunk_analyses(analyses: list[dict], file_path: str, model: str) -> dict:
    """여러 청크 분석 결과를 하나의 통합 분석으로 병합합니다."""
    if len(analyses) == 1:
        return analyses[0]

    merged = dict(analyses[0])

    # 리스트 필드: 중복 제거 후 합산
    list_fields = [
        "key_arguments", "primary_sources", "keywords",
        "related_works", "research_gaps", "liner_highlights", "uncertainty_notes",
    ]
    for field in list_fields:
        seen: set[str] = set()
        combined: list[str] = []
        for a in analyses:
            for item in a.get(field, []):
                item_str = str(item)
                if item_str not in seen:
                    seen.add(item_str)
                    combined.append(item)
        merged[field] = combined

    # 각주: 페이지 번호 기준 중복 제거 후 정렬
    seen_fn: set[tuple] = set()
    all_footnotes: list[dict] = []
    for a in analyses:
        for fn in a.get("footnotes", []):
            key = (fn.get("page"), fn.get("text", "")[:60])
            if key not in seen_fn:
                seen_fn.add(key)
                all_footnotes.append(fn)
    merged["footnotes"] = sorted(all_footnotes, key=lambda x: x.get("page", 0))

    # 핵심 주장: Claude Sonnet으로 청크 주장 통합 요약
    try:
        client = _get_client()
        all_theses = "\n".join(
            f"[청크{i+1}] {a.get('main_thesis', '')}"
            for i, a in enumerate(analyses)
            if a.get("main_thesis")
        )
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": (
                    "다음 청크별 핵심 주장을 하나의 통합된 핵심 주장으로 요약하세요 (2-3문장):\n\n"
                    + all_theses
                ),
            }]
        )
        merged["main_thesis"] = response.content[0].text.strip()
    except Exception as e:
        print(f"[Claude] 주장 통합 실패: {e}")
        merged["main_thesis"] = " | ".join(
            a.get("main_thesis", "") for a in analyses if a.get("main_thesis")
        )[:500]

    merged["_chunked"] = True
    merged["_chunk_count"] = len(analyses)
    return merged


def is_claude_available() -> bool:
    """Claude API 키와 패키지가 정상적으로 설정되어 있는지 확인합니다."""
    try:
        _get_api_key()
        import anthropic  # noqa
        return True
    except Exception:
        return False
