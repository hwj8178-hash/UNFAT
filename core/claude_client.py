"""
claude_client.py — Anthropic Claude API 클라이언트 v2

역할: 역사학 논문·1차 사료 심층 분석, 출처 기반 정리, 연구사 공백 도출
  - 로컬 파일(PDF/HWP/DOCX/이미지)에서 추출된 텍스트 분석
  - 목차(TOC) 기반 장/절/페이지 정확 출처 추적
  - 미국 자료: 날짜·기관·핵심 내용 특화 분석
  - 페이지 번호 포함 원문 각주 생성
  - 여러 논문 종합 → 연구사 공백 추론
  - 대용량 문서: 청킹 후 병합 분석

v2 신기능:
  - 완전히 새로운 메가 프롬프트 (장/절 귀속, 자료 유형 자동 감지)
  - 구조 추출 + 내용 분석 통합 1패스
  - 미국 역사 자료 특화 필드 (document_date, document_type, issuing_body)
  - 각주마다 chapter/section/subsection 귀속
  - 확장 JSON 스키마

설정: config/api_keys.json 에 "claude_api_key" 키 추가 필요
"""

from __future__ import annotations
import json
import re
from pathlib import Path

# ─── 한도 상수 ────────────────────────────────────────────────────────────────
SINGLE_PASS_CHARS   = 160_000   # ~40K 토큰, 대부분 학술 논문 커버
CHUNK_SIZE_CHARS    = 130_000
CHUNK_OVERLAP_CHARS = 8_000


# ─── 시스템 프롬프트 ──────────────────────────────────────────────────────────

HISTORY_SYSTEM_PROMPT = """당신은 역사학 전문 연구 분석 AI입니다. 학술 텍스트와 1차 사료를 엄밀하게 분석합니다.

【핵심 원칙】
1. 인용문은 원문 그대로 — 단 한 글자도 수정하지 않습니다
2. [p.숫자] 마커로 표시된 페이지 번호만 사용합니다 (추측 금지)
3. 목차([목차] 섹션)가 있으면 반드시 활용하여 장/절 귀속을 수행합니다
4. ⚠️[불확실] 구간의 인용은 uncertain: true로 표시합니다
5. 불확실한 정보는 "⚠️ 확인 필요"로 명시합니다
6. 미국 정부문서·신문·외교문서는 날짜·기관·수신자·발신자를 최우선으로 추출합니다
7. 반드시 JSON 형식만으로 응답합니다 — 앞뒤 설명 없이 JSON만

【한자(漢字) 처리 원칙 — 절대 준수】
- 한자 독음(讀音)은 한국어 고유 읽기를 사용합니다: 獨立→독립, 朝鮮→조선, 大韓→대한
- 유사어·동의어·해석어로 절대 대체 금지: 獨立을 "자립", "자주", "독자"로 바꾸지 않습니다
- 텍스트에 漢字(한글) 병기 형식이 있으면 괄호 안 한글이 정확한 독음입니다
- 한자 용어는 그 시대 한국어 독음을 그대로 유지하세요 (현대 중국어 발음 사용 금지)
- [한자 독음 안내] 섹션이 제공되면 반드시 그 표를 따르세요"""

SYNTHESIS_SYSTEM_PROMPT = """당신은 역사학 연구사 분석 전문가입니다.
여러 논문과 자료의 분석 결과를 종합하여 연구사의 흐름, 논쟁 구도, 공백을 파악합니다.
구체적인 시기·지역·계층·주제를 중심으로 미개척 분야를 제안합니다.
한국어로 답변하세요."""

FOOTNOTE_SYSTEM_PROMPT = """당신은 역사학 논문 각주 작성 보조 AI입니다.
제공된 페이지 텍스트에서 요청한 내용과 관련된 문장을 찾아 정확히 인용합니다.
- 원문 문장을 그대로 인용 (수정 절대 금지)
- 여러 후보가 있으면 가장 핵심적인 문장 선택
- 텍스트가 불확실하면 반드시 경고 표시
- 장/절 정보가 제공되면 반드시 명시"""


# ─── API 클라이언트 ───────────────────────────────────────────────────────────

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


# ─── 확장 JSON 스키마 ─────────────────────────────────────────────────────────

_ANALYSIS_SCHEMA = """{
  "title": "논문/자료 제목",
  "subtitle": "부제목 (있는 경우)",
  "authors": ["저자1", "저자2"],
  "year": "출판연도 (YYYY 형식)",
  "document_date": "정확한 날짜 (YYYY-MM-DD, 미국 신문·외교문서·정부문서에서 특히 중요)",
  "document_type": "academic_paper|monograph|newspaper_article|government_document|diplomatic_cable|memorandum|report|letter|testimony|other",
  "language": "ko|en|ja|zh|mixed",
  "journal_or_source": "학술지명 / 신문명 / 기관명",
  "publisher": "출판사 또는 발행 기관",
  "publication_place": "출판지 또는 발행지",
  "volume_issue": "권호 (예: Vol.3 No.2 / 제3권 제2호)",
  "page_range": "논문 수록 페이지 (예: pp.1-45)",
  "issuing_body": "발행 기관 (미국 자료: State Department, War Department 등)",
  "recipients": ["수신자 (외교문서·공문의 경우)"],
  "senders": ["발신자 (외교문서·공문의 경우)"],
  "classification": "기밀 등급 (SECRET, CONFIDENTIAL, TOP SECRET 등, 해당 시)",
  "toc": [
    {"level": 레벨정수_1to4, "number": "제1장|Chapter 1|1.1 등", "title": "섹션 제목", "page": 시작페이지_정수}
  ],
  "document_structure": "문서 전체 구조 요약 1-2문장 (목차 없으면 Claude가 직접 추론)",
  "historical_period": "다루는 역사적 시기 (예: 1930-1945, 조선후기, 일제강점기)",
  "geographical_scope": "지리적 범위 (예: 한반도 남부, 한성부, Washington D.C.)",
  "main_thesis": "핵심 주장 및 테제 — 역사적 맥락 포함 2-3문장",
  "key_arguments": [
    "주요 논거 1 — 근거 페이지 명시 (예: [p.23] 근거)",
    "주요 논거 2"
  ],
  "key_events": ["주요 사건 (날짜·장소·관련 인물 포함)"],
  "key_persons": ["주요 인물 (직책·역할 포함)"],
  "methodology": "연구 방법론, 사료 비판 방법, 분석 틀",
  "primary_sources": ["활용 1차 사료 / 주요 출처 문헌"],
  "footnotes": [
    {
      "page": 페이지_정수,
      "chapter": "제N장 제목 (목차 기반 귀속, 없으면 빈문자열)",
      "section": "제N절 제목 (목차 기반 귀속, 없으면 빈문자열)",
      "subsection": "소절 제목 (있는 경우)",
      "location_string": "제N장 > 제N절 > p.XX 형식 전체 위치",
      "text": "원문에서 그대로 추출한 핵심 인용 문장 — 수정 절대 금지",
      "context": "이 인용이 논증에서 담당하는 역할",
      "source_cited": "이 문장이 인용하는 또 다른 출처 (있는 경우)",
      "footnote_number": null,
      "uncertain": false
    }
  ],
  "keywords": ["핵심키워드", "시기", "지역", "주제", "방법론"],
  "related_works": ["언급·인용·비판하는 선행 연구"],
  "research_gaps": [
    "이 연구가 드러내는 기존 연구의 구체적 공백 (시기/지역/계층 명시)",
    "파생될 수 있는 새로운 연구 문제"
  ],
  "liner_highlights": ["라이너 하이라이트용 핵심 문장 (최대 5개)"],
  "uncertainty_notes": ["⚠️ 원본 확인 필요 항목"],
  "hanja_glossary": [
    {"hanja": "漢字", "reading": "한자", "meaning": "한자 (필요 시 뜻 보충)"}
  ]
}"""


def _build_analysis_prompt(
    text_with_structure: str,
    file_path: str,
    knowledge_context: str = "",
    liner_context: str = "",
    first_pages_text: str = "",
    document_date: str = "",
    document_language: str = "",
    toc_text: str = "",
    hanja_guide: str = "",
) -> str:
    """Claude 분석용 완전한 프롬프트를 생성합니다."""

    # 맥락 섹션 구성
    sections: list[str] = []

    if knowledge_context:
        sections.append(f"[기존 연구 지식 그래프]\n{knowledge_context}")

    if liner_context:
        sections.append(f"[Liner AI 보조 자료]\n{liner_context}")

    # 문서 메타 힌트
    meta_hints = []
    if document_date:
        meta_hints.append(f"감지된 날짜: {document_date}")
    if document_language:
        lang_map = {"ko": "한국어", "en": "영어", "ja": "일본어", "zh": "중국어", "mixed": "복수언어"}
        meta_hints.append(f"주 언어: {lang_map.get(document_language, document_language)}")
    if first_pages_text:
        meta_hints.append(f"[서지정보 감지용 첫 페이지]\n{first_pages_text[:2000]}")
    if meta_hints:
        sections.append("\n".join(meta_hints))

    if hanja_guide:
        sections.append(hanja_guide)

    context_block = "\n\n".join(sections)

    # 문서 유형별 추가 지시사항
    if document_language == "en" or (document_date and len(document_date) >= 4):
        lang_instruction = """
【미국/영어 자료 특별 지시사항】
- document_date를 최우선으로 정확하게 추출하세요 (날짜가 핵심 메타데이터)
- issuing_body(발행기관), senders(발신자), recipients(수신자)를 반드시 추출하세요
- classification(기밀등급)이 있으면 추출하세요 (SECRET, CONFIDENTIAL 등)
- document_type을 newspaper_article / government_document / diplomatic_cable / memorandum 중 정확히 선택하세요
- key_events와 key_persons를 날짜·장소·직책 포함하여 상세히 작성하세요"""
    elif document_language == "ko":
        lang_instruction = """
【한국어 학술 자료 특별 지시사항】
- 목차([문서 목차] 섹션)를 적극 활용하여 각 footnote의 chapter/section/subsection을 귀속하세요
- 한국사 시기 구분을 정확히 파악하세요 (삼국시대, 고려, 조선전기/중기/후기, 일제강점기 등)
- 연구사 공백은 구체적 시기·지역·계층·주제로 명시하세요"""
    else:
        lang_instruction = ""

    prompt = f"""다음 역사학 문서를 정밀하게 분석하세요.
[p.숫자] 마커가 페이지 번호입니다. [목차] 섹션이 있으면 반드시 활용하세요.
⚠️[불확실] 구간은 OCR/추출 오류 가능성 있음.
{lang_instruction}

{context_block}

=== 문서 텍스트 (파일: {file_path}) ===
{text_with_structure[:SINGLE_PASS_CHARS]}

위 문서를 다음 JSON 스키마로만 응답하세요. 앞뒤 설명 없이 JSON만:
{_ANALYSIS_SCHEMA}"""

    return prompt


# ─── 메인 분석 함수 ───────────────────────────────────────────────────────────

def analyze_document_with_claude(
    text_with_pages: str,
    file_path: str,
    knowledge_context: str = "",
    liner_context: str = "",
    model: str = "claude-opus-4-8",
    # v2: 추가 파라미터
    first_pages_text: str = "",
    document_date: str = "",
    document_language: str = "",
    toc_text: str = "",
    # v4: 한자 처리
    hanja_guide: str = "",
) -> dict:
    """
    Claude API로 역사학 문서를 심층 분석합니다.

    text_with_pages: [p.N] 또는 구조화 텍스트 (get_text_with_structure() 권장)
    knowledge_context: 기존 지식 그래프 요약
    liner_context: Liner AI 보조 자료
    first_pages_text: 서지정보 감지용 첫 페이지들
    document_date: 사전 감지된 날짜
    document_language: 사전 감지된 언어
    toc_text: 사전 감지된 목차 텍스트
    """
    client = _get_client()

    user_prompt = _build_analysis_prompt(
        text_with_structure=text_with_pages,
        file_path=file_path,
        knowledge_context=knowledge_context,
        liner_context=liner_context,
        first_pages_text=first_pages_text,
        document_date=document_date,
        document_language=document_language,
        toc_text=toc_text,
        hanja_guide=hanja_guide,
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            system=HISTORY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        )
        raw = response.content[0].text.strip()

        # JSON 추출 (Markdown 코드블록 제거)
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```\s*$', '', raw, flags=re.MULTILINE)
        json_match = re.search(r'\{[\s\S]*\}', raw)
        result = json.loads(json_match.group() if json_match else raw)
        result["_analyzed_by"] = "claude"
        result["_model"] = model
        return result

    except json.JSONDecodeError as e:
        raise RuntimeError(f"Claude 응답 JSON 파싱 실패: {e}\n응답 앞 500자: {raw[:500]}")


# ─── 각주 추출 ────────────────────────────────────────────────────────────────

def extract_footnote_with_claude(
    page_text: str,
    page_number: int,
    quote_hint: str,
    file_path: str,
    is_uncertain: bool = False,
    chapter_info: dict | None = None,
    model: str = "claude-sonnet-4-6",
) -> str:
    """
    특정 페이지 텍스트에서 요청 내용과 가장 관련된 문장을 찾아 각주 형식으로 반환합니다.

    chapter_info: {"chapter": "제2장 ...", "section": "제1절 ..."}
    """
    client = _get_client()

    uncertain_note = "⚠️ 이 페이지의 텍스트는 OCR/추출이 불확실합니다. " if is_uncertain else ""

    location_str = f"p.{page_number}"
    if chapter_info:
        parts = []
        for key in ("chapter", "section", "subsection"):
            if chapter_info.get(key):
                parts.append(chapter_info[key])
        parts.append(f"p.{page_number}")
        location_str = " > ".join(parts)

    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=FOOTNOTE_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"{uncertain_note}"
                f"아래는 '{file_path}'의 {location_str} 텍스트입니다.\n"
                f"'{quote_hint}'와 관련된 핵심 문장을 원문 그대로 찾아 "
                f"각주 형식으로 제시해주세요.\n"
                f"형식: \"인용문\" ({location_str})\n\n"
                f"=== {location_str} 텍스트 ===\n{page_text[:3000]}"
            )
        }]
    )
    result = response.content[0].text.strip()
    if is_uncertain and "⚠️" not in result:
        result += "\n⚠️ [이 인용은 텍스트 추출이 불확실합니다 — 원본 파일 직접 확인 필요]"
    return result


# ─── 연구사 공백 종합 ─────────────────────────────────────────────────────────

def synthesize_research_gaps_with_claude(
    papers_json_list: list[dict],
    model: str = "claude-opus-4-8",
) -> str:
    """여러 논문 분석 결과를 종합하여 연구사 공백과 새로운 문제의식을 도출합니다."""
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


# ─── 대용량 문서 청킹 분석 ────────────────────────────────────────────────────

def analyze_large_document_with_claude(
    text_with_pages: str,
    file_path: str,
    knowledge_context: str = "",
    model: str = "claude-opus-4-8",
    # v2
    first_pages_text: str = "",
    document_date: str = "",
    document_language: str = "",
    toc_text: str = "",
    # v4: 한자
    hanja_guide: str = "",
) -> dict:
    """
    대용량 문서를 청크 단위로 분석한 뒤 종합합니다.
    SINGLE_PASS_CHARS 이하이면 단일 패스로 처리합니다.
    """
    if len(text_with_pages) <= SINGLE_PASS_CHARS:
        return analyze_document_with_claude(
            text_with_pages, file_path, knowledge_context, model=model,
            first_pages_text=first_pages_text,
            document_date=document_date, document_language=document_language,
            toc_text=toc_text,
            hanja_guide=hanja_guide,
        )

    chunks = _split_into_page_chunks(text_with_pages, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
    print(f"[Claude] 대용량 청킹: {len(text_with_pages):,}자 → {len(chunks)}청크")

    chunk_analyses = []
    for i, chunk in enumerate(chunks, 1):
        print(f"[Claude] 청크 {i}/{len(chunks)} 분석 중...")
        try:
            # 첫 청크에만 TOC·서지정보·한자 안내 전달
            analysis = analyze_document_with_claude(
                text_with_pages=chunk,
                file_path=f"{file_path} [청크 {i}/{len(chunks)}]",
                knowledge_context=knowledge_context,
                model=model,
                first_pages_text=first_pages_text if i == 1 else "",
                document_date=document_date,
                document_language=document_language,
                toc_text=toc_text if i == 1 else "",
                hanja_guide=hanja_guide if i == 1 else "",
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

    sections = re.split(r'(?=\[p\.\d+\])', text)
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for section in sections:
        sec_size = len(section)
        if current_size + sec_size > chunk_size and current:
            chunks.append("".join(current))
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
        "authors", "key_arguments", "key_events", "key_persons",
        "primary_sources", "keywords", "related_works",
        "research_gaps", "liner_highlights", "uncertainty_notes",
        "recipients", "senders",
    ]
    for field in list_fields:
        seen: set[str] = set()
        combined: list = []
        for a in analyses:
            for item in a.get(field, []):
                item_str = str(item)
                if item_str not in seen:
                    seen.add(item_str)
                    combined.append(item)
        merged[field] = combined

    # TOC: 첫 청크 TOC 우선 (이후 청크는 중복 가능)
    if not merged.get("toc") and len(analyses) > 1:
        for a in analyses[1:]:
            if a.get("toc"):
                merged["toc"] = a["toc"]
                break

    # 각주: 페이지 번호 + 텍스트 앞 60자 기준 중복 제거 후 페이지순 정렬
    seen_fn: set[tuple] = set()
    all_footnotes: list[dict] = []
    for a in analyses:
        for fn in a.get("footnotes", []):
            key = (fn.get("page"), str(fn.get("text", ""))[:60])
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

    merged["_chunked"]      = True
    merged["_chunk_count"]  = len(analyses)
    return merged


# ─── 가용성 확인 ─────────────────────────────────────────────────────────────

def is_claude_available() -> bool:
    """Claude API 키와 패키지가 정상적으로 설정되어 있는지 확인합니다."""
    try:
        _get_api_key()
        import anthropic  # noqa
        return True
    except Exception:
        return False
