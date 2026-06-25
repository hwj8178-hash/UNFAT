"""
document_extractor.py — 역사학 연구용 문서 텍스트 추출 엔진 v2

지원 형식:
  PDF    → 페이지별 텍스트 추출 (pdfplumber)
  HWP    → 한글 파일 텍스트 추출 (hwp5 / LibreOffice 변환)
  DOCX   → 워드 파일 텍스트 추출 (python-docx)
  JPG/PNG → OCR 텍스트 추출 (easyocr, 한국어 지원)

v2 신기능:
  - 목차(TOC) 자동 감지 및 파싱 → 장/절 구조 인식
  - 페이지 → 장/절 역추적 (get_chapter_for_page)
  - 미국 자료 날짜 자동 감지 (document_date)
  - 언어 자동 감지 (ko/en/ja/zh)
  - Claude 프롬프트용 구조화 텍스트 출력 (get_text_with_structure)
"""

from __future__ import annotations
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── 데이터 클래스 ────────────────────────────────────────────────────────────

@dataclass
class PageContent:
    """단일 페이지/섹션의 텍스트와 메타데이터"""
    page_number: int
    text: str
    is_uncertain: bool = False
    uncertainty_reason: str = ""


@dataclass
class TocEntry:
    """목차 항목 — 장/절/소절 계층 구조"""
    level: int           # 1=편/부, 2=장, 3=절, 4=소절
    number: str          # "제1장", "1", "1.1", "Chapter 1" 등
    title: str           # 섹션 제목
    page_number: int     # 시작 페이지 번호 (-1 = 불명확)

    def full_label(self) -> str:
        """표시용 전체 라벨"""
        return f"{self.number} {self.title}".strip()


@dataclass
class ExtractedDocument:
    """추출된 전체 문서"""
    file_path: str
    file_type: str
    title: str
    pages: list[PageContent] = field(default_factory=list)
    total_pages: int = 0
    extraction_warnings: list[str] = field(default_factory=list)

    # v2: 구조 정보
    toc: list[TocEntry] = field(default_factory=list)
    document_date: str = ""          # "1945-03-15" 형식 (미국 자료 등)
    document_language: str = ""      # "ko", "en", "ja", "zh", "mixed"

    # v3: 국사편찬위원회 사료참조번호 (파일명 AUS* 자동 감지)
    nikh_reference: str = ""         # 전체 참조번호 (예: "AUS2012_001_0001_0001")
    nikh_ref_parsed: dict = field(default_factory=dict)  # 파싱된 구성요소

    # ── 기본 텍스트 프로퍼티 ──────────────────────────────────────────────────

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def first_pages_text(self) -> str:
        """처음 12페이지 텍스트 (서지정보·날짜·목차 감지용)"""
        return "\n\n".join(
            f"[p.{p.page_number}]\n{p.text}"
            for p in self.pages[:12] if p.text.strip()
        )

    @property
    def toc_text(self) -> str:
        """목차 텍스트 표현 (Claude 프롬프트 삽입용)"""
        if not self.toc:
            return ""
        lines = ["[문서 목차 / Table of Contents]"]
        for entry in self.toc:
            indent = "  " * (entry.level - 1)
            page_str = f" ··· p.{entry.page_number}" if entry.page_number > 0 else ""
            lines.append(f"{indent}{entry.full_label()}{page_str}")
        return "\n".join(lines)

    # ── 구조 검색 메서드 ─────────────────────────────────────────────────────

    def get_chapter_for_page(self, page_num: int) -> dict:
        """
        페이지 번호에 해당하는 장/절/소절 정보를 반환합니다.
        TOC가 없으면 빈 dict 반환.

        Returns:
            {"part": "제1편 ...", "chapter": "제2장 ...", "section": "제3절 ..."}
        """
        if not self.toc:
            return {}

        result: dict[str, str] = {}
        sorted_toc = sorted(self.toc, key=lambda e: (e.page_number, e.level))

        for entry in sorted_toc:
            if entry.page_number < 0:
                continue
            if entry.page_number > page_num:
                break
            label = entry.full_label()
            if entry.level == 1:
                result["part"] = label
                result.pop("chapter", None)
                result.pop("section", None)
                result.pop("subsection", None)
            elif entry.level == 2:
                result["chapter"] = label
                result.pop("section", None)
                result.pop("subsection", None)
            elif entry.level == 3:
                result["section"] = label
                result.pop("subsection", None)
            elif entry.level == 4:
                result["subsection"] = label

        return result

    def get_location_string(self, page_num: int) -> str:
        """
        "제2장 상업 발달 > 제1절 시장 구조 > p.45" 형식의 위치 문자열 반환
        """
        loc = self.get_chapter_for_page(page_num)
        parts = []
        for key in ("part", "chapter", "section", "subsection"):
            if key in loc:
                parts.append(loc[key])
        parts.append(f"p.{page_num}")
        return " > ".join(parts)

    # ── 텍스트 출력 메서드 ────────────────────────────────────────────────────

    def get_text_with_pages(self) -> str:
        """각주 작성에 사용할 페이지 번호 포함 텍스트"""
        parts = []
        for p in self.pages:
            if p.text.strip():
                marker = f"[p.{p.page_number}]"
                if p.is_uncertain:
                    marker += f" ⚠️[불확실: {p.uncertainty_reason}]"
                parts.append(f"{marker}\n{p.text}")
        return "\n\n".join(parts)

    def get_text_with_structure(self) -> str:
        """
        목차·장절 정보가 포함된 구조화 텍스트 (Claude 심층 분석용).
        각 페이지 앞에 [장/절] 헤더를 삽입합니다.
        """
        result_parts = []

        # 1. 목차 삽입
        if self.toc_text:
            result_parts.append(self.toc_text)
            result_parts.append("")

        # 2. 날짜 및 언어 정보
        meta = []
        if self.document_date:
            meta.append(f"[문서 날짜: {self.document_date}]")
        if self.document_language:
            lang_names = {"ko": "한국어", "en": "영어", "ja": "일본어", "zh": "중국어", "mixed": "복수언어"}
            meta.append(f"[언어: {lang_names.get(self.document_language, self.document_language)}]")
        if meta:
            result_parts.extend(meta)
            result_parts.append("")

        # 3. 페이지별 텍스트 + 장절 헤더
        prev_location = ""
        for p in self.pages:
            if not p.text.strip():
                continue

            # 이전 페이지와 다른 장/절에 진입하면 헤더 삽입
            if self.toc:
                loc = self.get_chapter_for_page(p.page_number)
                loc_str = " > ".join(
                    loc[k] for k in ("part", "chapter", "section", "subsection") if k in loc
                )
                if loc_str and loc_str != prev_location:
                    result_parts.append(f"\n{'─'*60}")
                    result_parts.append(f"[{loc_str}]")
                    prev_location = loc_str

            marker = f"[p.{p.page_number}]"
            if p.is_uncertain:
                marker += f" ⚠️[불확실: {p.uncertainty_reason}]"
            result_parts.append(f"{marker}\n{p.text}")

        return "\n\n".join(result_parts)

    def get_page_chunks(
        self, chunk_chars: int = 130_000, overlap_chars: int = 8_000
    ) -> list[str]:
        """
        대용량 문서를 위한 청크 단위 텍스트 분할.
        각 청크는 [p.N] 경계를 존중하며, overlap_chars만큼 중첩합니다.
        """
        full = self.get_text_with_structure() if self.toc else self.get_text_with_pages()
        if len(full) <= chunk_chars:
            return [full]

        sections = re.split(r'(?=\[p\.\d+\])', full)
        chunks: list[str] = []
        current: list[str] = []
        current_size = 0

        for sec in sections:
            sec_size = len(sec)
            if current_size + sec_size > chunk_chars and current:
                chunks.append("".join(current))
                tail: list[str] = []
                tail_size = 0
                for s in reversed(current):
                    if tail_size + len(s) > overlap_chars:
                        break
                    tail.insert(0, s)
                    tail_size += len(s)
                current = tail + [sec]
                current_size = tail_size + sec_size
            else:
                current.append(sec)
                current_size += sec_size

        if current:
            chunks.append("".join(current))

        return chunks


# ─── 공개 진입점 ─────────────────────────────────────────────────────────────

def extract_document(file_path: str) -> ExtractedDocument:
    """
    파일 경로를 받아 ExtractedDocument를 반환합니다.
    파일 형식을 자동으로 감지하고, 추출 후 목차·날짜·언어를 자동 감지합니다.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        doc = _extract_pdf(path)
    elif ext in (".hwp", ".hwpx"):
        doc = _extract_hwp(path)
    elif ext in (".docx", ".doc"):
        doc = _extract_docx(path)
    elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        doc = _extract_image(path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")

    # 후처리: 구조 분석
    _post_process(doc)
    return doc


# ─── 후처리: 목차·날짜·언어 감지 ─────────────────────────────────────────────

def _post_process(doc: ExtractedDocument) -> None:
    """추출 완료 후 목차, 날짜, 언어, NIKH 참조번호를 감지합니다."""
    if not doc.pages:
        return

    # 언어 감지 (처음 3000자 기준)
    sample = " ".join(p.text for p in doc.pages[:5])[:3000]
    doc.document_language = _detect_language(sample)

    # 날짜 감지 (처음 5페이지)
    first_text = " ".join(p.text for p in doc.pages[:5])
    doc.document_date = _detect_document_date(first_text)

    # 목차 감지 (처음 20페이지 내)
    toc = _detect_and_parse_toc(doc.pages[:20])
    if toc:
        doc.toc = toc
        print(f"[DocExtract] 목차 감지: {len(toc)}개 항목")
    else:
        print("[DocExtract] 목차 미감지 — Claude가 구조 추론")

    # 국사편찬위원회 AUS 사료참조번호 감지 (파일명 기준)
    nikh_ref, nikh_parsed = _parse_nikh_reference(doc.file_path)
    if nikh_ref:
        doc.nikh_reference = nikh_ref
        doc.nikh_ref_parsed = nikh_parsed
        print(f"[DocExtract] 국편 사료참조번호 감지: {nikh_ref}")


def _parse_nikh_reference(file_path: str) -> tuple[str, dict]:
    """
    파일명에서 국사편찬위원회 사료참조번호(AUS*)를 파싱합니다.

    국편 미국자료 참조번호 형식 예시:
      AUS2012_001_0001_0001   → 연도:2012, 컬렉션:001, 문서:0001, 페이지:0001
      AUS20120010001000100001 → 같은 구조, 구분자 없는 버전
      AUS_001_0001_0001       → 연도 없는 버전
      AUS-59-1234-001         → 하이픈 구분자 버전

    Returns:
      (참조번호_문자열, 파싱_딕셔너리)
      파일명이 AUS로 시작하지 않으면 ("", {}) 반환
    """
    stem = Path(file_path).stem  # 확장자 제외 파일명

    # AUS로 시작하는지 확인 (대소문자 무관)
    if not re.match(r'^AUS', stem, re.IGNORECASE):
        return "", {}

    ref = stem  # 전체 참조번호 = 파일명 (확장자 제외)
    parsed: dict = {"raw": ref, "prefix": "AUS"}

    # 구분자 정규화 (하이픈·언더스코어 → 언더스코어)
    normalized = re.sub(r'[-_]+', '_', stem)
    parts = normalized.split('_')

    # 첫 번째 파트에서 "AUS" 제거
    after_aus = parts[0][3:] if parts[0].upper().startswith('AUS') else ''
    remaining = ([after_aus] if after_aus else []) + parts[1:]
    # 빈 파트 제거
    remaining = [p for p in remaining if p]

    # ── 형식 1: AUS + 연도(4자리) + 나머지 ──────────────────────────────────
    if remaining and re.match(r'^\d{4}$', remaining[0]):
        parsed["digitization_year"] = remaining[0]
        remaining = remaining[1:]

    # ── 나머지 숫자 파트 순서대로 배정 ──────────────────────────────────────
    # 국편 사료번호 구조: 컬렉션번호 → 문서번호 → 페이지/아이템번호
    field_names = ["collection", "document_no", "item_no", "sub_no"]
    for i, val in enumerate(remaining[:4]):
        if re.match(r'^\d+$', val):
            parsed[field_names[i]] = val
        else:
            # 숫자가 아닌 파트는 기타 정보로 저장
            parsed.setdefault("extra", []).append(val)

    return ref, parsed


def _format_nikh_reference(ref: str, parsed: dict) -> str:
    """국편 사료참조번호를 사람이 읽기 좋은 형식으로 변환합니다."""
    if not ref:
        return ""
    parts = [f"참조번호: {ref}"]
    if parsed.get("digitization_year"):
        parts.append(f"수집연도: {parsed['digitization_year']}년")
    if parsed.get("collection"):
        parts.append(f"컬렉션: {int(parsed['collection']):03d}")
    if parsed.get("document_no"):
        parts.append(f"문서번호: {int(parsed['document_no']):04d}")
    if parsed.get("item_no"):
        parts.append(f"아이템: {int(parsed['item_no']):04d}")
    return " | ".join(parts)


def _detect_language(text: str) -> str:
    """텍스트에서 주 언어를 감지합니다."""
    if not text:
        return ""

    korean = len(re.findall(r'[가-힣]', text))
    japanese = len(re.findall(r'[぀-ゟ゠-ヿ]', text))
    chinese = len(re.findall(r'[一-鿿]', text))
    latin = len(re.findall(r'[a-zA-Z]', text))

    total = max(korean + japanese + chinese + latin, 1)

    if korean / total > 0.3:
        return "ko"
    elif japanese / total > 0.3:
        return "ja"
    elif chinese / total > 0.3:
        return "zh"
    elif latin / total > 0.3:
        return "en"
    return "mixed"


def _detect_document_date(text: str) -> str:
    """
    텍스트에서 날짜를 감지합니다.
    미국 신문·정부문서 형식 우선 지원.

    Returns: "YYYY-MM-DD", "YYYY-MM", "YYYY" 또는 ""
    """
    MONTH_NAMES = {
        "january": "01", "february": "02", "march": "03",
        "april": "04", "may": "05", "june": "06",
        "july": "07", "august": "08", "september": "09",
        "october": "10", "november": "11", "december": "12",
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "jun": "06", "jul": "07", "aug": "08", "sep": "09",
        "oct": "10", "nov": "11", "dec": "12",
    }

    patterns = [
        # "March 15, 1945" / "March 15 1945"
        (r'\b(January|February|March|April|May|June|July|August|September|October|November|December'
         r'|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2}),?\s+(\d{4})\b',
         lambda m: f"{m.group(3)}-{MONTH_NAMES[m.group(1).lower()[:3]]}-{int(m.group(2)):02d}"),

        # "15 March 1945" / "15 March, 1945"
        (r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December'
         r'|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?,?\s+(\d{4})\b',
         lambda m: f"{m.group(3)}-{MONTH_NAMES[m.group(2).lower()[:3]]}-{int(m.group(1)):02d}"),

        # "1945년 3월 15일"
        (r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',
         lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"),

        # "1945년 3월"
        (r'(\d{4})년\s*(\d{1,2})월(?!\s*\d)',
         lambda m: f"{m.group(1)}-{int(m.group(2)):02d}"),

        # "YYYY-MM-DD", "YYYY/MM/DD", or "YYYY.MM.DD" (Korean dot format)
        (r'\b(\d{4})[-/.](\d{2})[-/.](\d{2})\b',
         lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),

        # "MM/DD/YYYY" or "MM-DD-YYYY" (US format)
        (r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b',
         lambda m: f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"),

        # "1953년" — Korean year-only suffix
        (r'(\d{4})년(?!\s*\d+월)',
         lambda m: m.group(1)),

        # Standalone year near document header keywords
        (r'(?:dated?|date:|published?:?|issued?:?|작성\s*연도\s*:|일자\s*:)\s*[,:]?\s*(\d{4})',
         lambda m: m.group(1)),

        # "Volume ... 1945" — year only
        (r'\b(1[0-9]{3}|20[0-2][0-9])\b',
         lambda m: m.group(1)),
    ]

    for pattern, formatter in patterns:
        match = re.search(pattern, text[:3000], re.IGNORECASE)
        if match:
            try:
                result = formatter(match)
                # Sanity check: year should be 1000-2100
                year = int(result[:4])
                if 1000 <= year <= 2100:
                    return result
            except Exception:
                continue
    return ""


# ─── 목차 감지 및 파싱 ───────────────────────────────────────────────────────

# 목차 헤더 감지 패턴
_TOC_HEADER_RE = re.compile(
    r'^\s*(목\s*차|차\s*례|목\s*록|목\s*차\s*표|'
    r'Contents?|Table\s+of\s+Contents|CONTENTS|INDEX)\s*$',
    re.IGNORECASE | re.MULTILINE
)

# 한국어 목차 항목 패턴: "제2장 경제 변동 ····· 45" 또는 "2.1 경제 변동 ····· 45"
_TOC_KO_RE = re.compile(
    r'^[ \t]*(제?\s*\d+\s*[편부장절항목](?:\s*\d+[절항목])?'
    r'|\d{1,2}(?:\.\d{1,2}){0,3}\.?'
    r'|[가나다라마바사아자차카타파하]\.'
    r'|[IVXivx]{1,6}\.?)'
    r'[ \t]+([^\d\n]{1,60}?)'
    r'[ \t]*[·.…·\-]{0,30}[ \t]*'
    r'(\d{1,4})[ \t]*$',
    re.MULTILINE
)

# 영어 목차 항목 패턴: "Chapter 2. Economic Change ......... 45"
_TOC_EN_RE = re.compile(
    r'^[ \t]*(Chapter\s+\d+|Part\s+\d+|Section\s+\d+(?:\.\d+)?'
    r'|\d{1,2}(?:\.\d{1,2}){0,3}\.?'
    r'|[IVXivx]{1,6}\.?)'
    r'[ \t]*[.:]?[ \t]+'
    r'([^\d\n]{1,80}?)'
    r'[ \t]*[.…\-]{0,30}[ \t]*'
    r'(\d{1,4})[ \t]*$',
    re.IGNORECASE | re.MULTILINE
)


def _detect_and_parse_toc(pages: list[PageContent]) -> list[TocEntry]:
    """
    페이지 목록에서 목차를 감지하고 파싱합니다.

    전략:
    1. 목차 헤더('목차', 'Contents' 등)가 있는 페이지 찾기
    2. 해당 페이지 + 다음 2페이지에서 목차 항목 추출
    3. 실패 시 전체 처음 10페이지에서 패턴 매칭으로 시도
    """
    # 전략 1: 목차 헤더 페이지 기반
    toc_page_indices = [
        i for i, p in enumerate(pages)
        if _TOC_HEADER_RE.search(p.text)
    ]

    if toc_page_indices:
        toc_text_parts = []
        for idx in toc_page_indices:
            for j in range(idx, min(idx + 4, len(pages))):
                toc_text_parts.append(pages[j].text)
        combined = "\n".join(toc_text_parts)
        entries = _extract_toc_entries(combined)
        if len(entries) >= 3:
            return entries

    # 전략 2: 처음 10페이지 전체에서 패턴 매칭
    early_text = "\n".join(p.text for p in pages[:10])
    entries = _extract_toc_entries(early_text)

    # 품질 기준: 최소 3개 이상 페이지 번호가 있는 항목
    valid = [e for e in entries if e.page_number > 0]
    if len(valid) >= 3:
        return valid

    return []


def _extract_toc_entries(text: str) -> list[TocEntry]:
    """텍스트에서 목차 항목을 추출합니다."""
    entries: list[TocEntry] = []
    seen: set[str] = set()

    def _add(number: str, title: str, page: int):
        key = f"{number}|{title[:20]}"
        if key in seen:
            return
        seen.add(key)
        level = _guess_toc_level(number)
        entries.append(TocEntry(
            level=level,
            number=number.strip(),
            title=title.strip(),
            page_number=page,
        ))

    # 한국어 패턴 우선
    for m in _TOC_KO_RE.finditer(text):
        number = m.group(1).strip()
        title  = m.group(2).strip()
        page   = int(m.group(3))
        if title and 1 <= page <= 9999:
            _add(number, title, page)

    # 영어 패턴 (한국어 미감지 또는 보완)
    if len(entries) < 3:
        for m in _TOC_EN_RE.finditer(text):
            number = m.group(1).strip()
            title  = m.group(2).strip()
            page   = int(m.group(3))
            if title and 1 <= page <= 9999:
                _add(number, title, page)

    return entries


def _guess_toc_level(number: str) -> int:
    """번호 형식으로 목차 레벨 추정"""
    num = number.strip()

    # 명시적 레벨 키워드
    if re.search(r'편|部|part', num, re.IGNORECASE):
        return 1
    if re.search(r'^제?\s*\d+\s*장$|^chapter\s*\d+$', num, re.IGNORECASE):
        return 2
    if re.search(r'^제?\s*\d+\s*절$|^section\s*\d+$', num, re.IGNORECASE):
        return 3
    if re.search(r'^제?\s*\d+\s*[항목]$', num):
        return 4

    # 숫자 계층 기반
    dots = num.count('.')
    if dots == 0:
        return 2 if re.match(r'^\d+$', num) else 1
    elif dots == 1:
        return 3
    else:
        return 4

    return 2


# ─── PDF 추출 ─────────────────────────────────────────────────────────────────

def _extract_pdf(path: Path) -> ExtractedDocument:
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber가 설치되어 있지 않습니다. pip install pdfplumber 실행하세요.")

    doc = ExtractedDocument(
        file_path=str(path),
        file_type="PDF",
        title=path.stem,
    )

    try:
        with pdfplumber.open(path) as pdf:
            doc.total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                is_uncertain = False
                reason = ""

                if len(text.strip()) < 50 and page.images:
                    is_uncertain = True
                    reason = "스캔 이미지로 보임 — OCR 필요"
                    ocr_text = _ocr_pdf_page_image(page)
                    if ocr_text:
                        text = ocr_text
                        is_uncertain = False
                    else:
                        doc.extraction_warnings.append(
                            f"p.{i}: 텍스트 추출 불완전 — 원본 확인 필요"
                        )

                if "?" * 3 in text or "□" * 3 in text:
                    is_uncertain = True
                    reason = "인코딩 오류 의심"
                    doc.extraction_warnings.append(
                        f"p.{i}: 문자 인코딩 문제 — 원본 확인 필요"
                    )

                doc.pages.append(PageContent(
                    page_number=i,
                    text=text.strip(),
                    is_uncertain=is_uncertain,
                    uncertainty_reason=reason,
                ))
    except Exception as e:
        doc.extraction_warnings.append(f"PDF 추출 오류: {e}")

    return doc


def _ocr_pdf_page_image(page) -> str:
    """pdfplumber 페이지에서 이미지를 렌더링하여 OCR 수행"""
    try:
        import easyocr
        img = page.to_image(resolution=200)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            tmp_path = f.name
        reader = _get_ocr_reader()
        results = reader.readtext(tmp_path, detail=0, paragraph=True)
        os.unlink(tmp_path)
        return "\n".join(results)
    except Exception:
        return ""


# ─── HWP 추출 ─────────────────────────────────────────────────────────────────

def _extract_hwp(path: Path) -> ExtractedDocument:
    doc = ExtractedDocument(
        file_path=str(path),
        file_type="HWP",
        title=path.stem,
    )

    text = _hwp_via_hwp5txt(path)
    if not text:
        text = _hwp_via_libreoffice(path, doc)

    if not text:
        doc.extraction_warnings.append(
            "HWP 파일 추출 실패 — hwp5txt 또는 LibreOffice 설치 필요."
        )
        doc.pages.append(PageContent(
            page_number=1, text="",
            is_uncertain=True,
            uncertainty_reason="HWP 추출 실패 — 원본 파일 직접 확인 필요",
        ))
        return doc

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunk_size = max(1, len(paragraphs) // max(1, _estimate_pages(text)))
    for i, chunk_start in enumerate(range(0, len(paragraphs), chunk_size), start=1):
        chunk_text = "\n\n".join(paragraphs[chunk_start:chunk_start + chunk_size])
        doc.pages.append(PageContent(
            page_number=i,
            text=chunk_text,
            is_uncertain=True,
            uncertainty_reason="HWP 페이지 번호 추정값 — 원본 확인 권장",
        ))

    doc.total_pages = len(doc.pages)
    if doc.pages:
        doc.extraction_warnings.append(
            "HWP 파일의 페이지 번호는 추정값입니다. 원본 파일에서 정확한 페이지를 확인하세요."
        )
    return doc


def _hwp_via_hwp5txt(path: Path) -> str:
    try:
        result = subprocess.run(
            ["hwp5txt", str(path)],
            capture_output=True, text=True, timeout=30, encoding="utf-8"
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def _hwp_via_libreoffice(path: Path, doc: ExtractedDocument) -> str:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                ["libreoffice", "--headless", "--convert-to", "docx",
                 "--outdir", tmpdir, str(path)],
                capture_output=True, timeout=60
            )
            if result.returncode != 0:
                return ""
            converted = list(Path(tmpdir).glob("*.docx"))
            if not converted:
                return ""
            return _extract_docx_text(converted[0])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


# ─── DOCX 추출 ────────────────────────────────────────────────────────────────

def _extract_docx(path: Path) -> ExtractedDocument:
    doc = ExtractedDocument(
        file_path=str(path),
        file_type="DOCX",
        title=path.stem,
    )
    text = _extract_docx_text(path)
    if not text:
        doc.extraction_warnings.append("DOCX 추출 실패 — python-docx 설치 확인 필요")
        doc.pages.append(PageContent(
            page_number=1, text="",
            is_uncertain=True, uncertainty_reason="DOCX 추출 실패"
        ))
        return doc

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunk_size = max(10, len(paragraphs) // max(1, _estimate_pages(text)))
    for i, start in enumerate(range(0, len(paragraphs), chunk_size), start=1):
        chunk = "\n".join(paragraphs[start:start + chunk_size])
        doc.pages.append(PageContent(
            page_number=i, text=chunk,
            is_uncertain=True,
            uncertainty_reason="DOCX 페이지 번호 추정값 — 원본 확인 권장",
        ))

    doc.total_pages = len(doc.pages)
    doc.extraction_warnings.append(
        "DOCX 파일의 페이지 번호는 추정값입니다. 원본 파일에서 정확한 페이지를 확인하세요."
    )
    return doc


def _extract_docx_text(path: Path) -> str:
    try:
        from docx import Document
        d = Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)
    except Exception:
        return ""


# ─── 이미지 OCR 추출 ──────────────────────────────────────────────────────────

_ocr_reader = None

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["ko", "en"], gpu=False)
    return _ocr_reader


def _extract_image(path: Path) -> ExtractedDocument:
    doc = ExtractedDocument(
        file_path=str(path),
        file_type="IMAGE",
        title=path.stem,
        total_pages=1,
    )
    try:
        reader = _get_ocr_reader()
        results = reader.readtext(str(path), detail=0, paragraph=True)
        text = "\n".join(results)

        if not text.strip():
            doc.extraction_warnings.append("이미지에서 텍스트를 추출하지 못했습니다 — 원본 이미지 확인 필요")
            doc.pages.append(PageContent(
                page_number=1, text="",
                is_uncertain=True, uncertainty_reason="OCR 텍스트 없음",
            ))
        else:
            confidence_check = len(text) < 100
            doc.pages.append(PageContent(
                page_number=1,
                text=text,
                is_uncertain=confidence_check,
                uncertainty_reason="이미지 품질 불량으로 인한 OCR 불확실" if confidence_check else "",
            ))
    except Exception as e:
        doc.extraction_warnings.append(f"이미지 OCR 오류: {e} — easyocr 설치 확인 필요")
        doc.pages.append(PageContent(
            page_number=1, text="",
            is_uncertain=True, uncertainty_reason=f"OCR 실패: {e}",
        ))

    return doc


# ─── 유틸리티 ─────────────────────────────────────────────────────────────────

def _estimate_pages(text: str) -> int:
    """텍스트 길이로 대략적인 페이지 수 추정 (약 2000자/페이지)"""
    return max(1, len(text) // 2000)
