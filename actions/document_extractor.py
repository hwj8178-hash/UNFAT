"""
document_extractor.py — 역사학 연구용 문서 텍스트 추출 엔진

지원 형식:
  PDF    → 페이지별 텍스트 추출 (pdfplumber)
  HWP    → 한글 파일 텍스트 추출 (hwp5 / LibreOffice 변환)
  DOCX   → 워드 파일 텍스트 추출 (python-docx)
  JPG/PNG → OCR 텍스트 추출 (easyocr, 한국어 지원)
"""

from __future__ import annotations
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PageContent:
    """단일 페이지/섹션의 텍스트와 메타데이터"""
    page_number: int
    text: str
    is_uncertain: bool = False
    uncertainty_reason: str = ""


@dataclass
class ExtractedDocument:
    """추출된 전체 문서"""
    file_path: str
    file_type: str
    title: str
    pages: list[PageContent] = field(default_factory=list)
    total_pages: int = 0
    extraction_warnings: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def total_chars(self) -> int:
        """추출된 전체 텍스트의 총 글자 수"""
        return sum(len(p.text) for p in self.pages)

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

    def get_page_chunks(
        self, chunk_chars: int = 130_000, overlap_chars: int = 8_000
    ) -> list[str]:
        """
        대용량 문서를 위한 청크 단위 텍스트 분할.
        각 청크는 [p.N] 경계를 존중하며, 문맥 유지를 위해 overlap_chars만큼 중첩합니다.
        """
        full = self.get_text_with_pages()
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


def extract_document(file_path: str) -> ExtractedDocument:
    """
    파일 경로를 받아 ExtractedDocument를 반환합니다.
    파일 형식을 자동으로 감지하여 적절한 추출기를 사용합니다.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    elif ext in (".hwp", ".hwpx"):
        return _extract_hwp(path)
    elif ext in (".docx", ".doc"):
        return _extract_docx(path)
    elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        return _extract_image(path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext}")


# ─── PDF 추출 ────────────────────────────────────────────────────────────────

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

                # 텍스트가 너무 짧으면 스캔 이미지일 가능성
                if len(text.strip()) < 50 and page.images:
                    is_uncertain = True
                    reason = "스캔 이미지로 보임 — OCR 필요"
                    # 이미지 OCR 시도
                    ocr_text = _ocr_pdf_page_image(page)
                    if ocr_text:
                        text = ocr_text
                        is_uncertain = False
                    else:
                        doc.extraction_warnings.append(
                            f"p.{i}: 텍스트 추출 불완전 — 원본 확인 필요"
                        )

                # 인코딩 오류 감지
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


# ─── HWP 추출 ────────────────────────────────────────────────────────────────

def _extract_hwp(path: Path) -> ExtractedDocument:
    doc = ExtractedDocument(
        file_path=str(path),
        file_type="HWP",
        title=path.stem,
    )

    # 방법 1: hwp5txt 커맨드라인 도구 사용
    text = _hwp_via_hwp5txt(path)

    # 방법 2: LibreOffice로 DOCX 변환 후 추출
    if not text:
        text = _hwp_via_libreoffice(path, doc)

    if not text:
        doc.extraction_warnings.append(
            "HWP 파일 추출 실패 — hwp5txt 또는 LibreOffice 설치 필요. 원본 파일 직접 확인 필요."
        )
        doc.pages.append(PageContent(
            page_number=1,
            text="",
            is_uncertain=True,
            uncertainty_reason="HWP 추출 실패 — 원본 파일 직접 확인 필요",
        ))
        return doc

    # HWP는 페이지 경계를 정확히 알기 어려우므로 섹션으로 분할
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
    """hwp5txt CLI로 HWP 텍스트 추출"""
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
    """LibreOffice로 DOCX 변환 후 텍스트 추출"""
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


# ─── DOCX 추출 ───────────────────────────────────────────────────────────────

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

    # DOCX는 페이지 경계 감지가 어려움 — 섹션 나누기
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunk_size = max(10, len(paragraphs) // max(1, _estimate_pages(text)))
    for i, start in enumerate(range(0, len(paragraphs), chunk_size), start=1):
        chunk = "\n".join(paragraphs[start:start + chunk_size])
        doc.pages.append(PageContent(
            page_number=i,
            text=chunk,
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


# ─── 이미지 OCR 추출 ─────────────────────────────────────────────────────────

_ocr_reader = None

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        # 한국어 + 영어 지원
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
