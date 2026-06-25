"""
style_learner.py — 학습 논문 문체 코퍼스 관리자

논문이 분석될 때마다 실제 원문 문장들을 추출해서 corpus에 누적합니다.
작문 시 이 코퍼스에서 문체 예시를 뽑아 Claude에게 제공합니다.

핵심 원칙:
  - 저장하는 문장은 Claude가 생성한 요약이 아닌 원문에서 추출된 텍스트
  - AI 투의 작위적 표현을 쓰지 않도록 실제 학술 문장을 예시로 제공
"""

from __future__ import annotations
import json
import re
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

CORPUS_PATH = Path(__file__).parent.parent / "memory" / "style_corpus.json"

# 유효한 학술 문장인지 검사하는 최소/최대 글자 수
_MIN_SENT_LEN = 20
_MAX_SENT_LEN = 300

# 걸러낼 AI 투 표현 패턴 — 이런 문장은 코퍼스에 넣지 않음
_AI_CLICHE_RE = re.compile(
    r"이러한 맥락에서|다양한 측면에서|면밀히 검토|시사점을 제공|살펴보도록 하겠다"
    r"|주목할 만하다|종합적 관점|다각적|통합적 접근|본 연구는.*고찰하고자"
    r"|중요한 의미를 지닌다|핵심적인 역할|더불어.*또한.*아울러",
    re.IGNORECASE,
)

# 한국어 문장 종결 패턴
_SENT_END_RE = re.compile(r"[하였었겠]다[.。]?$|[이]다[.。]?$|[없있]다[.。]?$")


# ─── 공개 API ─────────────────────────────────────────────────────────────────

def update_style_corpus(analysis: dict) -> int:
    """
    논문 분석 결과에서 실제 문장들을 추출해 코퍼스에 추가합니다.
    Returns: 새로 추가된 문장 수
    """
    corpus = _load_corpus()

    title       = analysis.get("title", "제목미상")
    doc_type    = analysis.get("document_type", "unknown")
    period      = analysis.get("historical_period", "")
    language    = analysis.get("document_language", "")

    # 한국어·일본어 자료만 문체 학습 대상 (영어 원문은 제외)
    if language in ("en",):
        return 0

    sentences = _extract_sentences(analysis)
    if not sentences:
        return 0

    entry = {
        "title": title,
        "doc_type": doc_type,
        "period": period,
        "added_at": datetime.now().strftime("%Y-%m-%d"),
        "sentences": sentences,
        "connectives": _count_connectives(sentences),
        "avg_len": int(sum(len(s) for s in sentences) / len(sentences)),
    }

    # 같은 제목이 이미 있으면 교체
    corpus["papers"] = [p for p in corpus["papers"] if p.get("title") != title]
    corpus["papers"].append(entry)

    # 글로벌 패턴 갱신
    _update_global_patterns(corpus)

    _save_corpus(corpus)
    return len(sentences)


def get_style_examples(
    n: int = 8,
    doc_type: Optional[str] = None,
    prefer_period: Optional[str] = None,
) -> list[str]:
    """
    코퍼스에서 작문 스타일 예시 문장 n개를 반환합니다.

    doc_type: 우선 참조할 자료 유형 ("academic_paper", "monograph" 등)
    prefer_period: 우선 참조할 시대 ("1950-1953", "일제강점기" 등)
    """
    corpus = _load_corpus()
    if not corpus["papers"]:
        return []

    # 우선순위 풀 구성
    primary   = []
    secondary = []

    for paper in corpus["papers"]:
        sents = paper.get("sentences", [])
        if not sents:
            continue
        if doc_type and paper.get("doc_type") == doc_type:
            primary.extend(sents)
        elif prefer_period and prefer_period in paper.get("period", ""):
            primary.extend(sents)
        else:
            secondary.extend(sents)

    # 우선순위 풀에서 먼저 채우고 나머지는 secondary에서
    pool = primary + secondary
    if not pool:
        return []

    # 중복 제거 후 랜덤 샘플
    pool = list(dict.fromkeys(pool))   # 순서 유지 중복 제거
    random.shuffle(pool)
    return pool[:n]


def get_corpus_stats() -> dict:
    """코퍼스 현황 통계를 반환합니다."""
    corpus = _load_corpus()
    papers = corpus.get("papers", [])
    total_sents = sum(len(p.get("sentences", [])) for p in papers)
    return {
        "paper_count": len(papers),
        "total_sentences": total_sents,
        "papers": [
            {
                "title": p["title"],
                "sentence_count": len(p.get("sentences", [])),
                "avg_len": p.get("avg_len", 0),
                "period": p.get("period", ""),
            }
            for p in papers
        ],
        "top_connectives": dict(
            sorted(
                corpus.get("global_patterns", {}).get("connectives", {}).items(),
                key=lambda x: x[1], reverse=True
            )[:15]
        ),
    }


def get_anti_patterns() -> list[str]:
    """
    AI 투의 작위적 표현 목록 반환 — 작문 프롬프트의 '금지 표현'으로 사용.
    코퍼스에 실제로 드물게 등장하는 표현들을 우선 반환합니다.
    """
    corpus = _load_corpus()
    connectives = corpus.get("global_patterns", {}).get("connectives", {})

    # 고정 금지 목록 (AI가 자주 쓰는 상투적 표현)
    fixed = [
        "이러한 맥락에서",
        "다양한 측면에서 살펴볼 수 있다",
        "면밀히 검토할 필요가 있다",
        "시사점을 제공한다",
        "본 연구는 ~고찰하고자 한다",
        "중요한 의미를 지닌다",
        "통합적/종합적 관점에서",
        "살펴보도록 하겠다",
        "주목할 만하다",
        "다각적인 시각",
        "핵심적인 역할을 담당하였다",
        "심층적으로 분석",
        "논의가 필요하다",
        "이는 ~라는 점에서 의의가 있다",
    ]
    return fixed


# ─── 내부 함수 ────────────────────────────────────────────────────────────────

def _extract_sentences(analysis: dict) -> list[str]:
    """분석 결과에서 실제 문장들을 추출합니다."""
    candidates: list[str] = []

    # 1순위: footnotes 원문 (가장 신뢰도 높음 — 논문 원문 그대로)
    for fn in analysis.get("footnotes", []):
        text = fn.get("text", "").strip()
        for sent in _split_sentences(text):
            if _is_valid_sentence(sent):
                candidates.append(sent)

    # 2순위: key_arguments (AI 요약이지만 학술 문체를 어느 정도 모방)
    for arg in analysis.get("key_arguments", []):
        for sent in _split_sentences(arg):
            if _is_valid_sentence(sent):
                candidates.append(sent)

    # 3순위: main_thesis (하나의 긴 문장일 때 유용)
    thesis = analysis.get("main_thesis", "")
    for sent in _split_sentences(thesis):
        if _is_valid_sentence(sent) and len(sent) > 40:
            candidates.append(sent)

    # AI 투 표현이 있는 문장 제거
    filtered = [s for s in candidates if not _AI_CLICHE_RE.search(s)]

    # 최대 20문장 (논문당)
    return list(dict.fromkeys(filtered))[:20]


def _split_sentences(text: str) -> list[str]:
    """텍스트를 문장 단위로 분리합니다."""
    if not text:
        return []
    # 마침표·느낌표·물음표 기준 분리 (단, 소수점·약어 등 예외 처리)
    parts = re.split(r'(?<=[다요음])\.(?=\s)|(?<=[다요음])\s{2,}|(?<=。)', text)
    return [p.strip() for p in parts if p.strip()]


def _is_valid_sentence(text: str) -> bool:
    """유효한 학술 문장인지 검사합니다."""
    t = text.strip()
    if len(t) < _MIN_SENT_LEN or len(t) > _MAX_SENT_LEN:
        return False
    # 한글이 20% 이상 포함되어야 함
    korean_ratio = len(re.findall(r'[가-힣]', t)) / max(len(t), 1)
    if korean_ratio < 0.2:
        return False
    # 너무 많은 괄호나 특수문자는 제외
    if t.count('(') > 3 or t.count('[') > 3:
        return False
    return True


def _count_connectives(sentences: list[str]) -> dict:
    """문장들에서 접속사/연결어 사용 빈도를 카운트합니다."""
    CONNECTIVES = [
        "그러나", "한편", "따라서", "그러므로", "이에 따라", "이에 반해",
        "이와 달리", "반면", "또한", "특히", "나아가", "이처럼", "이러한",
        "즉", "다만", "그런데", "이와 같이", "하지만", "더욱이", "이에",
        "하여", "바면", "그리하여", "그 결과",
    ]
    counts: dict[str, int] = {}
    text = " ".join(sentences)
    for conn in CONNECTIVES:
        cnt = text.count(conn)
        if cnt > 0:
            counts[conn] = cnt
    return counts


def _update_global_patterns(corpus: dict) -> None:
    """전체 코퍼스의 글로벌 패턴(접속사 통계 등)을 갱신합니다."""
    merged: dict[str, int] = {}
    for paper in corpus["papers"]:
        for conn, cnt in paper.get("connectives", {}).items():
            merged[conn] = merged.get(conn, 0) + cnt
    corpus.setdefault("global_patterns", {})["connectives"] = merged


def _load_corpus() -> dict:
    if CORPUS_PATH.exists():
        try:
            return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"papers": [], "global_patterns": {"connectives": {}}}


def _save_corpus(corpus: dict) -> None:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
