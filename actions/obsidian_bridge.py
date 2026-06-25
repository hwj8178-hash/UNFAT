"""
obsidian_bridge.py — 옵시디안 볼트 연동 모듈

기능:
  - 분석된 논문/자료를 옵시디안 마크다운 노트로 저장
  - 노트 간 위키링크(wiki-link) 자동 생성
  - 개념 맵 및 연구자 네트워크 시각화
  - 지식 그래프 업데이트 및 추론 지원
  - 연구사 공백 및 새로운 문제의식 도출 인덱스 관리
"""

from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional


VAULT_CONFIG_KEY = "obsidian_vault_path"


def get_vault_path() -> Optional[Path]:
    """설정에서 옵시디안 볼트 경로를 불러옵니다."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        vault = cfg.get(VAULT_CONFIG_KEY, "")
        if vault:
            p = Path(vault).expanduser()
            if p.exists():
                return p
    except Exception:
        pass
    return None


def set_vault_path(vault_path: str) -> str:
    """옵시디안 볼트 경로를 설정에 저장합니다."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "api_keys.json"
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}
    cfg[VAULT_CONFIG_KEY] = str(vault_path)
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"옵시디안 볼트 경로가 설정되었습니다: {vault_path}"


# ─── 노트 생성/업데이트 ──────────────────────────────────────────────────────

def save_research_note(analysis: dict, vault_path: Optional[Path] = None) -> str:
    """
    분석 결과를 옵시디안 마크다운 노트로 저장합니다.
    analysis 딕셔너리 구조:
      title, file_path, file_type, authors, year, journal,
      main_thesis, key_arguments, methodology, primary_sources,
      footnotes, research_gaps, keywords, related_works, uncertainty_notes
    """
    vault = vault_path or get_vault_path()
    if not vault:
        return "❌ 옵시디안 볼트 경로가 설정되지 않았습니다. '옵시디안 볼트 경로 설정 [경로]' 명령을 사용하세요."

    # 저장 디렉터리 구조
    research_dir = vault / "역사학연구" / "논문분석"
    research_dir.mkdir(parents=True, exist_ok=True)

    title = analysis.get("title", "제목없음")
    safe_title = re.sub(r'[\\/:*?"<>|]', "_", title)
    note_path = research_dir / f"{safe_title}.md"

    content = _build_research_note(analysis)
    note_path.write_text(content, encoding="utf-8")

    # 키워드 인덱스 및 연결 업데이트
    _update_keyword_index(vault, title, analysis.get("keywords", []))
    _update_author_index(vault, title, analysis.get("authors", []))
    _update_concept_map(vault, title, analysis)

    return f"✅ 옵시디안 노트 저장 완료: {note_path}\n연결된 개념: {', '.join(analysis.get('keywords', [])[:5])}"


def _build_research_note(a: dict) -> str:
    """마크다운 형식의 연구 노트 생성"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = a.get("title", "제목없음")
    authors = ", ".join(a.get("authors", ["저자미상"]))
    year = a.get("year", "연도미상")
    journal = a.get("journal", "")
    source_file = a.get("file_path", "")

    # YAML 프론트매터
    tags = ["역사학연구"]
    tags += [f"키워드/{k}" for k in a.get("keywords", [])]
    tags += [f"저자/{au.replace(' ', '_')}" for au in a.get("authors", [])]
    tags_str = "\n".join(f"  - {t}" for t in tags)

    keywords_links = " ".join(f"[[{k}]]" for k in a.get("keywords", []))
    related_links = "\n".join(f"- [[{r}]]" for r in a.get("related_works", []))
    author_links = ", ".join(f"[[저자/{au}]]" for au in a.get("authors", []))

    # 각주 섹션 구성
    footnotes_section = _build_footnotes_section(a.get("footnotes", []))

    # 불확실 정보 경고
    uncertainty_section = ""
    if a.get("uncertainty_notes"):
        uncertainty_section = "\n## ⚠️ 확인 필요 항목\n"
        for note in a["uncertainty_notes"]:
            uncertainty_section += f"> ⚠️ **자세한 확인 필요**: {note}\n"

    # 연구 공백 섹션
    gaps_section = ""
    if a.get("research_gaps"):
        gaps_section = "\n## 🔍 연구사 공백 및 새로운 문제의식\n"
        for gap in a["research_gaps"]:
            gaps_section += f"- {gap}\n"

    note = f"""---
created: {now}
title: "{title}"
authors: [{authors}]
year: {year}
journal: "{journal}"
source_file: "{source_file}"
tags:
{tags_str}
---

# {title}

**저자**: {author_links}
**출판연도**: {year}
**출처**: {journal}
**원본파일**: `{source_file}`

---

## 핵심 주장 (Main Thesis)

{a.get("main_thesis", "_추출되지 않음_")}

---

## 주요 논거 (Key Arguments)

{_list_to_md(a.get("key_arguments", []))}

---

## 연구 방법론

{a.get("methodology", "_명시되지 않음_")}

---

## 주요 사료 (Primary Sources)

{_list_to_md(a.get("primary_sources", []))}

---

## 핵심 개념 및 키워드

{keywords_links}

---

## 관련 연구

{related_links or "_관련 연구 없음_"}

{gaps_section}
{uncertainty_section}
{footnotes_section}

---

## 메타데이터

- **분석 일시**: {now}
- **원본 파일**: [{source_file}]({source_file})
"""
    return note


def _build_footnotes_section(footnotes: list) -> str:
    """각주 섹션 구성"""
    if not footnotes:
        return ""

    lines = ["## 📌 원문 인용 및 각주 (Footnotes)\n"]
    lines.append("> 아래 인용문은 원문에서 직접 추출되었습니다. 각주 작성 시 그대로 사용 가능합니다.\n")

    for i, fn in enumerate(footnotes, start=1):
        page = fn.get("page", "?")
        text = fn.get("text", "")
        context = fn.get("context", "")
        is_uncertain = fn.get("uncertain", False)

        uncertainty_mark = " ⚠️[원본 확인 필요]" if is_uncertain else ""
        lines.append(f"**[{i}]** (p.{page}){uncertainty_mark}  ")
        if context:
            lines.append(f"*맥락: {context}*  ")
        lines.append(f'> "{text}"\n')

    return "\n".join(lines)


def _list_to_md(items: list) -> str:
    if not items:
        return "_없음_"
    return "\n".join(f"- {item}" for item in items)


# ─── 인덱스 및 연결 관리 ─────────────────────────────────────────────────────

def _update_keyword_index(vault: Path, note_title: str, keywords: list[str]) -> None:
    """키워드별 인덱스 노트 업데이트"""
    index_dir = vault / "역사학연구" / "개념인덱스"
    index_dir.mkdir(parents=True, exist_ok=True)

    for kw in keywords:
        safe_kw = re.sub(r'[\\/:*?"<>|]', "_", kw)
        idx_path = index_dir / f"{safe_kw}.md"

        if idx_path.exists():
            content = idx_path.read_text(encoding="utf-8")
            if f"[[{note_title}]]" not in content:
                content += f"\n- [[{note_title}]]"
        else:
            content = f"""# {kw}

## 이 개념을 다루는 논문/자료

- [[{note_title}]]
"""
        idx_path.write_text(content, encoding="utf-8")


def _update_author_index(vault: Path, note_title: str, authors: list[str]) -> None:
    """저자별 인덱스 노트 업데이트"""
    author_dir = vault / "역사학연구" / "저자인덱스"
    author_dir.mkdir(parents=True, exist_ok=True)

    for author in authors:
        safe_author = re.sub(r'[\\/:*?"<>|]', "_", author)
        author_path = author_dir / f"{safe_author}.md"

        if author_path.exists():
            content = author_path.read_text(encoding="utf-8")
            if f"[[{note_title}]]" not in content:
                content += f"\n- [[{note_title}]]"
        else:
            content = f"""# {author}

## 저작 목록

- [[{note_title}]]
"""
        author_path.write_text(content, encoding="utf-8")


def _update_concept_map(vault: Path, note_title: str, analysis: dict) -> None:
    """지식 그래프 데이터 업데이트 (JSON 형식)"""
    graph_path = vault / "역사학연구" / ".knowledge_graph.json"

    try:
        if graph_path.exists():
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
        else:
            graph = {"nodes": {}, "edges": [], "research_gaps": [], "last_updated": ""}
    except Exception:
        graph = {"nodes": {}, "edges": [], "research_gaps": [], "last_updated": ""}

    # 노드 추가/업데이트
    graph["nodes"][note_title] = {
        "type": "paper",
        "authors": analysis.get("authors", []),
        "year": analysis.get("year", ""),
        "keywords": analysis.get("keywords", []),
        "main_thesis": analysis.get("main_thesis", "")[:200],
    }

    # 키워드 간 엣지 추가
    keywords = analysis.get("keywords", [])
    for kw in keywords:
        if kw not in graph["nodes"]:
            graph["nodes"][kw] = {"type": "concept"}
        edge = {"from": note_title, "to": kw, "relation": "discusses"}
        if edge not in graph["edges"]:
            graph["edges"].append(edge)

    # 관련 연구 간 엣지 추가
    for related in analysis.get("related_works", []):
        edge = {"from": note_title, "to": related, "relation": "related_to"}
        if edge not in graph["edges"]:
            graph["edges"].append(edge)

    # 연구 공백 누적
    for gap in analysis.get("research_gaps", []):
        gap_entry = {"gap": gap, "identified_in": note_title,
                     "date": datetime.now().strftime("%Y-%m-%d")}
        if gap_entry not in graph["research_gaps"]:
            graph["research_gaps"].append(gap_entry)

    graph["last_updated"] = datetime.now().isoformat()
    graph_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")


# ─── 연구사 공백 분석 ────────────────────────────────────────────────────────

def analyze_research_landscape(vault_path: Optional[Path] = None) -> str:
    """
    지식 그래프를 분석하여 연구사 공백과 새로운 문제의식을 도출합니다.
    """
    vault = vault_path or get_vault_path()
    if not vault:
        return "❌ 옵시디안 볼트 경로가 설정되지 않았습니다."

    graph_path = vault / "역사학연구" / ".knowledge_graph.json"
    if not graph_path.exists():
        return "❌ 아직 분석된 논문이 없습니다. 먼저 논문을 분석해주세요."

    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except Exception:
        return "❌ 지식 그래프 로드 실패"

    papers = {k: v for k, v in graph["nodes"].items() if v.get("type") == "paper"}
    concepts = {k: v for k, v in graph["nodes"].items() if v.get("type") == "concept"}

    # 개념별 연구 빈도 계산
    concept_freq: dict[str, list[str]] = {}
    for edge in graph["edges"]:
        if edge.get("relation") == "discusses":
            concept_freq.setdefault(edge["to"], []).append(edge["from"])

    # 많이 다뤄진 주제와 적게 다뤄진 주제 분리
    sorted_concepts = sorted(concept_freq.items(), key=lambda x: len(x[1]), reverse=True)
    hot_topics = sorted_concepts[:5]
    underresearched = [(c, ps) for c, ps in sorted_concepts if len(ps) <= 1]

    # 연도별 분포
    year_dist: dict[str, int] = {}
    for p_data in papers.values():
        yr = str(p_data.get("year", "미상"))
        year_dist[yr] = year_dist.get(yr, 0) + 1

    # 누적된 연구 공백
    gaps = graph.get("research_gaps", [])

    lines = [
        f"# 연구사 현황 분석 ({datetime.now().strftime('%Y-%m-%d')})\n",
        f"**분석 논문 수**: {len(papers)}편",
        f"**주요 개념 수**: {len(concepts)}개\n",
        "## 🔥 집중 연구 주제 (연구가 많은 분야)",
    ]
    for concept, papers_list in hot_topics:
        lines.append(f"- **{concept}** — {len(papers_list)}편: {', '.join(f'[[{p}]]' for p in papers_list)}")

    lines.append("\n## 🔍 연구 공백 (미개척 분야)")
    if underresearched:
        for concept, papers_list in underresearched[:10]:
            lines.append(f"- **{concept}** — {len(papers_list)}편만 존재")
    else:
        lines.append("- 공백 데이터 부족 (논문을 더 추가해주세요)")

    lines.append("\n## 📅 연구 시기별 분포")
    for yr, cnt in sorted(year_dist.items()):
        lines.append(f"- {yr}: {cnt}편")

    lines.append("\n## 💡 식별된 새로운 문제의식")
    if gaps:
        seen = set()
        for gap_entry in gaps[-20:]:
            gap_text = gap_entry.get("gap", "")
            if gap_text and gap_text not in seen:
                seen.add(gap_text)
                source = gap_entry.get("identified_in", "")
                lines.append(f"- {gap_text} *(출처: [[{source}]])*")
    else:
        lines.append("- 아직 식별된 연구 공백 없음")

    result_text = "\n".join(lines)

    # 분석 결과를 옵시디안에 저장
    report_path = vault / "역사학연구" / "연구사분석.md"
    report_path.write_text(result_text, encoding="utf-8")

    return result_text


# ─── 학습 업데이트 ───────────────────────────────────────────────────────────

def get_knowledge_summary(vault_path: Optional[Path] = None) -> str:
    """현재 지식 그래프의 요약을 LLM 프롬프트용으로 반환합니다."""
    vault = vault_path or get_vault_path()
    if not vault:
        return ""

    graph_path = vault / "역사학연구" / ".knowledge_graph.json"
    if not graph_path.exists():
        return ""

    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    papers = [k for k, v in graph["nodes"].items() if v.get("type") == "paper"]
    concepts = [k for k, v in graph["nodes"].items() if v.get("type") == "concept"]
    gaps = [g["gap"] for g in graph.get("research_gaps", [])[-10:]]

    summary = f"[현재 연구 데이터베이스: 논문 {len(papers)}편, 개념 {len(concepts)}개"
    if papers:
        summary += f". 최근 분석: {', '.join(papers[-3:])}"
    if gaps:
        summary += f". 식별된 연구 공백: {'; '.join(gaps[:3])}"
    summary += "]"
    return summary
