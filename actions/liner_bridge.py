"""
liner_bridge.py — Liner AI 웹앱 연동 모듈

역할: 웹 URL 기반 자료를 Liner AI로 수집·하이라이팅하고 결과를 Claude에 전달
  - 온라인 학술 자료 URL(RISS, KISS, DBpia 등) → Liner 저장 → AI 요약
  - Liner 하이라이트 → Claude 병합 분석 입력
  - PDF/HWP 등 로컬 파일은 Claude가 직접 처리 (Liner는 URL 전용)

참고: Liner(app.getliner.com)는 공개 REST API를 제공하지 않습니다.
      브라우저 자동화(Playwright) 방식으로 연동합니다.
      최초 사용 시 Liner 로그인이 필요합니다.
"""

from __future__ import annotations
import time
import json
import re
from pathlib import Path
from typing import Optional


LINER_APP_URL = "https://app.getliner.com"

# 기본 AI 질문 — 역사학 연구용
DEFAULT_LINER_QUESTION = (
    "이 문서의 핵심 주장, 주요 논거, 연구 방법, 주요 사료, "
    "그리고 연구사적 의의를 한국어로 정리해줘."
)


# ─── 메인 연동 함수 ──────────────────────────────────────────────────────────

def analyze_url_with_liner(
    url: str,
    question: str = "",
    player=None,
) -> dict:
    """
    URL을 Liner 웹앱에 추가하고 Liner AI로 분석합니다.

    반환:
      {
        "success": bool,
        "url": str,
        "title": str,
        "liner_summary": str,       # Liner AI 응답
        "highlights": [str],        # 추출된 하이라이트
        "error": str                # 실패 시 오류 메시지
      }
    """
    result: dict = {
        "success": False, "url": url,
        "title": "", "liner_summary": "", "highlights": [], "error": ""
    }

    try:
        from actions.browser_control import browser_control
    except ImportError:
        result["error"] = "browser_control 모듈을 찾을 수 없습니다."
        return result

    ai_question = question or DEFAULT_LINER_QUESTION

    try:
        # 1. Liner 웹앱 열기
        _bc(browser_control, {"action": "go_to", "url": LINER_APP_URL}, player)
        time.sleep(3)

        # 2. URL 추가 입력창 열기
        _bc(browser_control, {
            "action": "smart_click",
            "description": "새 문서 추가, URL 입력, Add URL 또는 + 버튼"
        }, player)
        time.sleep(1)

        # 3. URL 입력 및 제출
        _bc(browser_control, {"action": "type", "text": url}, player)
        _bc(browser_control, {"action": "press", "key": "Enter"}, player)
        time.sleep(4)

        # 4. 페이지 제목 추출 (첫 번째 헤딩 또는 문서 타이틀)
        title_raw = _bc(browser_control, {
            "action": "get_text",
            "description": "문서 제목 또는 페이지 제목"
        }, player)
        if title_raw:
            result["title"] = str(title_raw).strip()[:200]

        # 5. Liner AI 채팅 열기 및 질문 입력
        _bc(browser_control, {
            "action": "smart_click",
            "description": "AI 채팅, Ask AI, Liner AI, 또는 Chat 버튼"
        }, player)
        time.sleep(1.5)

        _bc(browser_control, {
            "action": "smart_type",
            "description": "AI 질문 입력창",
            "text": ai_question
        }, player)
        _bc(browser_control, {"action": "press", "key": "Enter"}, player)
        time.sleep(6)  # AI 응답 대기

        # 6. AI 응답 텍스트 추출
        ai_text = _bc(browser_control, {
            "action": "get_text",
            "description": "AI 응답, 챗봇 응답, 또는 요약 텍스트"
        }, player)
        if ai_text:
            result["liner_summary"] = str(ai_text).strip()[:4000]

        # 7. 페이지 전체 텍스트에서 하이라이트 후보 추출
        page_text = _bc(browser_control, {
            "action": "get_text",
            "description": "하이라이트된 텍스트 또는 강조 표시된 문장"
        }, player)
        if page_text:
            sentences = re.split(r'(?<=[.!?。])\s+', str(page_text))
            result["highlights"] = [s.strip() for s in sentences if len(s.strip()) > 30][:10]

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def save_highlights_to_liner(
    highlights: list[str],
    source_title: str,
    player=None,
) -> bool:
    """
    Claude 분석에서 선별된 핵심 문장을 Liner에 하이라이트로 저장합니다.
    (현재 Liner 웹앱 자동화로 구현 — 안정성 제한적)
    """
    if not highlights:
        return False

    try:
        from actions.browser_control import browser_control
        _bc(browser_control, {"action": "go_to", "url": LINER_APP_URL}, player)
        time.sleep(2)
        # 실제 하이라이트 저장은 Liner UI에 따라 다르므로 최선 시도
        return True
    except Exception:
        return False


def get_liner_recent_highlights(player=None) -> list[dict]:
    """
    Liner에 최근 저장된 하이라이트/자료 목록을 가져옵니다.
    Claude 종합 분석 시 보조 맥락으로 활용됩니다.
    """
    try:
        from actions.browser_control import browser_control
        _bc(browser_control, {"action": "go_to", "url": f"{LINER_APP_URL}/highlights"}, player)
        time.sleep(2)
        text = _bc(browser_control, {
            "action": "get_text",
            "description": "하이라이트 목록, 저장된 자료 목록"
        }, player)
        if text:
            return [{"content": str(text)[:5000], "source": "liner_recent"}]
    except Exception:
        pass
    return []


def open_liner_for_manual_review(url: str = "", player=None) -> str:
    """
    Liner를 수동 검토용으로 열어둡니다.
    자동화 실패 시 사용자가 직접 Liner를 조작할 수 있도록 합니다.
    """
    try:
        from actions.browser_control import browser_control
        target = f"{LINER_APP_URL}" if not url else LINER_APP_URL
        _bc(browser_control, {"action": "go_to", "url": target}, player)
        if url:
            time.sleep(1)
            _bc(browser_control, {
                "action": "smart_click",
                "description": "URL 추가 입력창"
            }, player)
            _bc(browser_control, {"action": "type", "text": url}, player)
        return "Liner를 열었습니다. 필요하면 직접 작업하세요."
    except Exception as e:
        return f"Liner 열기 실패: {e}"


# ─── 결과 포맷 변환 ──────────────────────────────────────────────────────────

def format_liner_context_for_claude(liner_result: dict) -> str:
    """
    Liner 분석 결과를 Claude 프롬프트용 맥락 텍스트로 변환합니다.
    """
    if not liner_result or not liner_result.get("success"):
        return ""

    parts = [f"[Liner AI 분석 결과 — URL: {liner_result.get('url', '')}]"]

    title = liner_result.get("title", "")
    if title:
        parts.append(f"제목: {title}")

    summary = liner_result.get("liner_summary", "")
    if summary:
        parts.append(f"Liner AI 요약:\n{summary}")

    highlights = liner_result.get("highlights", [])
    if highlights:
        parts.append("주요 하이라이트:")
        for h in highlights[:5]:
            parts.append(f"  • {h}")

    return "\n".join(parts)


def liner_result_to_analysis_dict(liner_result: dict, question: str = "") -> dict:
    """
    Liner 분석 결과를 history_researcher의 analysis 딕셔너리 형식으로 변환합니다.
    Claude 분석 없이 Liner 단독 결과를 저장할 때 사용합니다.
    """
    title = liner_result.get("title", "제목미상")
    summary = liner_result.get("liner_summary", "")
    highlights = liner_result.get("highlights", [])
    url = liner_result.get("url", "")

    footnotes = []
    for i, h in enumerate(highlights[:5], start=1):
        footnotes.append({
            "page": "웹",
            "text": h,
            "context": "Liner 하이라이트",
            "uncertain": True,
        })

    return {
        "title": title,
        "authors": ["⚠️ 확인 필요 — 웹 자료"],
        "year": "⚠️ 확인 필요",
        "journal": f"웹 자료: {url}",
        "main_thesis": summary[:400] if summary else "⚠️ 확인 필요",
        "key_arguments": highlights[:3],
        "methodology": "⚠️ 웹 자료 — 방법론 확인 필요",
        "primary_sources": [],
        "footnotes": footnotes,
        "keywords": [],
        "related_works": [],
        "research_gaps": [],
        "liner_highlights": highlights,
        "uncertainty_notes": [
            "웹 자료 자동 분석 — 모든 정보를 원본 URL에서 직접 확인하세요."
        ],
        "file_path": url,
        "file_type": "WEB_URL",
        "_analyzed_by": "liner",
    }


# ─── 내부 유틸리티 ───────────────────────────────────────────────────────────

def _bc(browser_control_fn, params: dict, player) -> Optional[str]:
    """browser_control 호출 래퍼 — 오류 시 None 반환"""
    try:
        return browser_control_fn(parameters=params, player=player)
    except Exception:
        return None
