"""
conversation_log.py — 대화 내용 저장 및 세션 간 컨텍스트 공유

세션이 재시작되어도 최근 대화 맥락을 시스템 프롬프트에 주입하여
AI가 이전 대화 흐름을 이어받을 수 있게 합니다.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

_BASE     = Path(__file__).resolve().parent
_LOG_PATH = _BASE / "conversation_history.json"
_PAT_PATH = _BASE / "learned_patterns.json"
_MAX_TURNS = 60   # 롤링 윈도우 (초과 시 오래된 것부터 삭제)
_lock      = threading.Lock()


# ── 대화 기록 ─────────────────────────────────────────────────────────────────

def _load_raw() -> list[dict]:
    if not _LOG_PATH.exists():
        return []
    try:
        data = json.loads(_LOG_PATH.read_text("utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def append_turn(role: str, text: str) -> None:
    """대화 한 턴을 기록합니다. role: 'user' | 'assistant'"""
    if not text or not text.strip():
        return
    with _lock:
        turns = _load_raw()
        turns.append({
            "role": role,
            "text": text.strip()[:600],
            "ts":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        if len(turns) > _MAX_TURNS:
            turns = turns[-_MAX_TURNS:]
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LOG_PATH.write_text(
            json.dumps(turns, ensure_ascii=False, indent=2), "utf-8"
        )


def get_recent_context(n: int = 20) -> str:
    """
    최근 n개 턴을 시스템 프롬프트 삽입용 문자열로 반환합니다.
    세션이 바뀌어도 맥락이 이어지도록 _build_config()에서 호출하세요.
    """
    turns = _load_raw()
    if not turns:
        return ""
    recent = turns[-n:]
    lines  = ["[최근 대화 기록 — 맥락 파악 및 자연스러운 이어받기에 활용하세요]"]
    for t in recent:
        role_label = "원준" if t["role"] == "user" else "비서"
        lines.append(f"[{t.get('ts', '')}] {role_label}: {t['text']}")
    return "\n".join(lines) + "\n\n"


def get_stats() -> dict:
    turns = _load_raw()
    return {
        "total_turns":  len(turns),
        "oldest": turns[0]["ts"]  if turns else None,
        "newest": turns[-1]["ts"] if turns else None,
    }


# ── 학습 패턴 (명령 교정 자동 기록) ──────────────────────────────────────────

_CORRECTION_TRIGGERS = {
    "아니야", "아니에요", "그게 아니야", "그게 아니에요",
    "잘못됐어", "틀렸어", "잘못됐어요", "틀렸어요",
    "그만해", "닥쳐", "다시 해줘", "다시해줘",
}


def _load_patterns() -> list[dict]:
    if not _PAT_PATH.exists():
        return []
    try:
        data = json.loads(_PAT_PATH.read_text("utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def record_correction(user_said: str, last_assistant: str) -> None:
    """
    사용자가 교정 표현을 사용했을 때 직전 AI 응답과 함께 패턴을 기록합니다.
    누적 패턴은 get_learned_patterns_context()로 시스템 프롬프트에 주입됩니다.
    """
    with _lock:
        patterns = _load_patterns()
        patterns.append({
            "ts":        datetime.now().strftime("%Y-%m-%d %H:%M"),
            "user":      user_said.strip()[:200],
            "wrong_ai":  last_assistant.strip()[:300],
        })
        # 최근 30개만 유지
        if len(patterns) > 30:
            patterns = patterns[-30:]
        _PAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PAT_PATH.write_text(
            json.dumps(patterns, ensure_ascii=False, indent=2), "utf-8"
        )


def is_correction(text: str) -> bool:
    return any(t in text for t in _CORRECTION_TRIGGERS)


def get_learned_patterns_context() -> str:
    """교정 이력을 시스템 프롬프트용 문자열로 반환합니다."""
    patterns = _load_patterns()
    if not patterns:
        return ""
    recent = patterns[-10:]
    lines  = ["[과거 교정 이력 — 같은 실수를 반복하지 마세요]"]
    for p in recent:
        lines.append(f"[{p.get('ts','')}] 원준이 교정: \"{p['user']}\"")
        lines.append(f"  → 잘못된 AI 응답: \"{p['wrong_ai'][:120]}\"")
    return "\n".join(lines) + "\n\n"
