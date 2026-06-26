"""
dad_jokes_manager.py — 아재개그 목록 관리 및 웹 업데이트
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_BASE       = Path(__file__).resolve().parent.parent
_JOKES_PATH = _BASE / "memory" / "dad_jokes.json"

_BUILTIN: list[str] = [
    "세상에서 가장 빠른 닭이 뭔지 알아요? 후다닥!",
    "곰이 물에 빠지면 뭐가 될까요? 곰탕이요!",
    "자전거가 왜 넘어졌을까요? 두발자전거라서요!",
    "달력이 왜 슬플까요? 날이 가니까요!",
    "빵집에서 빵이 다 팔리면 뭐라고 할까요? 빵구났다!",
    "세상에서 제일 무거운 게 뭔지 알아요? 눈꺼풀이요. 눈이 무겁거든요!",
    "왜 수학책은 항상 우울할까요? 문제가 너무 많아서요!",
    "지구가 둥근 이유가 뭔지 알아요? 모난 데가 없어서요!",
    "슬픈 초콜릿은 뭘까요? 다크 초콜릿이요!",
    "피자가 결혼했어요. 잘 됐나요? 화덕에서 맺어졌으니 금실이 좋죠!",
    "가장 빨리 배우는 사람은요? 배달부요!",
    "왜 개미는 일을 열심히 할까요? 개미하니까요!",
    "세상에서 가장 긴 다리는요? 다리미요!",
    "냉장고가 웃으면 어떻게 될까요? 냉소적이 되죠!",
    "소가 웃으면 뭐가 될까요? 웃소요!",
    "왜 핸드폰은 떨어지면 안 될까요? 핸드폰이니까요!",
    "세상에서 가장 짧은 사람은? 단발머리요!",
    "왜 바나나는 외롭지 않을까요? 항상 한 다발이니까요!",
    "거북이가 넘어지면 뭐라고 할까요? 터틀링이요!",
    "가장 게으른 산은? 누워있는 산이요, 즉 누산이죠!",
]


def load_jokes() -> list[str]:
    """저장된 아재개그 목록을 반환합니다. 없으면 기본 목록을 반환."""
    if _JOKES_PATH.exists():
        try:
            data = json.loads(_JOKES_PATH.read_text("utf-8"))
            if isinstance(data, list) and len(data) >= 5:
                return data
        except Exception:
            pass
    return _BUILTIN.copy()


def save_jokes(jokes: list[str]) -> None:
    _JOKES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _JOKES_PATH.write_text(
        json.dumps(jokes, ensure_ascii=False, indent=2), "utf-8"
    )


def fetch_and_update_jokes(n: int = 30, api_key: str = "") -> tuple[bool, str]:
    """
    Gemini Google Search로 아재개그 n개를 수집해 기존 목록에 병합·저장합니다.

    Returns:
        (success: bool, message: str)
    """
    if not api_key:
        try:
            config_path = _BASE / "config" / "api_keys.json"
            api_key = json.loads(config_path.read_text("utf-8"))["gemini_api_key"]
        except Exception as e:
            return False, f"API 키 로드 실패: {e}"

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        prompt = (
            f"한국 인터넷에서 유명한 아재개그(dad joke/pun)를 {n}개 수집해주세요. "
            "구글 검색을 사용해 실제 유명한 한국 아재개그를 찾아주세요. "
            "반드시 JSON 배열 형식으로만 반환하세요. 코드블록(```) 없이 순수 JSON만. "
            '형식: ["개그1", "개그2", ...] '
            "각 항목은 '질문? 답변!' 형식의 완결된 하나의 문자열. "
            "가족 친화적이고 재미있는 말장난만. 중복 없음."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"tools": [{"google_search": {}}]},
        )

        raw = "".join(
            part.text
            for part in response.candidates[0].content.parts
            if hasattr(part, "text") and part.text
        ).strip()

        # JSON 배열 추출 (코드블록이 있어도 처리)
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if m:
            raw = m.group(0)

        new_jokes: list[str] = json.loads(raw)
        if not isinstance(new_jokes, list) or len(new_jokes) < 3:
            raise ValueError(f"결과 부족: {raw[:200]}")

        # 기존 목록과 병합 (순서 유지, 중복 제거)
        existing = load_jokes()
        seen     = set(existing)
        combined = existing + [j for j in new_jokes if j not in seen]
        save_jokes(combined)

        added = len(combined) - len(existing)
        return True, (
            f"아재개그 업데이트 완료! "
            f"신규 {added}개 추가 → 총 {len(combined)}개 보유 중입니다."
        )

    except Exception as e:
        print(f"[DadJokes] ❌ 웹 업데이트 실패: {e}")
        return False, f"아재개그 업데이트 실패: {e}"
