"""
writing_assistant.py — 학습 논문 문체 기반 작문 도우미

학습된 논문들의 실제 문장 패턴을 참조하여
AI 투의 작위적 표현 없이 학술 한국어 문장을 작성합니다.
"""

from __future__ import annotations
import re
from typing import Optional

from .style_learner import get_style_examples, get_anti_patterns, get_corpus_stats

# ─── 섹션 유형 ────────────────────────────────────────────────────────────────

_SECTION_GUIDE: dict[str, str] = {
    "intro": (
        "서론 단락입니다. 연구 배경과 문제의식을 자연스럽게 제시하고, "
        "기존 연구의 흐름 속에 이 주제가 어떻게 위치하는지를 보여주세요. "
        "'본 논문은 ~를 고찰한다' 같은 공식적 선언 없이 자연스럽게 시작하세요."
    ),
    "body": (
        "본문 단락입니다. 주어진 내용을 바탕으로 논리적 흐름을 갖춘 산문으로 서술하세요. "
        "각 문장이 앞 문장과 자연스럽게 이어지도록 접속사와 지시어를 적절히 사용하세요."
    ),
    "argument": (
        "논증 단락입니다. 하나의 핵심 주장을 제시하고, 근거를 들어 뒷받침하는 구조로 작성하세요. "
        "주장 → 근거 → 함의의 흐름을 갖추되 지나치게 도식적이지 않게 서술하세요."
    ),
    "conclusion": (
        "결론 단락입니다. 앞서 논의한 내용을 종합하되 단순 반복이 아닌 의미 있는 마무리를 제시하세요. "
        "연구사적 의미나 앞으로의 과제를 자연스럽게 짚어주세요."
    ),
    "analysis": (
        "사료/자료 분석 단락입니다. 자료의 내용을 정확하게 서술하고, "
        "그 역사적 맥락과 의미를 해석하세요. 원문을 인용할 때는 절제 있게 사용하세요."
    ),
    "refine": (
        "주어진 문장/단락을 다듬어 주세요. 의미는 그대로 유지하되 "
        "표현을 더 자연스럽고 학술적으로 개선하세요."
    ),
}

_LENGTH_GUIDE: dict[str, str] = {
    "short":  "2~3문장 (100~180자)",
    "medium": "1개 단락 (250~400자, 5~7문장)",
    "long":   "2~3개 단락 (600~900자)",
}


# ─── 공개 API ─────────────────────────────────────────────────────────────────

def write_research_draft(
    topic: str,
    content: str = "",
    section_type: str = "body",
    length: str = "medium",
    player=None,
) -> str:
    """
    학습된 논문들의 문체를 참조하여 학술 한국어 산문을 작성합니다.

    Args:
        topic:        작성 주제 또는 문장을 다듬을 때는 원문
        content:      포함할 아이디어, 키워드, 핵심 내용 (선택)
        section_type: intro / body / argument / conclusion / analysis / refine
        length:       short / medium / long
        player:       음성 출력 플레이어 (선택)
    """
    stats = get_corpus_stats()
    style_examples = get_style_examples(n=8)
    anti_patterns  = get_anti_patterns()

    if not style_examples:
        # 코퍼스가 비어 있으면 논문 분석을 먼저 안내
        return (
            "⚠️ 아직 학습된 논문이 없습니다.\n"
            "먼저 논문이나 단행본 파일을 분석해서 문체를 학습시켜 주세요.\n"
            "(예: '이 논문 분석해줘' → 파일 경로 제공)"
        )

    prompt = _build_writing_prompt(
        topic=topic,
        content=content,
        section_type=section_type,
        length=length,
        style_examples=style_examples,
        anti_patterns=anti_patterns,
    )

    result = _call_claude_for_writing(prompt)

    # 코퍼스 통계 첨부
    corpus_note = (
        f"\n\n---\n"
        f"📚 참조 코퍼스: {stats['paper_count']}편 논문 / "
        f"{stats['total_sentences']}개 문장"
    )
    return result + corpus_note


def write_with_style_check(draft: str) -> str:
    """
    작성된 초고를 검토하여 AI 투 표현과 개선점을 알려줍니다.
    """
    anti = get_anti_patterns()
    found = [p for p in anti if p in draft]

    if not found:
        feedback = "✅ AI 투 상투 표현이 발견되지 않았습니다."
    else:
        feedback = "⚠️ 아래 표현들은 AI 투로 느껴질 수 있습니다:\n"
        feedback += "\n".join(f"  • {p}" for p in found)

    style_examples = get_style_examples(n=5)
    if style_examples:
        feedback += "\n\n📝 참고 문장 스타일 예시:\n"
        feedback += "\n".join(f"  > {s}" for s in style_examples[:3])

    return feedback


# ─── 내부 함수 ────────────────────────────────────────────────────────────────

def _build_writing_prompt(
    topic: str,
    content: str,
    section_type: str,
    length: str,
    style_examples: list[str],
    anti_patterns: list[str],
) -> str:
    section_guide  = _SECTION_GUIDE.get(section_type, _SECTION_GUIDE["body"])
    length_guide   = _LENGTH_GUIDE.get(length, _LENGTH_GUIDE["medium"])
    examples_block = "\n".join(f"  • {s}" for s in style_examples)
    anti_block     = "\n".join(f"  ✗ {p}" for p in anti_patterns[:10])

    content_block = ""
    if content:
        content_block = f"\n[포함할 내용/아이디어]\n{content}\n"

    prompt = f"""당신은 한국 역사학 학술 논문을 전문적으로 작성하는 편집자입니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[스타일 학습 예시 — 실제 논문 원문 문장]
아래는 사용자가 분석한 논문들에서 추출한 실제 문장들입니다.
이 문장들의 어조, 호흡, 구조를 참고하여 작성하세요.
(내용을 베끼는 것이 아니라 문체와 패턴을 참조하는 것입니다)

{examples_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[절대 사용 금지 — AI 투 상투 표현]
아래 표현들과 이와 유사한 공식적·작위적 표현은 절대 사용하지 마세요.

{anti_block}

추가 금지 사항:
  ✗ "첫째... 둘째... 셋째..." 식의 기계적 나열
  ✗ 모든 문장을 "(이)라고 할 수 있다"로 마무리
  ✗ "또한"을 연속 두 번 이상 사용
  ✗ 짧은 명사형으로 끝나는 불완전 문장 나열
  ✗ 영어 혼용 없이 순한국어로 작성

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[작성 지침]
• {section_guide}
• 분량: {length_guide}
• 문체: 학술 논문 산문체 (해라체: ~하였다, ~였다, ~이다)
• 문장 간 자연스러운 흐름 유지
• 역사학 논문답게 구체적인 시기·인물·사건을 언급 가능
• 불확실한 내용은 단정 짓지 말고 "~로 보인다", "~로 추정된다"로 처리
{content_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[작성 주제]
{topic}

작성 결과만 출력하세요. 설명이나 메타 주석 없이 본문만 작성하세요."""

    return prompt


def _call_claude_for_writing(prompt: str) -> str:
    """Claude API를 호출하여 작문 결과를 반환합니다."""
    try:
        import anthropic
        from pathlib import Path
        import json

        config_path = Path(__file__).parent.parent / "config" / "api_keys.json"
        if not config_path.exists():
            return "❌ Claude API 키가 설정되지 않았습니다. config/api_keys.json을 확인해 주세요."

        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        api_key = cfg.get("claude") or cfg.get("anthropic")
        if not api_key:
            return "❌ Claude API 키가 없습니다."

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-opus-4-8",     # 창의적 작문은 Opus 사용
            max_tokens=2048,
            temperature=0.7,             # 다양성 확보
            system=(
                "당신은 한국 역사학 학술 논문 작성을 전문으로 하는 편집자입니다. "
                "실제 역사학자들의 문장 스타일을 정확히 모방하며, "
                "AI가 만들어내는 상투적 표현을 절대 사용하지 않습니다."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    except ImportError:
        return "❌ anthropic 패키지가 설치되지 않았습니다. pip install anthropic"
    except Exception as e:
        return f"❌ 작문 생성 실패: {e}"


def _is_claude_available() -> bool:
    try:
        from pathlib import Path
        import json
        cfg_path = Path(__file__).parent.parent / "config" / "api_keys.json"
        if not cfg_path.exists():
            return False
        cfg = json.loads(cfg_path.read_text())
        return bool(cfg.get("claude") or cfg.get("anthropic"))
    except Exception:
        return False
