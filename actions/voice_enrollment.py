"""
voice_enrollment.py — 화자 등록 및 인증 모듈

resemblyzer VoiceEncoder(d-vector)를 사용해 사용자 목소리 프로필을 저장하고,
웨이크워드 감지 시 화자를 인증합니다.

등록: enroll_voice()   — 마이크로 N개 샘플 녹음 → 평균 임베딩 → config/voice_profile.npy
로드: load_voice_profile() — 저장된 프로필 반환 (없으면 None)
검증: verify_speaker()  — PCM bytes → 임베딩 → 코사인 유사도 ≥ threshold
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

import numpy as np

PROFILE_PATH       = Path(__file__).resolve().parent.parent / "config" / "voice_profile.npy"
SIMILARITY_THRESHOLD = 0.80
ENROLL_SAMPLES     = 5
SAMPLE_RATE        = 16000
RECORD_SECONDS     = 3


# ── 지연 로딩 헬퍼 ─────────────────────────────────────────────────────────────

def _get_encoder():
    """resemblyzer VoiceEncoder 지연 로딩 — GPU 없이 CPU에서 실행"""
    try:
        from resemblyzer import VoiceEncoder
        return VoiceEncoder()
    except ImportError:
        raise RuntimeError(
            "resemblyzer 미설치 — pip install resemblyzer 실행하세요."
        )


# ── 공개 API ───────────────────────────────────────────────────────────────────

def enroll_voice(
    n_samples: int = ENROLL_SAMPLES,
    player=None,
    speak_fn=None,
) -> str:
    """
    사용자 목소리를 n_samples개 녹음하여 프로필(d-vector 평균)을 저장합니다.

    Args:
        n_samples: 녹음할 샘플 수 (기본 5, 권장 3~10)
        player:    UI 객체 (write_log 메서드 지원 시 로그 출력)
        speak_fn:  음성 출력 함수 (Gemini speak 등)

    Returns:
        결과 메시지 문자열
    """
    try:
        import sounddevice as sd
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError as e:
        return f"필수 패키지 미설치: {e} — pip install resemblyzer sounddevice"

    def _log(msg: str):
        print(f"[VoiceEnroll] {msg}")
        if player and hasattr(player, "write_log"):
            player.write_log(f"SYS: {msg}")

    def _speak(msg: str):
        if speak_fn:
            speak_fn(msg)

    encoder = VoiceEncoder()
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    embeddings: list[np.ndarray] = []
    n_samples = max(3, min(10, int(n_samples)))

    _log(f"목소리 등록 시작 — {n_samples}개 샘플을 녹음합니다.")
    _speak(f"목소리 등록을 시작합니다. {n_samples}번 말씀해 주세요. 준비되면 시작합니다.")
    time.sleep(1.0)

    for i in range(n_samples):
        _speak(f"{i + 1}번. 지금 말씀하세요.")
        time.sleep(0.4)
        _log(f"샘플 {i + 1}/{n_samples} — {RECORD_SECONDS}초 녹음 중...")

        recording = sd.rec(
            int(RECORD_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        sd.wait()
        time.sleep(0.3)

        wav = recording.flatten().astype(np.float32) / 32768.0
        try:
            wav_proc = preprocess_wav(wav, source_sr=SAMPLE_RATE)
            emb = encoder.embed_utterance(wav_proc)
            embeddings.append(emb)
            _log(f"샘플 {i + 1} 완료.")
        except Exception as e:
            _log(f"샘플 {i + 1} 처리 실패: {e}")

    if not embeddings:
        return "목소리 샘플 처리에 모두 실패했습니다. 마이크 연결을 확인하세요."

    mean_emb = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 1e-8:
        mean_emb /= norm

    np.save(str(PROFILE_PATH), mean_emb)
    _log(f"✅ 프로필 저장 완료: {PROFILE_PATH} ({len(embeddings)}/{n_samples} 샘플 사용)")

    msg = (
        f"목소리 등록이 완료되었습니다. "
        f"{len(embeddings)}개 샘플로 프로필을 저장했습니다. "
        f"이제 원준씨 목소리에만 웨이크워드가 반응합니다."
    )
    _speak(msg)
    return msg


def load_voice_profile() -> Optional[np.ndarray]:
    """저장된 목소리 프로필을 로드합니다. 없으면 None 반환."""
    if not PROFILE_PATH.exists():
        return None
    try:
        profile = np.load(str(PROFILE_PATH))
        norm = np.linalg.norm(profile)
        if norm > 1e-8:
            profile = profile / norm
        return profile
    except Exception as e:
        print(f"[VoiceEnroll] 프로필 로드 실패: {e}")
        return None


def verify_speaker(
    audio_bytes: bytes,
    sample_rate: int,
    threshold: float = SIMILARITY_THRESHOLD,
    encoder=None,
) -> tuple[bool, float]:
    """
    PCM bytes(int16)를 저장된 프로필과 비교하여 화자를 인증합니다.

    Returns:
        (인증 통과 여부, 코사인 유사도)
        프로필이 없으면 (True, 1.0) — 등록 전에는 모든 목소리 통과
    """
    profile = load_voice_profile()
    if profile is None:
        return True, 1.0

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        if encoder is None:
            encoder = VoiceEncoder()

        wav = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        wav_proc = preprocess_wav(wav, source_sr=sample_rate)
        emb = encoder.embed_utterance(wav_proc)
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb /= norm

        similarity = float(np.dot(profile, emb))
        return similarity >= threshold, similarity

    except Exception as e:
        print(f"[VoiceEnroll] 화자 검증 오류: {e}")
        return True, 0.0  # 오류 시 통과 (안전 폴백)
