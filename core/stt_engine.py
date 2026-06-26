"""
stt_engine.py — 한국어 특화 음성-텍스트 변환 엔진

우선순위:
  1. faster-whisper (로컬 오프라인, 한국어 정확도 매우 높음)
  2. SpeechRecognition + Google Speech API (인터넷 필요, 폴백)

faster-whisper 설치:
  pip install faster-whisper
"""

from __future__ import annotations
import io
import math
import struct
from pathlib import Path
from typing import Optional

# ─── 결과 타입 ────────────────────────────────────────────────────────────────

class STTResult:
    def __init__(self, text: str, confidence: float = 1.0, engine: str = ""):
        self.text       = text.strip().lower()
        self.confidence = confidence
        self.engine     = engine

    def __bool__(self) -> bool:
        return bool(self.text)

    def __repr__(self) -> str:
        return f"STTResult({self.text!r}, conf={self.confidence:.2f}, engine={self.engine!r})"


# ─── 캐시 ─────────────────────────────────────────────────────────────────────

_whisper_model      = None
_whisper_available  = None   # None=아직 미확인, True=사용가능, False=불가

_MODEL_CACHE_DIR = Path.home() / ".cache" / "unfat" / "whisper"


def _get_whisper_model():
    """faster-whisper tiny 모델 지연 로딩 (최초 1회만 다운로드)."""
    global _whisper_model, _whisper_available
    if _whisper_available is False:
        return None
    if _whisper_model is not None:
        return _whisper_model
    try:
        from faster_whisper import WhisperModel
        _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # tiny: ~39MB, 한국어 정확도 충분, CPU 추론 ~100–200ms
        # small 모델(~244MB)은 더 정확하지만 추론 시간 ~400ms
        _whisper_model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            download_root=str(_MODEL_CACHE_DIR),
        )
        _whisper_available = True
        print("[STT] ✅ faster-whisper tiny 모델 로드 완료 (한국어 오프라인 인식)")
        return _whisper_model
    except Exception as e:
        _whisper_available = False
        print(f"[STT] ℹ️  faster-whisper 없음 — Google Speech API 사용 (pip install faster-whisper 권장)")
        return None


# ─── 공개 API ─────────────────────────────────────────────────────────────────

def transcribe_korean(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    fallback_to_google: bool = True,
) -> STTResult:
    """
    PCM int16 오디오를 한국어로 전사합니다.

    Args:
        pcm_bytes:          raw PCM int16 bytes (mono, 16kHz 권장)
        sample_rate:        샘플링 레이트
        fallback_to_google: Google Speech API 폴백 허용 여부

    Returns:
        STTResult — bool(result) == False 이면 인식 실패
    """
    if len(pcm_bytes) < sample_rate * 2 * 0.3:  # 0.3초 미만 → 스킵
        return STTResult("", engine="too_short")

    if not _has_speech_energy(pcm_bytes):
        return STTResult("", engine="silence")

    # ① faster-whisper (로컬, 고정밀 한국어)
    model = _get_whisper_model()
    if model is not None:
        result = _transcribe_whisper(model, pcm_bytes, sample_rate)
        if result:
            return result

    # ② Google Speech API 폴백
    if fallback_to_google:
        return _transcribe_google(pcm_bytes, sample_rate)

    return STTResult("", engine="failed")


def is_faster_whisper_available() -> bool:
    """faster-whisper 사용 가능 여부 반환."""
    if _whisper_available is None:
        _get_whisper_model()
    return _whisper_available is True


def preload_whisper_model():
    """앱 시작 시 미리 모델을 로드합니다 (첫 인식 지연 방지)."""
    _get_whisper_model()


# ─── 내부 구현 ────────────────────────────────────────────────────────────────

def _transcribe_whisper(model, pcm_bytes: bytes, sample_rate: int) -> STTResult:
    """faster-whisper로 한국어 전사."""
    try:
        import numpy as np

        audio_arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # resample to 16kHz if needed (faster-whisper expects 16kHz)
        if sample_rate != 16000:
            import scipy.signal
            audio_arr = scipy.signal.resample_poly(audio_arr, 16000, sample_rate)

        segments_gen, _info = model.transcribe(
            audio_arr,
            language="ko",           # 한국어 고정 — 자동 감지보다 훨씬 정확
            beam_size=3,             # 빔 크기: 정확도 vs 속도 균형 (1=빠름, 5=정확)
            best_of=3,
            temperature=0.0,         # 결정적 출력 (낮은 온도 = 할루시네이션 감소)
            vad_filter=True,         # 내장 VAD로 무음 구간 자동 제거
            vad_parameters={
                "min_silence_duration_ms": 250,
                "speech_pad_ms": 150,
                "threshold": 0.3,    # 민감도 낮게 — 잡음에서 오인식 감소
            },
            word_timestamps=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )

        # generator 물질화
        segments = list(segments_gen)

        if not segments:
            return STTResult("", engine="whisper_no_seg")

        text = " ".join(s.text for s in segments).strip()

        # 한국어 최소 품질 필터: 너무 짧거나 영문만이면 의심
        if not text or len(text) < 2:
            return STTResult("", engine="whisper_empty")

        # 평균 log probability → confidence 근사 (−1.0~0 → 0~1)
        avg_lp = sum(s.avg_logprob for s in segments) / len(segments)
        confidence = min(1.0, max(0.0, avg_lp + 1.0))

        # 낮은 신뢰도는 스킵 (기준: −0.7 이상만 채택)
        if avg_lp < -0.7 and len(text) < 5:
            return STTResult("", engine="whisper_low_conf")

        return STTResult(text, confidence=confidence, engine="whisper")

    except Exception as e:
        print(f"[STT] faster-whisper 오류: {e}")
        return STTResult("", engine="whisper_err")


def _transcribe_google(pcm_bytes: bytes, sample_rate: int) -> STTResult:
    """Google Speech API로 한국어 전사 (폴백)."""
    try:
        import speech_recognition as sr
        rec = sr.Recognizer()
        rec.energy_threshold         = 300
        rec.dynamic_energy_threshold = False

        audio = sr.AudioData(pcm_bytes, sample_rate, 2)
        text  = rec.recognize_google(audio, language="ko-KR")
        return STTResult(text, confidence=0.80, engine="google")
    except Exception:
        return STTResult("", engine="google_fail")


def _has_speech_energy(pcm_bytes: bytes, threshold: float = 250.0) -> bool:
    """RMS 에너지 기반 무음 빠른 판별."""
    if len(pcm_bytes) < 2:
        return False
    try:
        count   = len(pcm_bytes) // 2
        samples = struct.unpack(f"{count}h", pcm_bytes[:count * 2])
        rms     = math.sqrt(sum(s * s for s in samples) / count)
        return rms >= threshold
    except Exception:
        return True  # 판별 실패 → 통과
