"""
hand_gesture.py — 손 제스처 기반 컴퓨터 제어 모듈

노트북 카메라 → MediaPipe Hands (21개 랜드마크) → 제스처 분류 → 컴퓨터 동작

────────────────────────────────────────────
 제스처 → 동작 매핑
────────────────────────────────────────────
 ☝️  검지만              → 마우스 포인터 이동
 ✊  주먹 (쥐기)          → 왼쪽 클릭
 ✌️  검지+중지 (V사인)    → 스크롤 (손 높이로 제어)
 👌  OK 사인             → 오른쪽 클릭
 👍  엄지 위              → 볼륨 증가
 👎  엄지 아래            → 볼륨 감소
 ✋  손바닥 (다섯 손가락)  → 미디어 재생/일시정지
 🖐  네 손가락 (엄지 제외) → 화면 캡처
 ←→  손 스와이프          → 브라우저 뒤로/앞으로
────────────────────────────────────────────
필수: pip install mediapipe opencv-python pyautogui
"""

from __future__ import annotations
import sys
import threading
import time
from collections import deque
from enum import Enum
from typing import Callable, Optional

import cv2
import numpy as np


# ─── 제스처 열거형 ──────────────────────────────────────────────────────────

class Gesture(Enum):
    NONE          = "대기"
    FIST          = "주먹_클릭"
    OPEN_HAND     = "손바닥_재생"
    POINTER       = "검지_마우스이동"
    PEACE         = "V사인_스크롤"
    THUMBS_UP     = "엄지위_볼륨업"
    THUMBS_DOWN   = "엄지아래_볼륨다운"
    OK            = "OK_우클릭"
    FOUR_FINGERS  = "네손가락_캡처"
    SWIPE_LEFT    = "스와이프왼쪽_뒤로"
    SWIPE_RIGHT   = "스와이프오른쪽_앞으로"
    SCROLL_DOWN   = "세손가락_스크롤다운"   # 검지+중지+약지 → 아래 스크롤
    ZOOM          = "핀치_확대축소"          # 엄지+검지 간격 → 화면 줌


# 화면 표시용 라벨
GESTURE_LABEL = {
    Gesture.NONE:         "○ 대기",
    Gesture.FIST:         "✊ 클릭",
    Gesture.OPEN_HAND:    "✋ 재생/정지",
    Gesture.POINTER:      "☝ 마우스 이동",
    Gesture.PEACE:        "✌ 스크롤 (상하)",
    Gesture.THUMBS_UP:    "👍 볼륨 +",
    Gesture.THUMBS_DOWN:  "👎 볼륨 -",
    Gesture.OK:           "👌 우클릭",
    Gesture.FOUR_FINGERS: "🖐 화면 캡처",
    Gesture.SWIPE_LEFT:   "← 뒤로",
    Gesture.SWIPE_RIGHT:  "→ 앞으로",
    Gesture.SCROLL_DOWN:  "↓↓↓ 스크롤 다운",
    Gesture.ZOOM:         "🔍 확대/축소",
}

# 제스처별 최소 실행 간격 (ms) — 너무 빠른 반복 방지
DEBOUNCE_MS: dict[Gesture, int] = {
    Gesture.FIST:         700,
    Gesture.OPEN_HAND:   1300,
    Gesture.OK:           900,
    Gesture.THUMBS_UP:    350,
    Gesture.THUMBS_DOWN:  350,
    Gesture.FOUR_FINGERS: 2500,
    Gesture.SWIPE_LEFT:   900,
    Gesture.SWIPE_RIGHT:  900,
    Gesture.SCROLL_DOWN:  80,   # 연속 스크롤 — 80ms 간격
}

# 손가락 랜드마크 인덱스 (MediaPipe)
# TIP: 4,8,12,16,20 / PIP(중간 관절): 3,6,10,14,18
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]


# ─── 핵심 컨트롤러 클래스 ─────────────────────────────────────────────────────

class HandGestureController:
    """
    별도 스레드에서 실행되는 손 제스처 인식 컨트롤러.
    카메라를 열고 MediaPipe로 손을 추적하여 제스처를 컴퓨터 동작에 매핑합니다.
    """

    def __init__(
        self,
        camera_index: int = 0,
        speak_fn: Optional[Callable[[str], None]] = None,
        player=None,
        mouse_smoothing: float = 0.20,
        show_preview: bool = True,
    ):
        self.camera_index    = camera_index
        self.speak_fn        = speak_fn
        self.player          = player
        self.mouse_smoothing = mouse_smoothing  # 낮을수록 부드럽고 느림
        self.show_preview    = show_preview

        self.running         = False
        self._thread: Optional[threading.Thread] = None
        self._cap            = None

        # 마우스 스무딩 상태
        self._mx: Optional[float] = None
        self._my: Optional[float] = None

        # 스크롤 기준 y좌표
        self._scroll_ref: Optional[float] = None

        # 줌 기준 거리 (엄지-검지 간격)
        self._zoom_ref: Optional[float] = None

        # 제스처 디바운싱 — {Gesture: last_triggered_ms}
        self._last_ts: dict[Gesture, float] = {}

        # 스와이프 감지 — (timestamp, wrist_x) 히스토리
        self._swipe_history: deque = deque(maxlen=15)
        self._last_swipe_ts  = 0.0

        # 현재 감지 정보 (상태 조회용)
        self.current_gesture = Gesture.NONE
        self.current_fingers = [False] * 5

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def start(self) -> str:
        if self.running:
            return "손 제스처 인식이 이미 실행 중입니다."
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="GestureLoop")
        self._thread.start()
        msg = (
            "손 제스처 인식을 시작합니다. "
            "카메라 창에서 Q 키를 누르면 종료됩니다. "
            "☝ 검지로 마우스를 이동하고, ✊ 주먹으로 클릭하세요."
        )
        self._say(msg)
        return msg

    def stop(self) -> str:
        self.running = False
        cv2.destroyAllWindows()
        msg = "손 제스처 인식을 종료합니다."
        self._say(msg)
        return msg

    def status(self) -> str:
        if not self.running:
            return "비활성화 — '손 제스처 시작'으로 실행하세요."
        label = GESTURE_LABEL.get(self.current_gesture, "—")
        fingers = "".join("●" if f else "○" for f in self.current_fingers)
        return f"✅ 실행 중  |  제스처: {label}  |  손가락: [{fingers}]"

    def set_smoothing(self, value: float):
        self.mouse_smoothing = max(0.05, min(1.0, value))

    # ── 메인 루프 ─────────────────────────────────────────────────────────────

    def _loop(self):
        try:
            import mediapipe as mp
        except ImportError:
            self._say("mediapipe 패키지가 없습니다. pip install mediapipe 를 실행하세요.")
            self.running = False
            return

        mp_hands   = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_styles  = mp.solutions.drawing_styles

        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            self._say("카메라를 열 수 없습니다. 카메라가 연결되어 있는지 확인하세요.")
            self.running = False
            return

        sw, sh = self._screen_size()

        with mp_hands.Hands(
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.72,
            min_tracking_confidence=0.65,
        ) as hands:

            while self.running:
                ok, frame = self._cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue

                frame = cv2.flip(frame, 1)        # 거울 모드 (자연스러운 제어)
                h, w  = frame.shape[:2]
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = hands.process(rgb)
                rgb.flags.writeable = True

                gesture = Gesture.NONE
                fingers = [False] * 5

                if res.multi_hand_landmarks and res.multi_handedness:
                    lm_list  = res.multi_hand_landmarks[0].landmark
                    handedness = res.multi_handedness[0].classification[0].label

                    fingers  = self._fingers_state(lm_list, handedness)
                    gesture  = self._classify(fingers, lm_list)

                    # 스와이프는 별도 감지 (어떤 제스처와도 병행 가능)
                    swipe = self._detect_swipe(lm_list[0].x)
                    if swipe:
                        gesture = swipe

                    self.current_gesture = gesture
                    self.current_fingers = fingers

                    # 컴퓨터 동작 실행
                    self._execute(gesture, lm_list, sw, sh)

                    # 랜드마크 시각화
                    if self.show_preview:
                        mp_drawing.draw_landmarks(
                            frame,
                            res.multi_hand_landmarks[0],
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connection_style(),
                        )
                else:
                    # 손이 인식되지 않으면 상태 초기화
                    self.current_gesture = Gesture.NONE
                    self.current_fingers = [False] * 5
                    self._scroll_ref = None
                    self._zoom_ref   = None
                    self._mx = self._my = None

                if self.show_preview:
                    self._draw_hud(frame, gesture, fingers)
                    cv2.imshow("Gesture Control  [Q: 종료]", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        self.running = False
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()

    # ── 손가락 상태 감지 ──────────────────────────────────────────────────────

    @staticmethod
    def _fingers_state(lm, hand_label: str) -> list[bool]:
        """
        각 손가락이 펴져 있는지 감지합니다.
        반환: [엄지, 검지, 중지, 약지, 소지]
        """
        up = []

        # 엄지: 좌우 방향으로 판단 (카메라 반전 + 손 방향 고려)
        # 미러 이미지이므로 Left/Right가 뒤바뀜
        if hand_label == "Left":   # 카메라 기준 왼손 = 실제 오른손
            up.append(lm[FINGER_TIPS[0]].x < lm[FINGER_PIPS[0]].x)
        else:
            up.append(lm[FINGER_TIPS[0]].x > lm[FINGER_PIPS[0]].x)

        # 나머지 4 손가락: tip.y < pip.y → 위로 펴짐
        for i in range(1, 5):
            up.append(lm[FINGER_TIPS[i]].y < lm[FINGER_PIPS[i]].y)

        return up

    # ── 제스처 분류 ──────────────────────────────────────────────────────────

    @staticmethod
    def _classify(f: list[bool], lm) -> Gesture:
        """
        손가락 상태(f)와 랜드마크(lm)로 제스처를 결정합니다.
        f = [엄지, 검지, 중지, 약지, 소지]
        """
        count = sum(f)

        # 주먹 (0개)
        if count == 0:
            return Gesture.FIST

        # 손바닥 (5개)
        if count == 5:
            return Gesture.OPEN_HAND

        # 검지만 → 마우스 이동
        if f == [False, True, False, False, False]:
            return Gesture.POINTER

        # 검지 + 중지 → 스크롤
        if f == [False, True, True, False, False]:
            return Gesture.PEACE

        # 엄지만 → 위아래 판별
        if f == [True, False, False, False, False]:
            # 엄지 끝이 손목보다 충분히 위에 있으면 엄지 위
            if lm[4].y < lm[0].y - 0.04:
                return Gesture.THUMBS_UP
            return Gesture.THUMBS_DOWN

        # 네 손가락 (엄지 제외) → 캡처
        if f == [False, True, True, True, True]:
            return Gesture.FOUR_FINGERS

        # 세 손가락 (검지+중지+약지, 소지·엄지 접음) → 스크롤 다운
        if f == [False, True, True, True, False]:
            return Gesture.SCROLL_DOWN

        # OK 사인: 엄지+검지 끝이 가깝고 나머지 펴짐
        dist = np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
        if dist < 0.055 and f[2] and f[3] and f[4]:
            return Gesture.OK

        # 핀치/스프레드: 엄지+검지만 펴고 나머지 접음 → 줌
        if f[0] and f[1] and not f[2] and not f[3] and not f[4]:
            return Gesture.ZOOM

        return Gesture.NONE

    # ── 스와이프 감지 ─────────────────────────────────────────────────────────

    def _detect_swipe(self, wrist_x: float) -> Optional[Gesture]:
        now = time.monotonic()
        self._swipe_history.append((now, wrist_x))

        # 최근 0.45초 데이터만 분석
        recent = [(t, x) for t, x in self._swipe_history if now - t <= 0.45]
        if len(recent) < 8:
            return None

        dx = recent[-1][1] - recent[0][1]   # 정규화 거리 (0~1 범위)
        if abs(dx) > 0.22 and now - self._last_swipe_ts > 0.8:
            self._last_swipe_ts = now
            self._swipe_history.clear()
            return Gesture.SWIPE_RIGHT if dx > 0 else Gesture.SWIPE_LEFT

        return None

    # ── 동작 실행 ────────────────────────────────────────────────────────────

    def _execute(self, gesture: Gesture, lm, sw: int, sh: int):
        try:
            import pyautogui
        except ImportError:
            return

        now_ms = time.monotonic() * 1000

        # ── 연속 동작 (매 프레임 실행) ───────────────────────────────────────

        if gesture == Gesture.POINTER:
            # 검지 끝(lm[8])을 화면 좌표에 매핑, 지수 이동 평균 스무딩
            tx = lm[8].x * sw
            ty = lm[8].y * sh
            if self._mx is None:
                self._mx, self._my = tx, ty
            s = self.mouse_smoothing
            self._mx = self._mx * (1 - s) + tx * s
            self._my = self._my * (1 - s) + ty * s
            pyautogui.moveTo(int(self._mx), int(self._my), _pause=False)
            return

        if gesture == Gesture.PEACE:
            # 검지 중간 관절(lm[6]) 높이 변화로 스크롤량 결정
            cy = lm[6].y
            if self._scroll_ref is None:
                self._scroll_ref = cy
                return
            delta = self._scroll_ref - cy   # 양수 = 위로 올림 = 위로 스크롤
            ticks = int(delta * 35)
            if abs(ticks) >= 1:
                pyautogui.scroll(ticks, _pause=False)
            self._scroll_ref = cy
            return

        if gesture == Gesture.THUMBS_UP:
            if now_ms - self._last_ts.get(gesture, 0) > DEBOUNCE_MS[gesture]:
                self._last_ts[gesture] = now_ms
                self._volume(+2)
            return

        if gesture == Gesture.THUMBS_DOWN:
            if now_ms - self._last_ts.get(gesture, 0) > DEBOUNCE_MS[gesture]:
                self._last_ts[gesture] = now_ms
                self._volume(-2)
            return

        # 세 손가락 → 스크롤 다운 (연속, 80ms 간격)
        if gesture == Gesture.SCROLL_DOWN:
            if now_ms - self._last_ts.get(gesture, 0) > DEBOUNCE_MS[gesture]:
                self._last_ts[gesture] = now_ms
                pyautogui.scroll(-3, _pause=False)
            return

        # 핀치/스프레드 → 줌 인/아웃 (Ctrl + 마우스 휠)
        if gesture == Gesture.ZOOM:
            dist = float(np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y))
            if self._zoom_ref is None:
                self._zoom_ref = dist
                return
            delta = dist - self._zoom_ref          # 양수=벌림=줌인, 음수=좁힘=줌아웃
            ticks = int(delta * 40)                # 감도 계수
            if abs(ticks) >= 1:
                try:
                    import pyautogui as _pag
                    _pag.keyDown('ctrl')
                    _pag.scroll(ticks, _pause=False)
                    _pag.keyUp('ctrl')
                except Exception:
                    pass
            self._zoom_ref = dist
            return

        # ── 단발 동작 (디바운싱 적용) ────────────────────────────────────────

        cooldown = DEBOUNCE_MS.get(gesture, 800)
        if now_ms - self._last_ts.get(gesture, 0) < cooldown:
            return
        self._last_ts[gesture] = now_ms

        if gesture == Gesture.FIST:
            pyautogui.click(_pause=False)

        elif gesture == Gesture.OK:
            pyautogui.rightClick(_pause=False)

        elif gesture == Gesture.OPEN_HAND:
            pyautogui.press('playpause')

        elif gesture == Gesture.FOUR_FINGERS:
            if sys.platform == "win32":
                pyautogui.hotkey('win', 'shift', 's')
            elif sys.platform == "darwin":
                pyautogui.hotkey('command', 'shift', '3')
            else:
                pyautogui.hotkey('ctrl', 'print_screen')

        elif gesture == Gesture.SWIPE_LEFT:
            if sys.platform == "win32":
                pyautogui.hotkey('alt', 'left')
            elif sys.platform == "darwin":
                pyautogui.hotkey('command', '[')
            else:
                pyautogui.hotkey('alt', 'left')

        elif gesture == Gesture.SWIPE_RIGHT:
            if sys.platform == "win32":
                pyautogui.hotkey('alt', 'right')
            elif sys.platform == "darwin":
                pyautogui.hotkey('command', ']')
            else:
                pyautogui.hotkey('alt', 'right')

    # ── 볼륨 조절 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _volume(delta: int):
        try:
            import pyautogui
            if sys.platform in ("win32", "darwin"):
                key = 'volumeup' if delta > 0 else 'volumedown'
                for _ in range(abs(delta)):
                    pyautogui.press(key)
            else:
                import subprocess
                sign = '+' if delta > 0 else '-'
                subprocess.run(
                    ['pactl', 'set-sink-volume', '@DEFAULT_SINK@',
                     f'{sign}{abs(delta) * 5}%'],
                    capture_output=True,
                )
        except Exception:
            pass

    # ── HUD 오버레이 그리기 ───────────────────────────────────────────────────

    @staticmethod
    def _draw_hud(frame: np.ndarray, gesture: Gesture, fingers: list[bool]):
        h, w = frame.shape[:2]
        label = GESTURE_LABEL.get(gesture, "—")

        # 반투명 하단 바
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 90), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # 제스처 라벨
        color = (60, 220, 80) if gesture != Gesture.NONE else (120, 120, 120)
        cv2.putText(
            frame, label,
            (12, h - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA,
        )

        # 손가락 상태 표시 (5개 원)
        labels_kr = ["T", "1", "2", "3", "4"]
        for i, (up, lbl) in enumerate(zip(fingers, labels_kr)):
            cx = 12 + i * 42
            cy = h - 24
            fill_color = (50, 220, 50) if up else (60, 60, 60)
            cv2.circle(frame, (cx + 14, cy), 15, fill_color, -1)
            cv2.putText(
                frame, lbl, (cx + 9, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        # 상태 점
        dot_color = (50, 220, 50) if gesture != Gesture.NONE else (80, 80, 80)
        cv2.circle(frame, (w - 20, h - 70), 9, dot_color, -1)

        # 도움말 (오른쪽 상단)
        hints = [
            "☝  =  Mouse",
            "✊  =  Click",
            "✌  =  Scroll",
            "3fingers = ScrollDn",
            "T+1 = Zoom",
            "👌  =  R-Click",
            "👍  =  Vol+",
            "👎  =  Vol-",
        ]
        for i, hint in enumerate(hints):
            cv2.putText(
                frame, hint,
                (w - 140, 22 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
            )

    # ── 유틸리티 ─────────────────────────────────────────────────────────────

    def _say(self, text: str):
        if self.speak_fn:
            try:
                self.speak_fn(text)
            except Exception:
                pass

    @staticmethod
    def _screen_size() -> tuple[int, int]:
        try:
            import pyautogui
            return pyautogui.size()
        except Exception:
            return 1920, 1080


# ─── 전역 인스턴스 및 진입점 ─────────────────────────────────────────────────

_ctrl: Optional[HandGestureController] = None


def hand_gesture_control(
    command: str,
    parameters: dict,
    player=None,
    speak_fn: Optional[Callable] = None,
) -> str:
    """
    main.py 도구 핸들러 진입점.

    command:
      start     - 인식 시작 (카메라 창 열림)
      stop      - 인식 종료
      status    - 현재 상태 조회
      smoothing - 마우스 스무딩 조정 (0.05 ~ 1.0)
    """
    global _ctrl

    if command == "start":
        if _ctrl and _ctrl.running:
            return "손 제스처 인식이 이미 실행 중입니다."
        _ctrl = HandGestureController(
            camera_index=int(parameters.get("camera_index", 0)),
            speak_fn=speak_fn,
            player=player,
            show_preview=bool(parameters.get("show_preview", True)),
        )
        return _ctrl.start()

    if command == "stop":
        if not _ctrl or not _ctrl.running:
            return "손 제스처 인식이 실행 중이 아닙니다."
        return _ctrl.stop()

    if command == "status":
        if not _ctrl:
            return "손 제스처 인식 비활성화 상태입니다."
        return _ctrl.status()

    if command == "smoothing":
        if not _ctrl:
            return "먼저 시작해주세요."
        v = float(parameters.get("value", 0.2))
        _ctrl.set_smoothing(v)
        return f"마우스 스무딩이 {v:.2f}로 설정되었습니다."

    return f"알 수 없는 명령: {command}"
