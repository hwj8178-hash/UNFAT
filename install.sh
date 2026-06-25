#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# UNFAT 스마트 설치 스크립트 — Python 3.11 ~ 3.14 호환
# 사용법: bash install.sh
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

PYTHON=${PYTHON:-python3}
PIP="$PYTHON -m pip"

# ─── 색상 출력 ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✅ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $*${NC}"; }
err()  { echo -e "${RED}❌ $*${NC}"; }

# ─── Python 버전 확인 ─────────────────────────────────────────────────────────
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  UNFAT 설치 스크립트"
echo "  Python: $PY_VER"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    err "Python 3.11 이상이 필요합니다 (현재: $PY_VER)"
    exit 1
fi

# ─── pip 업그레이드 ───────────────────────────────────────────────────────────
echo ""
echo "📦 pip 업그레이드..."
$PIP install --upgrade pip --quiet

# ─── 1단계: 코어 패키지 (항상 성공) ─────────────────────────────────────────
echo ""
echo "━━━━━ 1단계: 코어 패키지 설치 ━━━━━"
$PIP install -r requirements.txt
ok "코어 패키지 설치 완료"

# ─── 2단계: GUI 자동화 패키지 ────────────────────────────────────────────────
echo ""
echo "━━━━━ 2단계: GUI 자동화 패키지 (pyautogui, pygetwindow) ━━━━━"
echo "ℹ️  Python $PY_VER — 바이너리 wheel 없을 경우 소스 빌드 시도"

install_gui_packages() {
    # 바이너리 우선 시도
    if $PIP install pyautogui pygetwindow --quiet 2>/dev/null; then
        ok "pyautogui + pygetwindow 설치 완료 (바이너리)"
        return 0
    fi

    warn "바이너리 설치 실패 — 소스 빌드 시도 중..."

    # Linux: xlib 헤더 안내
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "   Linux 소스 빌드를 위해 필요한 경우:"
        echo "   sudo apt-get install -y python3-xlib xdotool scrot"
    fi

    if $PIP install pyautogui pygetwindow --no-binary=:all: --quiet 2>/dev/null; then
        ok "pyautogui + pygetwindow 소스 빌드 성공"
        return 0
    fi

    warn "pyautogui/pygetwindow 설치 실패"
    echo "   → 손 제스처 컴퓨터 제어, 창 전환 기능이 비활성화됩니다"
    echo "   → 역사학 연구 분석 기능에는 영향 없습니다"
    return 0  # 실패해도 계속
}

install_gui_packages

# ─── 3단계: 화자 인증 (resemblyzer + webrtcvad) ──────────────────────────────
echo ""
echo "━━━━━ 3단계: 화자 인증 패키지 (resemblyzer) ━━━━━"

install_resemblyzer() {
    # webrtcvad 바이너리 시도
    if $PIP install webrtcvad --quiet 2>/dev/null; then
        ok "webrtcvad 설치 완료 (바이너리)"
        $PIP install resemblyzer --quiet 2>/dev/null && ok "resemblyzer 설치 완료" || true
        return 0
    fi

    warn "webrtcvad 바이너리 없음 — 소스 빌드 시도 중..."
    echo "   소스 빌드에는 gcc/clang 컴파일러가 필요합니다"

    # 소스 빌드 시도
    if $PIP install webrtcvad --no-binary=:all: --quiet 2>/dev/null; then
        ok "webrtcvad 소스 빌드 성공"
        $PIP install resemblyzer --quiet 2>/dev/null && ok "resemblyzer 설치 완료" || true
        return 0
    fi

    # webrtcvad 빌드 실패 → 스텁 모드로 resemblyzer만 설치
    warn "webrtcvad 빌드 실패 — 스텁 모드 활성화"
    echo "   resemblyzer를 webrtcvad 없이 설치합니다"
    echo "   (화자 인증 기능은 동작하나 무음 구간 필터링 정확도가 약간 낮아집니다)"

    # resemblyzer --no-deps 설치 후 필수 의존성만 설치
    if $PIP install resemblyzer --no-deps --quiet 2>/dev/null; then
        $PIP install librosa numpy scipy --quiet 2>/dev/null || true
        ok "resemblyzer 스텁 모드 설치 완료"
        echo "   ℹ️  앱 실행 시 webrtcvad 스텁이 자동으로 주입됩니다"
        return 0
    fi

    warn "resemblyzer 설치 실패 — 화자 인증 기능 비활성화"
    echo "   → 웨이크워드는 인식되나 화자 인증 없이 동작합니다"
    return 0
}

install_resemblyzer

# ─── 완료 ────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ok "UNFAT 설치 완료 (Python $PY_VER)"
echo ""
echo "  핵심 기능 확인:"
$PYTHON -c "import anthropic; print('  ✅ Claude AI (논문 분석)')" 2>/dev/null || echo "  ⚠️  anthropic 미설치"
$PYTHON -c "import pdfplumber; print('  ✅ PDF 추출')" 2>/dev/null || echo "  ⚠️  pdfplumber 미설치"
$PYTHON -c "import easyocr; print('  ✅ OCR (한자 인식 포함)')" 2>/dev/null || echo "  ⚠️  easyocr 미설치"
$PYTHON -c "import pyautogui; print('  ✅ GUI 자동화')" 2>/dev/null || echo "  ⚪ pyautogui 미설치 (선택 기능)"
$PYTHON -c "import resemblyzer; print('  ✅ 화자 인증')" 2>/dev/null || echo "  ⚪ resemblyzer 미설치 (선택 기능)"
echo ""
echo "  실행: python main.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
