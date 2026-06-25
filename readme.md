# 🤖 W.O.N.J.U.N.S (V.1)
### The Ultimate Cross-Platform Personal AI Assistant — By WONJUN HEO



A real-time voice AI that can hear, see, understand, and control your computer — on any OS. Supporting Windows, macOS, and Linux. Built with Gemini integration for maximum stability and performance, delivering zero subscriptions and total digital autonomy.

---

## ✨ Overview

WONJUNS V.1 represents a massive milestone in the Jarvis series, evolving into a fully connected, highly persistent, and remote-accessible system. It completely bridges the gap between your mobile device, desktop OS, and human intent. Through real-time Gemini reasoning, Mark 46 allows you to control your PC from your phone, share large files securely, and maintain deep contextual conversations across sessions.

It's not just an assistant — it's an extension of your digital life.

---

## 🚀 Capabilities

### Core Features
| Feature | Description |
|---|---|
| 🎙️ Real-time Voice | Ultra-low latency conversation in any language |
| 🖥️ System Control | Launch apps, manage files, execute terminal commands |
| 🧩 Autonomous Tasks | High-level planning for complex, multi-step goals |
| 👁️ Visual Awareness | Real-time screen processing and webcam vision |
| 🧠 Persistent Memory | Deeply remembers your projects, preferences, and personal context |
| ⌨️ Hybrid Input | Seamlessly switch between keyboard typing and voice commands |

---

## 🆕 What's New in XLVI

- 📱 **Full Remote Phone Control** — Take command of your entire desktop operating system directly from your smartphone, anywhere, anytime.
- 🧠 **Advanced Long-Term Memory** — Upgraded memory architecture allows Jarvis to contextually remember past interactions, preferences, and complex workflows across reboots.
- 🚀 **Powered by Gemini Integration** — Re-engineered from the ground up to utilize the full speed and precision of the Google Gemini API for ultimate reasoning and stability.
- ⚡ **Next-Gen Performance & Stability** — Comprehensive system-wide optimizations delivering faster response times and rock-solid execution on Windows, Mac, and Linux.
- 📂 **Advanced File Handling & Hybrid Input** — Fluidly switch between voice or keyboard input, and drag-and-drop code, PDFs, or images for instant analysis and automation.
- 🔒 **Secure Mobile File Sharing** — Wirelessly and securely share files or entire folders up to 500 MB from your phone directly to your computer with complete privacy.

---

---

## 🏛️ 역사학 연구 특화 기능 (History Research Module)

이 버전은 역사학 연구자를 위한 전문 기능이 추가되었습니다.

### 📂 지원 파일 형식
| 형식 | 처리 방법 | 각주 정확도 |
|---|---|---|
| PDF | pdfplumber (페이지별 추출) | ⭐⭐⭐ 높음 |
| HWP/한글 | hwp5txt 또는 LibreOffice 변환 | ⭐⭐ 중간 (페이지 추정) |
| DOCX/Word | python-docx | ⭐⭐ 중간 (페이지 추정) |
| JPG/PNG | EasyOCR (한국어·영어) | ⭐⭐ 중간 (OCR 품질 의존) |

### 🔬 핵심 기능
| 기능 | 음성 명령 예시 |
|---|---|
| 논문 분석 | "이 PDF 논문 분석해줘", "방금 올린 파일 읽어줘" |
| 각주 생성 | "23페이지에서 식민지 관련 각주 만들어줘" |
| 연구사 공백 | "지금까지 정리된 연구에서 공백 찾아줘" |
| 옵시디안 연동 | "옵시디안 볼트 경로 설정해줘" |

### 🗂️ 옵시디안 연동 구조
```
옵시디안볼트/
└── 역사학연구/
    ├── 논문분석/          ← 각 논문 분석 노트
    ├── 개념인덱스/        ← 키워드별 인덱스
    ├── 저자인덱스/        ← 저자별 논문 목록
    ├── 연구사분석.md      ← 연구사 공백 보고서
    └── .knowledge_graph.json  ← 지식 그래프 데이터
```

### ⚙️ 추가 설치 (역사학 모듈)
```bash
pip install pdfplumber python-docx easyocr
# HWP 파일 지원 (선택):
pip install pyhwp
# 또는 LibreOffice 설치 (https://www.libreoffice.org)
```

---

## ⚡ Quick Start

```bash
git clone [https://github.com/FatihMakes/Mark-XLVI.git](https://github.com/FatihMakes/Mark-XLVI.git)
cd Mark-XLVI
pip install -r requirements.txt
playwright install
python main.py

```

> ⚠️ **Installation Note:** To keep the repository lightweight, some OS-specific dependencies are not bundled in `requirements.txt`. If you run into a `ModuleNotFoundError`, simply install the missing package via `pip install <module_name>` for your specific system.

---

## 📋 Requirements

| Requirement | Details |
| --- | --- |
| **OS** | Windows 10/11, macOS, or Linux |
| **Python** | 3.11 or 3.12 |
| **Microphone** | Required for voice interaction |
| **API Key** | Free Gemini API key |

---

## ⚠️ License

Personal and non-commercial use only.
