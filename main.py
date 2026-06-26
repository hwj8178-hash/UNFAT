import asyncio
import queue as _queue
import re
import threading
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import sounddevice as sd
from google import genai
from google.genai import types
from ui import AssistantUI
from memory.memory_manager import (
    load_memory, update_memory, format_memory_for_prompt,
)

from actions.file_processor import file_processor
from actions.flight_finder     import flight_finder
from actions.open_app          import open_app
from actions.weather_report    import weather_action
from actions.send_message      import send_message
from actions.reminder          import reminder
from actions.computer_settings import computer_settings
from actions.screen_processor  import screen_process
from actions.youtube_video     import youtube_video
from actions.desktop           import desktop_control
from actions.browser_control   import browser_control
from actions.file_controller   import file_controller
from actions.code_helper       import code_helper
from actions.dev_agent         import dev_agent
from actions.web_search        import web_search as web_search_action
from actions.computer_control  import computer_control
from actions.game_updater      import game_updater
from actions.history_researcher import history_research_action
from actions.obsidian_bridge   import set_vault_path, analyze_research_landscape, get_vault_path
from actions.hand_gesture      import hand_gesture_control
from actions.voice_enrollment  import enroll_voice as _enroll_voice_fn
from core.claude_client        import is_claude_available as _check_claude


def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


BASE_DIR        = get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
PROMPT_PATH     = BASE_DIR / "core" / "prompt.txt"
LIVE_MODEL          = "models/gemini-2.5-flash-native-audio-preview-12-2025"
CHANNELS            = 1
SEND_SAMPLE_RATE    = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE          = 1024

def _get_api_key() -> str:
    with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["gemini_api_key"]


def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "당신은 WONJUNS, 원준의 멀티 AI 연구 비서입니다. "
            "간결하고 명확하게 답변하며, 항상 적절한 도구를 호출하여 작업을 완료합니다. "
            "결과를 추측하거나 시뮬레이션하지 말고 항상 도구를 호출하세요."
        )

_CTRL_RE = re.compile(r"<ctrl\d+>", re.IGNORECASE)

def _clean_transcript(text: str) -> str:
    text = _CTRL_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    # 연속 공백·줄바꿈 정리
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

TOOL_DECLARATIONS = [
    {
        "name": "open_app",
        "description": (
            "Opens any application on the computer. "
            "Use this whenever the user asks to open, launch, or start any app, "
            "website, or program. Always call this tool — never just say you opened it."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {
                    "type": "STRING",
                    "description": "Exact name of the application (e.g. 'WhatsApp', 'Chrome', 'Spotify')"
                }
            },
            "required": ["app_name"]
        }
    },
    {
        "name": "web_search",
        "description": "Searches the web for any information.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query":  {"type": "STRING", "description": "Search query"},
                "mode":   {"type": "STRING", "description": "search (default) or compare"},
                "items":  {"type": "ARRAY", "items": {"type": "STRING"}, "description": "Items to compare"},
                "aspect": {"type": "STRING", "description": "price | specs | reviews"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "weather_report",
        "description": "Gives the weather report to user",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "city": {"type": "STRING", "description": "City name"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "send_message",
        "description": "Sends a text message via WhatsApp, Telegram, or other messaging platform.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "receiver":     {"type": "STRING", "description": "Recipient contact name"},
                "message_text": {"type": "STRING", "description": "The message to send"},
                "platform":     {"type": "STRING", "description": "Platform: WhatsApp, Telegram, etc."}
            },
            "required": ["receiver", "message_text", "platform"]
        }
    },
    {
        "name": "reminder",
        "description": "Sets a timed reminder using Task Scheduler.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "date":    {"type": "STRING", "description": "Date in YYYY-MM-DD format"},
                "time":    {"type": "STRING", "description": "Time in HH:MM format (24h)"},
                "message": {"type": "STRING", "description": "Reminder message text"}
            },
            "required": ["date", "time", "message"]
        }
    },
    {
        "name": "youtube_video",
        "description": (
            "Controls YouTube. Use for: playing videos, summarizing a video's content, "
            "getting video info, or showing trending videos."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "play | summarize | get_info | trending (default: play)"},
                "query":  {"type": "STRING", "description": "Search query for play action"},
                "save":   {"type": "BOOLEAN", "description": "Save summary to Notepad (summarize only)"},
                "region": {"type": "STRING", "description": "Country code for trending e.g. TR, US"},
                "url":    {"type": "STRING", "description": "Video URL for get_info action"},
            },
            "required": []
        }
    },
    {
        "name": "screen_process",
        "description": (
            "Captures and analyzes the screen or webcam image. "
            "MUST be called when user asks what is on screen, what you see, "
            "analyze my screen, look at camera, etc. "
            "You have NO visual ability without this tool. "
            "After calling this tool, stay SILENT — the vision module speaks directly."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "angle": {"type": "STRING", "description": "'screen' to capture display, 'camera' for webcam. Default: 'screen'"},
                "text":  {"type": "STRING", "description": "The question or instruction about the captured image"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "computer_settings",
        "description": (
            "Controls the computer: volume, brightness, window management, keyboard shortcuts, "
            "typing text on screen, closing apps, fullscreen, dark mode, WiFi, restart, shutdown, "
            "scrolling, tab management, zoom, screenshots, lock screen, refresh/reload page. "
            "Use for ANY single computer control command."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "The action to perform"},
                "description": {"type": "STRING", "description": "Natural language description of what to do"},
                "value":       {"type": "STRING", "description": "Optional value: volume level, text to type, etc."}
            },
            "required": []
        }
    },
    {
        "name": "browser_control",
        "description": (
            "Controls any web browser. Use for: opening websites, searching the web, "
            "clicking elements, filling forms, scrolling, screenshots, navigation, any web-based task. "
            "Always pass the 'browser' parameter when the user specifies a browser (e.g. 'open in Edge', "
            "'use Firefox', 'open Chrome'). Multiple browsers can run simultaneously."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "go_to | search | click | type | scroll | fill_form | smart_click | smart_type | get_text | get_url | press | new_tab | close_tab | screenshot | back | forward | reload | switch | list_browsers | close | close_all"},
                "browser":     {"type": "STRING", "description": "Target browser: chrome | edge | firefox | opera | operagx | brave | vivaldi | safari. Omit to use the currently active browser."},
                "url":         {"type": "STRING", "description": "URL for go_to / new_tab action"},
                "query":       {"type": "STRING", "description": "Search query for search action"},
                "engine":      {"type": "STRING", "description": "Search engine: google | bing | duckduckgo | yandex (default: google)"},
                "selector":    {"type": "STRING", "description": "CSS selector for click/type"},
                "text":        {"type": "STRING", "description": "Text to click or type"},
                "description": {"type": "STRING", "description": "Element description for smart_click/smart_type"},
                "direction":   {"type": "STRING", "description": "up | down for scroll"},
                "amount":      {"type": "INTEGER", "description": "Scroll amount in pixels (default: 500)"},
                "key":         {"type": "STRING", "description": "Key name for press action (e.g. Enter, Escape, F5)"},
                "path":        {"type": "STRING", "description": "Save path for screenshot"},
                "incognito":   {"type": "BOOLEAN", "description": "Open in private/incognito mode"},
                "clear_first": {"type": "BOOLEAN", "description": "Clear field before typing (default: true)"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "file_controller",
        "description": "Manages files and folders: list, create, delete, move, copy, rename, read, write, find, disk usage.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "list | create_file | create_folder | delete | move | copy | rename | read | write | find | largest | disk_usage | organize_desktop | info"},
                "path":        {"type": "STRING", "description": "File/folder path or shortcut: desktop, downloads, documents, home"},
                "destination": {"type": "STRING", "description": "Destination path for move/copy"},
                "new_name":    {"type": "STRING", "description": "New name for rename"},
                "content":     {"type": "STRING", "description": "Content for create_file/write"},
                "name":        {"type": "STRING", "description": "File name to search for"},
                "extension":   {"type": "STRING", "description": "File extension to search (e.g. .pdf)"},
                "count":       {"type": "INTEGER", "description": "Number of results for largest"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "desktop_control",
        "description": "Controls the desktop: wallpaper, organize, clean, list, stats.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "wallpaper | wallpaper_url | organize | clean | list | stats | task"},
                "path":   {"type": "STRING", "description": "Image path for wallpaper"},
                "url":    {"type": "STRING", "description": "Image URL for wallpaper_url"},
                "mode":   {"type": "STRING", "description": "by_type or by_date for organize"},
                "task":   {"type": "STRING", "description": "Natural language desktop task"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "code_helper",
        "description": "Writes, edits, explains, runs, or builds code files.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "write | edit | explain | run | build | auto (default: auto)"},
                "description": {"type": "STRING", "description": "What the code should do or what change to make"},
                "language":    {"type": "STRING", "description": "Programming language (default: python)"},
                "output_path": {"type": "STRING", "description": "Where to save the file"},
                "file_path":   {"type": "STRING", "description": "Path to existing file for edit/explain/run/build"},
                "code":        {"type": "STRING", "description": "Raw code string for explain"},
                "args":        {"type": "STRING", "description": "CLI arguments for run/build"},
                "timeout":     {"type": "INTEGER", "description": "Execution timeout in seconds (default: 30)"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "dev_agent",
        "description": "Builds complete multi-file projects from scratch: plans, writes files, installs deps, opens VSCode, runs and fixes errors.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "description":  {"type": "STRING", "description": "What the project should do"},
                "language":     {"type": "STRING", "description": "Programming language (default: python)"},
                "project_name": {"type": "STRING", "description": "Optional project folder name"},
                "timeout":      {"type": "INTEGER", "description": "Run timeout in seconds (default: 30)"},
            },
            "required": ["description"]
        }
    },
    {
        "name": "computer_control",
        "description": "Direct computer control: type, click, hotkeys, scroll, move mouse, screenshots, find elements on screen.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "type | smart_type | click | double_click | right_click | hotkey | press | scroll | move | copy | paste | screenshot | wait | clear_field | focus_window | screen_find | screen_click | random_data | user_data"},
                "text":        {"type": "STRING", "description": "Text to type or paste"},
                "x":           {"type": "INTEGER", "description": "X coordinate"},
                "y":           {"type": "INTEGER", "description": "Y coordinate"},
                "keys":        {"type": "STRING", "description": "Key combination e.g. 'ctrl+c'"},
                "key":         {"type": "STRING", "description": "Single key e.g. 'enter'"},
                "direction":   {"type": "STRING", "description": "up | down | left | right"},
                "amount":      {"type": "INTEGER", "description": "Scroll amount (default: 3)"},
                "seconds":     {"type": "NUMBER",  "description": "Seconds to wait"},
                "title":       {"type": "STRING",  "description": "Window title for focus_window"},
                "description": {"type": "STRING",  "description": "Element description for screen_find/screen_click"},
                "type":        {"type": "STRING",  "description": "Data type for random_data"},
                "field":       {"type": "STRING",  "description": "Field for user_data: name|email|city"},
                "clear_first": {"type": "BOOLEAN", "description": "Clear field before typing (default: true)"},
                "path":        {"type": "STRING",  "description": "Save path for screenshot"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "game_updater",
        "description": (
            "THE ONLY tool for ANY Steam or Epic Games request. "
            "Use for: installing, downloading, updating games, listing installed games, "
            "checking download status, scheduling updates. "
            "ALWAYS call directly for any Steam/Epic/game request. "
            "NEVER use browser_control or web_search for Steam/Epic."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":    {"type": "STRING",  "description": "update | install | list | download_status | schedule | cancel_schedule | schedule_status (default: update)"},
                "platform":  {"type": "STRING",  "description": "steam | epic | both (default: both)"},
                "game_name": {"type": "STRING",  "description": "Game name (partial match supported)"},
                "app_id":    {"type": "STRING",  "description": "Steam AppID for install (optional)"},
                "hour":      {"type": "INTEGER", "description": "Hour for scheduled update 0-23 (default: 3)"},
                "minute":    {"type": "INTEGER", "description": "Minute for scheduled update 0-59 (default: 0)"},
                "shutdown_when_done": {"type": "BOOLEAN", "description": "Shut down PC when download finishes"},
            },
            "required": []
        }
    },
    {
        "name": "flight_finder",
        "description": "Searches Google Flights and speaks the best options.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "origin":      {"type": "STRING",  "description": "Departure city or airport code"},
                "destination": {"type": "STRING",  "description": "Arrival city or airport code"},
                "date":        {"type": "STRING",  "description": "Departure date (any format)"},
                "return_date": {"type": "STRING",  "description": "Return date for round trips"},
                "passengers":  {"type": "INTEGER", "description": "Number of passengers (default: 1)"},
                "cabin":       {"type": "STRING",  "description": "economy | premium | business | first"},
                "save":        {"type": "BOOLEAN", "description": "Save results to Notepad"},
            },
            "required": ["origin", "destination", "date"]
        }
    },
    {
        "name": "shutdown_assistant",
        "description": (
            "WONJUNS를 완전히 종료합니다. "
            "사용자가 대화 종료, 프로그램 닫기, 작별 인사, 'WONJUNS 꺼줘', '종료해줘', "
            "'잘자 원준스', 'goodbye', 'shut down', 'stop wonjuns' 등을 말하면 호출하세요. "
            "어떤 언어로든 종료 의도를 표현하면 즉시 호출합니다."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        }
    },
    {
    "name": "file_processor",
    "description": (
        "Processes any file that the user has uploaded or dropped onto the interface. "
        "Use this when the user refers to an uploaded file and wants an action on it. "
        "Supports: images (describe/ocr/resize/compress/convert), "
        "PDFs (summarize/extract_text/to_word), "
        "Word docs & text files (summarize/fix/reformat/translate), "
        "CSV/Excel (analyze/stats/filter/sort/convert), "
        "JSON/XML (validate/format/analyze), "
        "code files (explain/review/fix/optimize/run/document/test), "
        "audio (transcribe/trim/convert/info), "
        "video (trim/extract_audio/extract_frame/compress/transcribe/info), "
        "archives (list/extract), "
        "presentations (summarize/extract_text). "
        "ALWAYS call this tool when a file has been uploaded and the user gives a command about it. "
        "If the user's command is ambiguous, pick the most logical action for that file type."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "file_path": {
                "type": "STRING",
                "description": "Full path to the uploaded file. Leave empty to use the currently uploaded file."
            },
            "action": {
                "type": "STRING",
                "description": (
                    "What to do with the file. Examples by type:\n"
                    "image: describe | ocr | resize | compress | convert | info\n"
                    "pdf: summarize | extract_text | to_word | info\n"
                    "docx/txt: summarize | fix | reformat | translate_hint | word_count | to_bullet\n"
                    "csv/excel: analyze | stats | filter | sort | convert | info\n"
                    "json: validate | format | analyze | to_csv\n"
                    "code: explain | review | fix | optimize | run | document | test\n"
                    "audio: transcribe | trim | convert | info\n"
                    "video: trim | extract_audio | extract_frame | compress | transcribe | info | convert\n"
                    "archive: list | extract\n"
                    "pptx: summarize | extract_text | analyze"
                )
            },
            "instruction": {
                "type": "STRING",
                "description": "Free-form instruction if action doesn't cover it. E.g. 'translate this to Turkish', 'find all email addresses'"
            },
            "format": {
                "type": "STRING",
                "description": "Target format for conversion. E.g. 'mp3', 'pdf', 'csv', 'png'"
            },
            "width":     {"type": "INTEGER", "description": "Target width for image resize"},
            "height":    {"type": "INTEGER", "description": "Target height for image resize"},
            "scale":     {"type": "NUMBER",  "description": "Scale factor for image resize (e.g. 0.5)"},
            "quality":   {"type": "INTEGER", "description": "Quality 1-100 for image/video compress"},
            "start":     {"type": "STRING",  "description": "Start time for trim: seconds or HH:MM:SS"},
            "end":       {"type": "STRING",  "description": "End time for trim: seconds or HH:MM:SS"},
            "timestamp": {"type": "STRING",  "description": "Timestamp for video frame extraction HH:MM:SS"},
            "column":    {"type": "STRING",  "description": "Column name for CSV filter/sort"},
            "value":     {"type": "STRING",  "description": "Filter value for CSV filter"},
            "condition": {"type": "STRING",  "description": "Filter condition: equals|contains|gt|lt"},
            "ascending": {"type": "BOOLEAN", "description": "Sort order for CSV sort (default: true)"},
            "save":      {"type": "BOOLEAN", "description": "Save result to file (default: true)"},
            "destination": {"type": "STRING", "description": "Output folder for archive extract"},
        },
        "required": []
    }
},
    {
        "name": "analyze_research_document",
        "description": (
            "역사학 연구 문서(논문, 역사 자료)를 심층 분석합니다. "
            "PDF, HWP(한글), DOCX(워드), JPG, PNG 형식을 지원합니다. "
            "목차를 자동으로 감지하여 장/절/소절 구조를 파악합니다. "
            "각 인용문이 몇 장 몇 절 몇 페이지에서 나왔는지 정확히 추적합니다. "
            "미국 자료(정부문서, 신문, 외교전문 등)는 날짜·기관·기밀등급·발신수신자를 추출합니다. "
            "한국 학술논문은 역사학적 시기·지역·계층·연구사 공백을 분석합니다. "
            "미국 국립문서기록청(NARA) 자료의 경우 Record Group(RG)과 Entry 번호를 함께 전달하면 "
            "옵시디안 노트에 NARA 아카이브 출처가 정확히 기록됩니다. "
            "사용자가 'RG 59, Entry 1234' 같은 아카이브 식별자를 언급하면 반드시 파라미터로 전달하세요. "
            "분석 결과는 옵시디안(Obsidian) 볼트에 자동 저장됩니다. "
            "사용자가 논문이나 자료 파일을 분석해달라고 하면 반드시 이 도구를 사용하세요."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "file_path": {
                    "type": "STRING",
                    "description": "분석할 파일의 전체 경로 (PDF, HWP, DOCX, JPG, PNG)"
                },
                "record_group": {
                    "type": "STRING",
                    "description": (
                        "미국 NARA 레코드그룹 번호. "
                        "사용자가 'RG 59', '레코드그룹 59', 'Record Group 59' 등을 언급하면 '59' 또는 'RG 59'로 추출. "
                        "예: '59', 'RG 59', 'RG 242', 'RG 319'"
                    )
                },
                "entry": {
                    "type": "STRING",
                    "description": (
                        "미국 NARA 엔트리 번호. "
                        "사용자가 'Entry 1234', '엔트리 1234', 'Entry A1 1234' 등을 언급하면 추출. "
                        "예: '1234', 'A1 1234', 'UD-WW 1234'"
                    )
                },
                "box": {
                    "type": "STRING",
                    "description": "박스 번호 (언급된 경우). 예: '5', 'Box 5'"
                },
                "folder": {
                    "type": "STRING",
                    "description": "폴더명 (언급된 경우). 예: 'Korea 1945'"
                },
                "save_to_obsidian": {
                    "type": "BOOLEAN",
                    "description": "분석 결과를 옵시디안에 저장할지 여부 (기본값: true)"
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "find_research_gaps",
        "description": (
            "지금까지 분석된 모든 역사학 논문과 자료를 바탕으로 "
            "연구사의 공백과 새로운 문제의식을 도출합니다. "
            "연구자가 '연구 공백 찾아줘', '새로운 주제 추천해줘', "
            "'지금까지 정리된 연구 분석해줘' 등을 요청할 때 사용하세요."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        }
    },
    {
        "name": "generate_footnote",
        "description": (
            "특정 문서의 특정 페이지에서 인용문을 찾아 '제N장 > 제N절 > p.N' 형식의 "
            "정밀 출처가 포함된 논문 각주를 생성합니다. "
            "목차가 감지된 문서의 경우 어느 장 어느 절에서 인용했는지도 자동으로 식별합니다. "
            "사용자가 '~쪽에서 각주 만들어줘', '~페이지 인용 찾아줘', "
            "'N장 N절 인용 뽑아줘' 등을 요청할 때 사용하세요."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "file_path": {
                    "type": "STRING",
                    "description": "대상 파일의 전체 경로"
                },
                "page": {
                    "type": "INTEGER",
                    "description": "인용문이 있는 페이지 번호"
                },
                "quote_hint": {
                    "type": "STRING",
                    "description": "찾을 인용문의 키워드나 일부 내용"
                }
            },
            "required": ["file_path", "page", "quote_hint"]
        }
    },
    {
        "name": "liner_research",
        "description": (
            "웹 URL의 학술 자료를 Liner AI로 수집·하이라이팅한 뒤 Claude AI로 심층 분석합니다. "
            "RISS, KISS, DBpia, Google Scholar 등 온라인 논문 URL을 분석할 때 사용하세요. "
            "로컬 파일(PDF/HWP/DOCX)은 analyze_research_document를 사용하세요. "
            "사용자가 '이 링크 분석해줘', 'URL 논문 읽어줘', '라이너로 저장해줘' 등을 요청할 때 사용하세요."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "url": {
                    "type": "STRING",
                    "description": "분석할 학술 논문 또는 자료의 웹 URL"
                },
                "question": {
                    "type": "STRING",
                    "description": "Liner AI에게 할 질문 (없으면 기본 역사학 분석 질문 사용)"
                },
                "save_to_obsidian": {
                    "type": "BOOLEAN",
                    "description": "결과를 옵시디안에 저장할지 여부 (기본값: true)"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "set_obsidian_vault",
        "description": (
            "옵시디안 볼트(저장소) 경로를 설정합니다. "
            "처음 사용 시 또는 볼트 경로 변경 시 반드시 설정해야 합니다. "
            "사용자가 '옵시디안 볼트 설정', '저장 경로 지정' 등을 요청할 때 사용하세요."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "vault_path": {
                    "type": "STRING",
                    "description": "옵시디안 볼트 폴더의 절대 경로 (예: C:/Users/user/Obsidian/History)"
                }
            },
            "required": ["vault_path"]
        }
    },
    {
        "name": "hand_gesture_control",
        "description": (
            "노트북 카메라로 손 제스처를 인식하여 컴퓨터를 제어합니다. "
            "제스처 목록: 주먹(미디어 일시정지/재생), 손바닥 펼치기(스크롤 중지), "
            "검지 포인터(마우스 이동), 평화의 손(볼륨 조절), 엄지 위(볼륨 증가), "
            "엄지 아래(볼륨 감소), OK 손(클릭), 네 손가락(Alt+Tab), "
            "스와이프 좌우(가상 데스크탑 전환). "
            "사용자가 '손 인식 켜줘', '제스처 컨트롤 시작', '손으로 컴퓨터 제어', "
            "'핸드 트래킹 시작' 등을 요청하면 start 명령을 호출하세요. "
            "'손 인식 꺼줘', '제스처 종료'는 stop 명령을 사용하세요."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "start | stop | status | smoothing"
                },
                "camera_index": {
                    "type": "INTEGER",
                    "description": "카메라 번호 (기본값: 0, 노트북 내장 카메라)"
                },
                "show_preview": {
                    "type": "BOOLEAN",
                    "description": "카메라 미리보기 창 표시 여부 (기본값: true)"
                },
                "value": {
                    "type": "NUMBER",
                    "description": "스무딩 값 (smoothing 명령 시 사용, 0.05~1.0, 기본값: 0.20)"
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "write_research_draft",
        "description": (
            "학습된 논문들의 실제 문체를 참조하여 학술 한국어 산문을 작성합니다. "
            "AI 투의 작위적 표현 없이 실제 역사학 논문처럼 작성합니다. "
            "'서론 써줘', '이 내용 논문 스타일로 작성해줘', '결론 단락 만들어줘', "
            "'이 주제로 논문 초고 써줘', '문장 다듬어줘', '학술 문체로 바꿔줘' 등에 즉시 호출하세요. "
            "논문 분석이 많을수록 더 정확한 문체 모방이 가능합니다."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "topic": {
                    "type": "STRING",
                    "description": (
                        "작성 주제 또는 다듬을 문장/단락. "
                        "예: '한국전쟁 초기 미국의 개입 결정과 그 배경', "
                        "'이 문장을 다듬어줘: [원문]'"
                    )
                },
                "content": {
                    "type": "STRING",
                    "description": (
                        "포함할 아이디어, 키워드, 핵심 내용 (선택). "
                        "예: '맥아더 전략, 유엔 결의안 678호, 인천상륙작전'"
                    )
                },
                "section_type": {
                    "type": "STRING",
                    "description": (
                        "작성할 단락 유형. "
                        "intro(서론) / body(본문) / argument(논증) / "
                        "conclusion(결론) / analysis(사료분석) / refine(문장다듬기). "
                        "기본값: body"
                    )
                },
                "length": {
                    "type": "STRING",
                    "description": (
                        "분량. short(2~3문장) / medium(1단락) / long(2~3단락). "
                        "기본값: medium"
                    )
                },
            },
            "required": ["topic"]
        }
    },
    {
        "name": "check_writing_style",
        "description": (
            "작성한 초고에서 AI 투의 상투적 표현을 찾아 개선점을 알려줍니다. "
            "'이 문장 AI 같아 보여?', '초고 검토해줘', 'AI 표현 있어?', "
            "'내 글 스타일 체크해줘' 등에 호출하세요."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "draft": {
                    "type": "STRING",
                    "description": "검토할 초고 텍스트"
                }
            },
            "required": ["draft"]
        }
    },
    {
        "name": "get_style_corpus_stats",
        "description": (
            "학습된 논문 문체 코퍼스 현황을 보여줍니다. "
            "'문체 학습 현황 보여줘', '몇 개 논문 학습됐어', '코퍼스 상태' 등에 호출하세요."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "enroll_voice",
        "description": (
            "사용자 목소리를 학습하여 웨이크워드 화자 인증 프로필을 저장합니다. "
            "등록 완료 후 원준씨 목소리에만 웨이크워드('Wake up Wonjuns', '원준아 일어나' 등)가 반응합니다. "
            "'내 목소리 등록해줘', '목소리 학습해줘', '목소리 등록', '화자 인증 설정', "
            "'내 목소리만 인식해줘', '목소리 프로필 만들어줘' 등의 발화에 즉시 호출하세요."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "n_samples": {
                    "type": "INTEGER",
                    "description": "녹음할 샘플 수 (기본값: 5, 최소 3, 최대 10)"
                }
            },
            "required": []
        }
    },
    {
        "name": "save_memory",
        "description": (
            "Save an important personal fact about the user to long-term memory. "
            "Call this silently whenever the user reveals something worth remembering: "
            "name, age, city, job, preferences, hobbies, relationships, projects, or future plans. "
            "Do NOT call for: weather, reminders, searches, or one-time commands. "
            "Do NOT announce that you are saving — just call it silently. "
            "Values must be in English regardless of the conversation language."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {
                    "type": "STRING",
                    "description": (
                        "identity — name, age, birthday, city, job, language, nationality | "
                        "preferences — favorite food/color/music/film/game/sport, hobbies | "
                        "projects — active projects, goals, things being built | "
                        "relationships — friends, family, partner, colleagues | "
                        "wishes — future plans, things to buy, travel dreams | "
                        "notes — habits, schedule, anything else worth remembering"
                    )
                },
                "key":   {"type": "STRING", "description": "Short snake_case key (e.g. name, favorite_food, sister_name)"},
                "value": {"type": "STRING", "description": "Concise value in English (e.g. Fatih, pizza, older sister)"},
            },
            "required": ["category", "key", "value"]
        }
    },
]

# --- Wake word detector ---

class WakeWordDetector:
    """
    'Wake up Wonjuns' / '원준아 일어나' 등의 구문을 감지하면 on_wake() 를 호출합니다.
    영어(en-US)와 한국어(ko-KR) 인식을 순서대로 시도합니다.
    """

    # ── 영어 웨이크 문구 ─────────────────────────────────────────────────────
    _EN_PHRASES = [
        "wake up wonjuns", "wake up wanjuns", "wake up won juns",
        "wonjuns wake up", "hey wonjuns",    "hi wonjuns",
        "wake wonjuns",    "okay wonjuns",   "ok wonjuns",
    ]

    # ── 한국어 웨이크 문구 ───────────────────────────────────────────────────
    _KO_PHRASES = [
        # 일어나 (자다가 일어나 = wake up) — 가장 자연스러운 호출
        "원준아 일어나",   "원준 일어나",    "원준스 일어나",
        "일어나 원준아",   "일어나 원준",    "일어나 원준스",
        "원준아 일어나요", "원준아 일어나봐", "원준아 일어나라",
        # 깨어나 / 깨워 (깨어나다 = awaken)
        "원준아 깨어나",   "원준 깨어나",    "원준스 깨어나",
        "원준아 깨워",     "원준 깨워",      "원준스 깨워",
        "깨어나 원준아",   "깨어나 원준",
        "원준아 깨어나요", "원준아 깨워줘",
        # 켜줘 / 켜 (전원 켜기 = turn on)
        "원준아 켜줘",     "원준 켜줘",      "원준스 켜줘",
        "원준아 켜",       "켜줘 원준아",    "원준아 켜봐",
        # 시작 / 시작해 (start)
        "원준아 시작",     "원준아 시작해",  "원준아 시작해줘",
        "원준 시작",       "원준스 시작",    "시작해 원준아",
        "원준아 시작하자", "원준아 시작할게",
        # 활성화 (activate)
        "원준아 활성화",   "원준 활성화",    "원준스 활성화",
        "원준아 활성화해", "원준아 활성화해줘",
        # 기동 (boot / launch)
        "원준아 기동",     "원준 기동",      "원준스 기동",
        "원준아 기동해",   "원준아 기동해줘",
        # 준비 (get ready)
        "원준아 준비해",   "원준아 준비",    "원준 준비해",
        "원준스 준비",
    ]

    def __init__(self, sample_rate: int = 16000, on_wake=None):
        self._rate              = sample_rate
        self._on_wake           = on_wake
        self._q: _queue.Queue[bytes] = _queue.Queue(maxsize=500)
        self._active            = True
        self._cooldown          = 0.0
        self._voice_profile     = None   # np.ndarray or None
        self._encoder           = None   # resemblyzer VoiceEncoder (lazy)
        self._verify_threshold  = 0.80
        self._load_voice_profile()
        threading.Thread(
            target=self._loop, daemon=True, name="wake-detector"
        ).start()

    def _load_voice_profile(self):
        """config/voice_profile.npy 로드 (없으면 None 유지)."""
        try:
            from actions.voice_enrollment import load_voice_profile
            self._voice_profile = load_voice_profile()
            if self._voice_profile is not None:
                print("[WONJUNS] 🔐 목소리 프로필 로드 완료 — 화자 인증 활성화")
            else:
                print("[WONJUNS] ℹ️  목소리 프로필 없음 — 모든 목소리 허용")
        except Exception as e:
            print(f"[WONJUNS] ⚠️  프로필 로드 실패: {e}")
            self._voice_profile = None

    def reload_profile(self):
        """목소리 등록 완료 후 새 프로필을 다시 읽어들입니다."""
        self._encoder = None  # 인코더 캐시 초기화
        self._load_voice_profile()

    def _get_encoder(self):
        """resemblyzer VoiceEncoder 지연 로딩 및 캐싱."""
        if self._encoder is None:
            try:
                # webrtcvad 스텁 먼저 주입 (Python 3.14 호환)
                from actions.voice_enrollment import _inject_webrtcvad_stub_if_needed
                _inject_webrtcvad_stub_if_needed()
                from resemblyzer import VoiceEncoder
                self._encoder = VoiceEncoder()
            except ImportError:
                pass  # resemblyzer 미설치 시 검증 건너뜀
        return self._encoder

    def _verify_speaker(self, audio_bytes: bytes) -> bool:
        """
        프로필이 없으면 True(무조건 통과),
        프로필이 있으면 코사인 유사도 ≥ threshold 여부 반환.
        """
        if self._voice_profile is None:
            return True
        encoder = self._get_encoder()
        if encoder is None:
            return True  # resemblyzer 미설치 — 검증 건너뜀
        try:
            from actions.voice_enrollment import verify_speaker
            passed, score = verify_speaker(
                audio_bytes, self._rate,
                threshold=self._verify_threshold,
                encoder=encoder,
            )
            if passed:
                print(f"[WONJUNS] 🔐 화자 인증 통과 (유사도: {score:.2f})")
            else:
                print(f"[WONJUNS] 🔐 화자 인증 실패 — 다른 목소리 (유사도: {score:.2f})")
            return passed
        except Exception as e:
            print(f"[WONJUNS] ⚠️  화자 검증 오류: {e} — 통과 처리")
            return True

    def feed(self, pcm_bytes: bytes):
        try:
            self._q.put_nowait(pcm_bytes)
        except _queue.Full:
            pass

    def stop(self):
        self._active = False

    def _loop(self):
        import time
        from core.stt_engine import transcribe_korean, is_faster_whisper_available

        # STT 엔진 상태 출력
        if is_faster_whisper_available():
            print("[WONJUNS] 🎙️ 웨이크워드 엔진: faster-whisper (오프라인 한국어 고정밀)")
        else:
            # faster-whisper 없으면 Google Speech API 폴백
            try:
                import speech_recognition  # noqa
                print("[WONJUNS] 🎙️ 웨이크워드 엔진: Google Speech API (pip install faster-whisper 권장)")
            except ImportError:
                print("[WONJUNS] ⚠️  STT 엔진 없음 — 웨이크워드 비활성")
                print("[WONJUNS]    설치: pip install faster-whisper")
                return

        WINDOW = self._rate * 4 * 2  # 4초 윈도우 @ 16kHz int16 (기존 3초 → 향상)
        POLL   = 0.08                # 80ms 폴링 (기존 400ms → 5배 단축)
        acc    = b""

        while self._active:
            time.sleep(POLL)

            chunks: list[bytes] = []
            try:
                while True:
                    chunks.append(self._q.get_nowait())
            except _queue.Empty:
                pass

            if not chunks:
                continue

            acc += b"".join(chunks)
            if len(acc) > WINDOW * 2:
                acc = acc[-WINDOW:]

            # 최소 1초 이상 누적돼야 인식 시작
            if len(acc) < self._rate * 2:
                continue

            now = time.time()
            if now < self._cooldown:
                continue

            # ── 한국어 웨이크워드 인식 (Korean-first) ────────────────────────
            result = transcribe_korean(acc[-WINDOW:], self._rate)

            if not result:
                continue

            detected_text   = result.text
            phrase_detected = False

            # ① 한국어 문구 확인 (우선)
            if any(p in detected_text for p in self._KO_PHRASES):
                phrase_detected = True
                print(f"[WONJUNS] 🎤 한국어 웨이크워드 감지: '{detected_text}' [{result.engine}]")

            # ② 영어 문구 확인 (보조)
            if not phrase_detected and any(p in detected_text for p in self._EN_PHRASES):
                phrase_detected = True
                print(f"[WONJUNS] 🎤 영어 웨이크워드 감지: '{detected_text}' [{result.engine}]")

            if phrase_detected:
                # ③ 화자 인증 — 등록된 프로필이 있을 때만 검사
                if self._verify_speaker(acc[-WINDOW:]):
                    self._cooldown = now + 4.0
                    if self._on_wake:
                        self._on_wake()
                else:
                    print("[WONJUNS] 🔐 웨이크워드 무시 — 등록된 목소리가 아님")
                    self._cooldown = now + 1.0  # 짧은 쿨다운


# --- Plugin system ---


class AssistantLive:

    def __init__(self, ui: AssistantUI):
        self.ui             = ui
        self.session        = None
        self.audio_in_queue = None
        self.out_queue      = None
        self._loop          = None
        self._is_speaking   = False
        self._speaking_lock = threading.Lock()
        self._phone_active  = False   # True while phone mic is streaming; pauses PC mic
        self.ui.on_text_command  = self._on_text_command
        self.ui.on_remote_clicked = self._make_remote_key
        self._turn_done_event: asyncio.Event | None = None
        self._dashboard     = None
        self._wake_detector: WakeWordDetector | None = None

    def _make_remote_key(self):
        """Called from Qt main thread when user presses Remote Control."""
        if self._dashboard is None:
            self.ui.write_log(
                "SYS: Dashboard unavailable. "
                "Run: pip install fastapi \"uvicorn[standard]\" cryptography"
            )
            return None
        key    = self._dashboard.new_key()
        url    = self._dashboard.get_url()
        manual = self._dashboard.get_manual_url()
        return url, key, f"{url}/auto-login?key={key}", manual

    def _on_text_command(self, text: str):
        if not self._loop or not self.session:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )

    def set_speaking(self, value: bool):
        with self._speaking_lock:
            self._is_speaking = value
        if value:
            self.ui.set_state("SPEAKING")
        elif not self.ui.muted:
            self.ui.set_state("LISTENING")

    def speak(self, text: str):
        if not self._loop or not self.session:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )

    def speak_error(self, tool_name: str, error: str):
        short = str(error)[:120]
        self.ui.write_log(f"ERR: {tool_name} — {short}")
        self.speak(f"원준씨, {tool_name} 실행 중 오류가 발생했습니다. {short}")

    def _on_wake_word(self):
        """웨이크워드 감지 콜백 — 뮤트 해제 후 Gemini에 인사 요청."""
        if not self.ui.muted:
            return
        self.ui.wake_up()
        def _greet():
            import time
            time.sleep(0.5)
            if self._loop and self.session:
                asyncio.run_coroutine_threadsafe(
                    self.session.send_client_content(
                        turns={"parts": [{
                            "text": "웨이크워드로 활성화됐습니다. 원준씨에게 준비됐다고 짧게 인사하세요."
                        }]},
                        turn_complete=True,
                    ),
                    self._loop,
                )
        threading.Thread(target=_greet, daemon=True).start()

    def _build_config(self) -> types.LiveConnectConfig:
        from datetime import datetime

        memory     = load_memory()
        mem_str    = format_memory_for_prompt(memory)
        sys_prompt = _load_system_prompt()

        now      = datetime.now()
        time_str = now.strftime("%A, %B %d, %Y — %I:%M %p")
        time_ctx = (
            f"[CURRENT DATE & TIME]\n"
            f"Right now it is: {time_str}\n"
            f"Use this to calculate exact times for reminders.\n\n"
        )

        # 역사학 연구 지식 그래프 요약 (옵시디안 연동 시)
        try:
            knowledge_ctx = get_knowledge_summary()
        except Exception:
            knowledge_ctx = ""

        parts = [time_ctx]
        if mem_str:
            parts.append(mem_str)
        if knowledge_ctx:
            parts.append(knowledge_ctx)
        parts.append(sys_prompt)

        # ── 한국어 음성 인식 최적화 지시 ─────────────────────────────────────
        ko_speech_hint = (
            "\n[음성 인식 설정]\n"
            "사용자의 모국어는 한국어입니다.\n"
            "• 발화가 불명확하거나 잡음이 있어도 한국어로 최대한 해석하세요.\n"
            "• 한국어 구어체를 자연스럽게 이해합니다: 반말/존댓말 혼용, 축약형(어떡게→어떻게), 연음, 필러('어', '음', '그', '저').\n"
            "• 필러 단어는 의도 파악 후 무시합니다.\n"
            "• 영어 단어가 섞인 한국어 발화도 그대로 처리합니다.\n"
        )
        parts.append(ko_speech_hint)

        # ── 한국어 전사 언어 코드 설정 ────────────────────────────────────────
        try:
            transcription_cfg = types.AudioTranscriptionConfig(language_code="ko-KR")
        except Exception:
            transcription_cfg = {}

        # ── VAD 감도 설정 (한국어: 끊김 없이 말하는 경향 반영) ───────────────
        try:
            realtime_cfg = types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=False,
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                    prefix_padding_ms=300,
                    silence_duration_ms=800,  # 한국어 발화 사이 쉬는 시간 반영
                )
            )
        except Exception:
            realtime_cfg = None

        config_kwargs = dict(
            response_modalities=["AUDIO"],
            output_audio_transcription=transcription_cfg,
            input_audio_transcription=transcription_cfg,
            system_instruction="\n".join(parts),
            tools=[{"function_declarations": TOOL_DECLARATIONS}],
            session_resumption=types.SessionResumptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Charon"
                    )
                )
            ),
        )
        if realtime_cfg is not None:
            config_kwargs["realtime_input_config"] = realtime_cfg

        return types.LiveConnectConfig(**config_kwargs)

    async def _execute_tool(self, fc) -> types.FunctionResponse:
        name = fc.name
        args = dict(fc.args or {})

        print(f"[WONJUNS] 🔧 {name}  {args}")
        self.ui.set_state("THINKING")

        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}})
                print(f"[Memory] 💾 save_memory: {category}/{key} = {value}")
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "ok", "silent": True}
            )

        loop   = asyncio.get_event_loop()
        result = "Done."

        try:
            if name == "open_app":
                r = await loop.run_in_executor(None, lambda: open_app(parameters=args, response=None, player=self.ui))
                result = r or f"Opened {args.get('app_name')}."

            elif name == "weather_report":
                r = await loop.run_in_executor(None, lambda: weather_action(parameters=args, player=self.ui))
                result = r or "Weather delivered."

            elif name == "browser_control":
                r = await loop.run_in_executor(None, lambda: browser_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "file_controller":
                r = await loop.run_in_executor(None, lambda: file_controller(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "send_message":
                r = await loop.run_in_executor(None, lambda: send_message(parameters=args, response=None, player=self.ui, session_memory=None))
                result = r or f"Message sent to {args.get('receiver')}."

            elif name == "reminder":
                r = await loop.run_in_executor(None, lambda: reminder(parameters=args, response=None, player=self.ui))
                result = r or "Reminder set."

            elif name == "youtube_video":
                r = await loop.run_in_executor(None, lambda: youtube_video(parameters=args, response=None, player=self.ui))
                result = r or "Done."

            elif name == "screen_process":
                threading.Thread(
                    target=screen_process,
                    kwargs={"parameters": args, "response": None,
                            "player": self.ui, "session_memory": None},
                    daemon=True
                ).start()
                result = "Vision module activated. Stay completely silent — vision module will speak directly."

            elif name == "computer_settings":
                r = await loop.run_in_executor(None, lambda: computer_settings(parameters=args, response=None, player=self.ui))
                result = r or "Done."

            elif name == "desktop_control":
                r = await loop.run_in_executor(None, lambda: desktop_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "code_helper":
                r = await loop.run_in_executor(None, lambda: code_helper(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "dev_agent":
                r = await loop.run_in_executor(None, lambda: dev_agent(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "web_search":
                r = await loop.run_in_executor(None, lambda: web_search_action(parameters=args, player=self.ui))
                result = r or "Done."
            elif name == "file_processor":
                if not args.get("file_path") and self.ui.current_file:
                    args["file_path"] = self.ui.current_file
                r = await loop.run_in_executor(
                    None,
                    lambda: file_processor(parameters=args, player=self.ui, speak=self.speak)
                )
                result = r or "Done."

            elif name == "computer_control":
                r = await loop.run_in_executor(None, lambda: computer_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "game_updater":
                r = await loop.run_in_executor(None, lambda: game_updater(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "flight_finder":
                r = await loop.run_in_executor(None, lambda: flight_finder(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "analyze_research_document":
                if not args.get("file_path") and self.ui.current_file:
                    args["file_path"] = self.ui.current_file
                ai_label = "Claude" if _check_claude() else "Gemini"
                rg_info = ""
                if args.get("record_group"):
                    rg_info = f" [RG {args['record_group']}"
                    if args.get("entry"):
                        rg_info += f" / Entry {args['entry']}"
                    rg_info += "]"
                self.ui.write_log(
                    f"SYS: 문서 분석 시작 [{ai_label}]{rg_info} — {args.get('file_path', '')}"
                )
                ui_ref = self.ui
                r = await loop.run_in_executor(
                    None,
                    lambda: history_research_action("analyze_document", args, player=ui_ref)
                )
                result = r or "문서 분석이 완료되었습니다."

            elif name == "liner_research":
                url = args.get("url", "")
                self.ui.write_log(f"SYS: Liner + Claude 분석 시작 — {url}")
                ui_ref = self.ui
                r = await loop.run_in_executor(
                    None,
                    lambda: history_research_action("analyze_url", args, player=ui_ref)
                )
                result = r or "URL 분석이 완료되었습니다."

            elif name == "write_research_draft":
                topic = args.get("topic", "")
                stype = args.get("section_type", "body")
                self.ui.write_log(f"SYS: 학술 작문 생성 중 [{stype}] — {topic[:40]}")
                ui_ref = self.ui
                r = await loop.run_in_executor(
                    None,
                    lambda: history_research_action("write_research_draft", args, player=ui_ref)
                )
                result = r or "작문이 완료되었습니다."

            elif name == "check_writing_style":
                draft = args.get("draft", "")
                self.ui.write_log(f"SYS: 초고 문체 검토 중 ({len(draft)}자)")
                r = await loop.run_in_executor(
                    None,
                    lambda: history_research_action("check_writing_style", args)
                )
                result = r or "문체 검토가 완료되었습니다."

            elif name == "get_style_corpus_stats":
                r = await loop.run_in_executor(
                    None,
                    lambda: history_research_action("get_style_corpus_stats", {})
                )
                result = r or "코퍼스 통계를 불러왔습니다."

            elif name == "find_research_gaps":
                ai_label = "Claude + " if _check_claude() else ""
                self.ui.write_log(f"SYS: {ai_label}연구사 공백 분석 중...")
                ui_ref = self.ui
                r = await loop.run_in_executor(
                    None,
                    lambda: history_research_action("find_research_gaps", {}, player=ui_ref)
                )
                result = r or "연구사 공백 분석이 완료되었습니다."

            elif name == "generate_footnote":
                ui_ref = self.ui
                r = await loop.run_in_executor(
                    None,
                    lambda: history_research_action("generate_footnote", args, player=ui_ref)
                )
                result = r or "각주 생성이 완료되었습니다."

            elif name == "set_obsidian_vault":
                r = await loop.run_in_executor(
                    None,
                    lambda: history_research_action(
                        "set_vault_path",
                        {"vault_path": args.get("vault_path", "")}
                    )
                )
                result = r or "옵시디안 볼트가 설정되었습니다."

            elif name == "hand_gesture_control":
                action = args.get("action", "start")
                self.ui.write_log(f"SYS: 손 제스처 제어 — {action}")
                ui_ref = self.ui
                speak_fn = self.speak
                r = await loop.run_in_executor(
                    None,
                    lambda: hand_gesture_control(action, args, player=ui_ref, speak_fn=speak_fn)
                )
                result = r or "완료."

            elif name == "enroll_voice":
                n_samples = int(args.get("n_samples", 5))
                self.ui.write_log(f"SYS: 목소리 등록 시작 — {n_samples}개 샘플")
                ui_ref   = self.ui
                speak_fn = self.speak
                r = await loop.run_in_executor(
                    None,
                    lambda: _enroll_voice_fn(
                        n_samples=n_samples,
                        player=ui_ref,
                        speak_fn=speak_fn,
                    )
                )
                result = r or "목소리 등록이 완료되었습니다."
                # 웨이크워드 감지기에 새 프로필 즉시 반영
                if self._wake_detector:
                    self._wake_detector.reload_profile()

            elif name == "shutdown_assistant":
                self.ui.write_log("SYS: Shutdown requested.")
                self.speak("알겠습니다, 원준씨. 종료합니다.")
                def _shutdown():
                    import time, os
                    time.sleep(1)
                    os._exit(0)
                threading.Thread(target=_shutdown, daemon=True).start()

            else:
                result = f"Unknown tool: {name}"

        except Exception as e:
            result = f"Tool '{name}' failed: {e}"
            traceback.print_exc()
            self.speak_error(name, e)

        if not self.ui.muted:
            self.ui.set_state("LISTENING")

        print(f"[WONJUNS] 📤 {name} → {str(result)[:80]}")
        return types.FunctionResponse(
            id=fc.id, name=name,
            response={"result": result}
        )

    async def _send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send_realtime_input(media=msg)

    async def _listen_audio(self):
        print("[WONJUNS] 🎤 Mic started")
        loop = asyncio.get_event_loop()

        def callback(indata, frames, time_info, status):
            data = indata.tobytes()
            # 웨이크워드 감지기에 항상 공급 (뮤트 여부 무관)
            if self._wake_detector:
                self._wake_detector.feed(data)
            # 음소거 해제 시에만 Gemini로 전송
            with self._speaking_lock:
                wonjuns_speaking = self._is_speaking
            if not wonjuns_speaking and not self.ui.muted and not self._phone_active:
                loop.call_soon_threadsafe(
                    self.out_queue.put_nowait,
                    {"data": data, "mime_type": "audio/pcm"}
                )

        try:
            with sd.InputStream(
                samplerate=SEND_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                callback=callback,
            ):
                print("[WONJUNS] 🎤 Mic stream open")
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[WONJUNS] ❌ Mic: {e}")
            raise

    async def _receive_audio(self):
        print("[WONJUNS] 👂 Recv started")
        out_buf, in_buf = [], []

        try:
            while True:
                async for response in self.session.receive():

                    if response.data:
                        if self._turn_done_event and self._turn_done_event.is_set():
                            self._turn_done_event.clear()
                        self.audio_in_queue.put_nowait(response.data)

                    if response.server_content:
                        sc = response.server_content

                        if sc.output_transcription and sc.output_transcription.text:
                            txt = _clean_transcript(sc.output_transcription.text)
                            if txt:
                                out_buf.append(txt)

                        if sc.input_transcription and sc.input_transcription.text:
                            txt = _clean_transcript(sc.input_transcription.text)
                            if txt:
                                in_buf.append(txt)

                        if sc.turn_complete:
                            if self._turn_done_event:
                                self._turn_done_event.set()

                            full_in = " ".join(in_buf).strip()
                            if full_in:
                                self.ui.write_log(f"You: {full_in}")
                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "user",
                                        "text": full_in,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            in_buf = []

                            full_out = " ".join(out_buf).strip()
                            if full_out:
                                self.ui.write_log(f"Wonjuns: {full_out}")
                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "assistant",
                                        "text": full_out,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            out_buf = []

                    if response.tool_call:
                        fn_responses = []
                        for fc in response.tool_call.function_calls:
                            print(f"[WONJUNS] 📞 {fc.name}")
                            fr = await self._execute_tool(fc)
                            fn_responses.append(fr)
                        await self.session.send_tool_response(
                            function_responses=fn_responses
                        )
        except Exception as e:
            print(f"[WONJUNS] ❌ Recv: {e}")
            traceback.print_exc()
            raise

    async def _play_audio(self):
        print("[WONJUNS] 🔊 Play started")

        stream = sd.RawOutputStream(
            samplerate=RECEIVE_SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=CHUNK_SIZE,
        )
        stream.start()

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        self.audio_in_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    if (
                        self._turn_done_event
                        and self._turn_done_event.is_set()
                        and self.audio_in_queue.empty()
                    ):
                        self.set_speaking(False)
                        self._turn_done_event.clear()
                    continue
                self.set_speaking(True)
                await asyncio.to_thread(stream.write, chunk)
        except Exception as e:
            print(f"[WONJUNS] ❌ Play: {e}")
            raise
        finally:
            self.set_speaking(False)
            stream.stop()
            stream.close()

    async def _relay_phone_audio(self) -> None:
        """Forward phone mic PCM chunks from dashboard queue into the Gemini Live session."""
        q = self._dashboard._phone_audio_queue
        while True:
            try:
                chunk = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # No audio for 1 s → phone mic inactive, give PC mic back
                self._phone_active = False
                continue
            self._phone_active = True   # phone is streaming — silence PC mic
            with self._speaking_lock:
                speaking = self._is_speaking
            if not speaking and not self.ui.muted:
                try:
                    self.out_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    pass

    def _on_phone_connected(self) -> None:
        self.ui.write_log("SYS: Phone connected via Remote Dashboard.")
        self.ui.notify_phone_connected()

    # ── dashboard command relay ─────────────────────────────────────────────

    async def _process_dashboard_commands(self) -> None:
        while True:
            try:
                text = await asyncio.wait_for(
                    self._dashboard._command_queue.get(), timeout=0.5
                )
                if not text:
                    continue
                # Wait up to 8s for session to become ready after a wake
                for _ in range(80):
                    if self.session:
                        break
                    await asyncio.sleep(0.1)
                if self.session:
                    await self.session.send_client_content(
                        turns={"parts": [{"text": text}]},
                        turn_complete=True,
                    )
                    self.ui.write_log(f"[Web]: {text}")
                else:
                    print(f"[Dashboard] Dropped command (no session): {text}")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"[Dashboard] Command error: {e}")
                await asyncio.sleep(0.5)

    # ── main loop ───────────────────────────────────────────────────────────

    async def run(self):
        self._loop = asyncio.get_event_loop()

        # STT 엔진 사전 로드 (첫 웨이크워드 지연 방지)
        try:
            from core.stt_engine import preload_whisper_model
            await asyncio.get_event_loop().run_in_executor(None, preload_whisper_model)
        except Exception:
            pass

        # 웨이크워드 감지기 — 최초 1회 생성
        if self._wake_detector is None:
            self._wake_detector = WakeWordDetector(
                sample_rate=SEND_SAMPLE_RATE,
                on_wake=self._on_wake_word,
            )

        client = genai.Client(
            api_key=_get_api_key(),
            http_options={"api_version": "v1beta"}
        )

        # Start dashboard (optional — needs: pip install fastapi "uvicorn[standard]" cryptography)
        try:
            from dashboard.server import DashboardServer
            self._dashboard = DashboardServer()
            self._dashboard.set_connect_callback(self._on_phone_connected)
            asyncio.create_task(self._dashboard.serve())
            # Runs for the whole lifetime, not just inside an active session
            asyncio.create_task(self._process_dashboard_commands())
        except Exception as e:
            print(f"[Dashboard] Disabled: {e}")
            self._dashboard = None

        while True:
            try:
                print("[WONJUNS] Connecting...")
                self.ui.set_state("THINKING")
                config = self._build_config()

                async with (
                    client.aio.live.connect(model=LIVE_MODEL, config=config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session          = session
                    self.audio_in_queue   = asyncio.Queue()
                    self.out_queue        = asyncio.Queue(maxsize=200)
                    self._turn_done_event = asyncio.Event()

                    print("[WONJUNS] Connected.")
                    self.ui.set_state("LISTENING")
                    self.ui.write_log("SYS: WONJUNS online. — 'Wake up Wonjuns'라고 말해 활성화하세요.")

                    if self._dashboard:
                        await self._dashboard.broadcast({"type": "status", "state": "active"})

                    tg.create_task(self._send_realtime())
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._receive_audio())
                    tg.create_task(self._play_audio())
                    if self._dashboard:
                        tg.create_task(self._relay_phone_audio())

            except Exception as e:
                print(f"[WONJUNS] Error: {e}")
                traceback.print_exc()
            finally:
                self.session = None

            self.set_speaking(False)
            self.ui.set_state("SLEEPING")

            if self._dashboard:
                await self._dashboard.broadcast({"type": "status", "state": "sleeping"})

            print("[WONJUNS] Reconnecting in 3s...")
            await asyncio.sleep(3)

def main():
    ui = AssistantUI("face.png")

    def runner():
        ui.wait_for_api_key()
        assistant = AssistantLive(ui)
        try:
            asyncio.run(assistant.run())
        except KeyboardInterrupt:
            print("\n🔴 Shutting down...")

    threading.Thread(target=runner, daemon=True).start()
    ui.root.mainloop()

if __name__ == "__main__":
    main()