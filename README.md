# ☿ Project Mercury

> **Local · Private · Offline** — A voice-controlled AI agent that runs entirely on your Mac. No cloud, no API keys, no data leaving your machine.

---
### Youtube Video
https://youtu.be/nee8HdI8ArI

## Overview

Project Mercury is a **local AI voice agent** built for Apple Silicon (M1/M2). You speak a command, Mercury transcribes it with Whisper, classifies the intent with a tiny local LLM, and executes the action — all in a premium, Apple-inspired web UI.

### What Mercury can do

| Intent | Example | Action |
|--------|---------|--------|
| 💬 **Chat** | *"What is a closure in JavaScript?"* | Conversational answer |
| 📄 **Create file** | *"Create a file called notes.txt"* | Writes `output/notes.txt` |
| 💻 **Write code** | *"Write a Python merge sort"* | Generates & saves code file |
| 📝 **Summarize** | *"Summarize this: …"* | Returns a concise summary |

---

## Architecture

```
Browser (http://localhost:8000)
    │
    │  WAV audio (encoded in-browser, no ffmpeg needed)
    │  JSON for text/intent/execute
    ▼
main.py  ──  FastAPI REST API + static file server
    │
    ├── audio_processor.py  ──  MLX Whisper (Apple Silicon STT)
    ├── agent.py            ──  Ollama (qwen2.5:0.5b intent + LLM)
    ├── tools.py            ──  Sandboxed file ops → output/
    ├── memory.py           ──  In-process session history
    └── config.py           ──  Central config (paths, model IDs)
```

### Key design choices

| Layer | Technology | Why |
|-------|-----------|-----|
| **Web server** | FastAPI + Uvicorn | Fast, async, zero config |
| **Frontend** | Vanilla HTML/CSS/JS | No build step, no framework overhead |
| **STT** | `mlx-whisper` (Apple Silicon) | Uses Neural Engine; fast & low‑RAM |
| **LLM / Intent** | Ollama `qwen2.5:0.5b` | 397 MB; fits in 8 GB unified memory |
| **Audio encoding** | Browser `AudioContext` → WAV | **No ffmpeg required on server** |
| **Privacy** | 100% local | Zero network calls to external APIs |

---

## Audio Pipeline (important fix)

The browser captures microphone audio via `MediaRecorder` (produces WebM/Opus). MLX Whisper's `load_audio()` normally calls the system `ffmpeg` binary to decode this — but `ffmpeg` is not installed.

**Fix**: Audio is re-encoded to **PCM WAV (16 kHz, mono)** entirely in the browser using `AudioContext.decodeAudioData()` before being uploaded. The server receives a clean WAV file that Whisper reads natively, with no external dependencies.

```
Mic → MediaRecorder (WebM/Opus)
         │
         ▼  [browser: AudioContext.decodeAudioData → WAV encoder]
      16-bit PCM WAV, 16 kHz, mono
         │
         ▼  POST /api/transcribe
      mlx_whisper.transcribe(wav_path) → text ✓
```

---

## Prerequisites

| Requirement | Install |
|-------------|---------|
| Python 3.11+ | `brew install python` or system Python |
| Ollama | Download from [ollama.com](https://ollama.com) |
| `qwen2.5:0.5b` model | `ollama pull qwen2.5:0.5b` |
| Internet (first run only) | MLX Whisper downloads the model on first use |

> **No ffmpeg required.** Audio conversion happens in the browser.

---

## Quick Start

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd Project-Mercury

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Ollama (in a separate terminal)
/Applications/Ollama.app/Contents/Resources/ollama serve

# 5. Pull the model (first time only)
/Applications/Ollama.app/Contents/Resources/ollama pull qwen2.5:0.5b

# 6. Start Mercury
python main.py
```

Then open **http://localhost:8000** in your browser.

> **Tip**: Add Ollama to your PATH for convenience:  
> `export PATH="/Applications/Ollama.app/Contents/Resources:$PATH"`

---

## Environment Variables

Copy `.env.example` to `.env` and edit as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen2.5:0.5b` | Ollama model for intent + generation |
| `WHISPER_MODEL_REPO` | `mlx-community/whisper-base.en-mlx` | MLX Whisper model |
| `WHISPER_MODEL_FALLBACK` | `base.en` | openai-whisper fallback model |
| `OUTPUT_DIR` | `./output` | Sandboxed directory for all file operations |

---

## UI Modes

The floating pill navbar at the top switches between four modes:

### 🟢 Basic Mode (default)
Clean chat-like interface. Speak or type your command. After voice input:
- The chat bubble shows **"🎙️ Transcribing…"** with animated dots while Whisper processes
- Once complete, the bubble updates to **"🎙️ Heard: [your words]"** so you can confirm what was captured
- Mercury's response appears below

### ⚡ Pipeline Mode
Real-time step-by-step visualization:
```
[Audio Input] → [Transcription] → [Intent Detection] → [Action] → [Output]
```
Each step updates live with status indicators (loading / success / error).

### 🖥️ System
Live status of Ollama, Whisper backend, and the output sandbox path.

### 📁 Files
Grid of all files Mercury has created. Each card has Preview and Download buttons.

---

## Human-in-the-Loop (HITL)

For **destructive** intents (`CREATE_FILE`, `WRITE_CODE`), Mercury pauses before executing and shows a confirmation modal:

```
Do you want to execute this action?
Intent: WRITE_CODE
Request: "Write a Python sorting script"
File: mercury_code.py

  [ ✓ Approve ]   [ Cancel ]
```

Only after you click **Approve** does Mercury write to disk.

---

## Safety & Security

- **Sandboxed output**: All file operations are strictly restricted to the `output/` directory. Path traversal attempts (`../`, absolute paths) are blocked by `tools.py`.
- **No network calls**: The application never makes requests to external services.
- **No persistent server-side storage**: Session history exists only in memory while the server runs.

---

## File Structure

```
Project-Mercury/
├── main.py              # FastAPI server + REST API endpoints
├── agent.py             # Ollama intent classification + LLM generation
├── audio_processor.py   # MLX Whisper STT (Apple Silicon optimized)
├── tools.py             # Sandboxed file operations
├── memory.py            # Session state management
├── config.py            # Central configuration (paths, model IDs)
├── static/
│   └── index.html       # Complete web UI (HTML + CSS + JS, single file)
├── output/              # All Mercury-generated files land here
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── app.py               # Legacy Streamlit UI (kept for reference, not used)
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the web UI |
| `/api/transcribe` | POST | Accepts WAV audio file, returns transcription text |
| `/api/classify` | POST | Classifies intent from transcription text |
| `/api/execute` | POST | Runs the action (chat / file / code / summarize) |
| `/api/files` | GET | Lists all files in `output/` |
| `/api/files/{name}/preview` | GET | Returns file content as JSON |
| `/api/files/{name}/download` | GET | Downloads the file |
| `/api/status` | GET | Ollama + Whisper health check |
| `/api/history` | GET | Returns session conversation history |

---

## Troubleshooting

### Voice recording not working
- Ensure the browser has **microphone permission** (look for the mic icon in the address bar)
- The mic button turns **red** while recording — click again to stop
- After stopping, the button briefly shows "🔄 Encoding audio…" — this is normal

### Ollama not connected (System panel shows offline)
```bash
# Start Ollama
/Applications/Ollama.app/Contents/Resources/ollama serve

# Verify it's running
/Applications/Ollama.app/Contents/Resources/ollama list
```

### MLX Whisper slow on first use
The model (`whisper-base.en-mlx`, ~150 MB) is downloaded from HuggingFace on the first transcription request. Subsequent requests use the cached model and are fast.

### "No speech detected" error
- Speak clearly and closely to the mic
- Ensure you record for at least 1–2 seconds
- Check that your Mac's input device is set to the correct microphone in **System Settings → Sound**

---

## Dependencies

```
fastapi          # REST API framework
uvicorn          # ASGI server
python-multipart # Multipart form data (audio upload)
mlx-whisper      # Apple Silicon STT (primary)
openai-whisper   # CPU fallback STT
ollama           # Local LLM client
pydub            # Audio utilities
soundfile        # Audio file I/O
python-dotenv    # .env file loading
```

---

## License

MIT — built as an internship assignment demonstrating local AI agent architecture on Apple Silicon.