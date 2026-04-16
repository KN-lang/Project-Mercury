"""
main.py — Project Mercury
FastAPI backend: wraps existing audio/agent/tool modules and serves the web UI.
"""
import io
import logging
from functools import lru_cache
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import (
    APP_VERSION, CONFIRMATION_REQUIRED, OLLAMA_MODEL,
    OUTPUT_DIR, WHISPER_MODEL_REPO,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Project Mercury", version=APP_VERSION)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── In-memory session history ─────────────────────────────────────────────────
_sessions: dict[str, list] = {}


# ── Cached singletons ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _audio_processor():
    from audio_processor import AudioProcessor
    return AudioProcessor()


@lru_cache(maxsize=1)
def _agent():
    from agent import MercuryAgent
    return MercuryAgent()


@lru_cache(maxsize=1)
def _executor():
    from tools import ToolExecutor
    return ToolExecutor()


# ── HTML Entry Point ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html = STATIC_DIR / "index.html"
    if html.exists():
        return HTMLResponse(html.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Mercury: index.html missing</h1>", status_code=404)


# ── API: Transcribe ───────────────────────────────────────────────────────────

@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        content = await audio.read()
        # Derive format hint from MIME type (browser sends audio/webm)
        mime = audio.content_type or "audio/webm"
        fmt = mime.split("/")[-1].split(";")[0].strip()
        if fmt in {"mpeg", "mp4"}:
            fmt = "mp3"
        processor = _audio_processor()
        logger.info(f"Transcribing audio: {len(content)} bytes, format: {fmt}, mine: {mime}")
        text = processor.transcribe(content, fmt=fmt)
        if not text:
            logger.warning("Transcription returned empty text.")
            return JSONResponse({"success": False, "transcription": "",
                                 "error": "No speech detected. Please try again."})
        logger.info(f"Transcription successful: '{text[:50]}...'")
        return {"success": True, "transcription": text}
    except Exception as exc:
        logger.error("Transcribe error", exc_info=True)
        raise HTTPException(500, str(exc))


# ── API: Classify Intent ──────────────────────────────────────────────────────

class ClassifyReq(BaseModel):
    transcription: str
    session_id: str = "default"


@app.post("/api/classify")
async def classify(req: ClassifyReq):
    try:
        result = _agent().classify_intent(req.transcription)
        intent = result.get("intent", "CHAT")
        return {
            "success": True,
            "intent": intent,
            "confidence": result.get("confidence", 0.5),
            "requires_confirmation": intent in CONFIRMATION_REQUIRED,
            "parameters": {
                "filename": result.get("filename"),
                "language": result.get("language"),
                "content":  result.get("content"),
                "task":     result.get("task"),
                "text":     result.get("text"),
            },
        }
    except Exception as exc:
        logger.error("Classify error", exc_info=True)
        raise HTTPException(500, str(exc))


# ── API: Execute Action ───────────────────────────────────────────────────────

class ExecuteReq(BaseModel):
    intent: str
    transcription: str
    parameters: dict = {}
    session_id: str = "default"


@app.post("/api/execute")
async def execute(req: ExecuteReq):
    try:
        from tools import lang_to_extension
        agent    = _agent()
        executor = _executor()
        history  = _sessions.get(req.session_id, [])
        params   = req.parameters
        intent   = req.intent
        trans    = req.transcription
        output   = ""
        success  = True
        extras   = {}

        if intent == "CHAT":
            output = agent.chat(trans, history)

        elif intent == "SUMMARIZE":
            output = agent.summarize(params.get("text") or trans)

        elif intent == "CREATE_FILE":
            fname = params.get("filename") or "mercury_note.txt"
            success, output = executor.create_file(fname, params.get("content") or "")

        elif intent == "WRITE_CODE":
            lang  = (params.get("language") or "python").lower()
            task  = params.get("task") or params.get("content") or trans
            fname = params.get("filename") or f"mercury_code.{lang_to_extension(lang)}"
            code  = agent.generate_code(task, lang)
            success, output = executor.write_code(fname, code, lang)
            extras["code_preview"] = code
            extras["language"]     = lang

        else:
            output  = "Mercury couldn't map that request to an action. Please rephrase."
            success = False

        entry = {"intent": intent, "transcription": trans,
                 "output": output, "success": success}
        _sessions.setdefault(req.session_id, []).append(entry)

        return {"success": success, "output": output, **extras}

    except Exception as exc:
        logger.error("Execute error", exc_info=True)
        raise HTTPException(500, str(exc))


# ── API: Files ────────────────────────────────────────────────────────────────

CODE_EXTS = {"py","js","ts","rs","go","cpp","c","java","sh","html","css","sql"}

@app.get("/api/files")
async def list_files():
    executor = _executor()
    files = []
    for name in executor.list_files():
        p   = OUTPUT_DIR / name
        ext = Path(name).suffix.lstrip(".").lower()
        files.append({
            "name": name,
            "size": p.stat().st_size if p.exists() else 0,
            "type": "code" if ext in CODE_EXTS else "text",
            "ext":  ext,
        })
    return {"files": files}


@app.get("/api/files/{filename}/preview")
async def preview_file(filename: str):
    p = (OUTPUT_DIR / Path(filename).name).resolve()
    if not str(p).startswith(str(OUTPUT_DIR.resolve())):
        raise HTTPException(403, "Access denied")
    if not p.exists():
        raise HTTPException(404, "File not found")
    return {"name": filename, "content": p.read_text(encoding="utf-8", errors="replace")}


@app.get("/api/files/{filename}/download")
async def download_file(filename: str):
    p = (OUTPUT_DIR / Path(filename).name).resolve()
    if not str(p).startswith(str(OUTPUT_DIR.resolve())):
        raise HTTPException(403, "Access denied")
    if not p.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(p), filename=filename)


# ── API: System Status ────────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    ollama_ok = False
    try:
        import ollama as _ol; _ol.list(); ollama_ok = True
    except Exception:
        pass

    whisper_be = "Not installed"
    try:
        import mlx_whisper; whisper_be = "MLX (Apple Silicon)"  # noqa
    except ImportError:
        try:
            import whisper; whisper_be = "CPU (OpenAI Whisper)"  # noqa
        except ImportError:
            pass

    return {
        "ollama":  {"connected": ollama_ok, "model": OLLAMA_MODEL},
        "whisper": {"backend": whisper_be,  "model": WHISPER_MODEL_REPO},
        "output_dir": str(OUTPUT_DIR),
        "version": APP_VERSION,
    }


# ── API: History ──────────────────────────────────────────────────────────────

@app.get("/api/history")
async def get_history(session_id: str = "default"):
    return {"history": _sessions.get(session_id, [])}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
