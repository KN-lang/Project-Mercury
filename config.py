"""
config.py — Project Mercury
Central configuration: paths, model identifiers, and environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()

# CRITICAL: All file operations are sandboxed to this directory.
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Speech-to-Text (Whisper on Apple Silicon via MLX) ─────────────────────────
# Primary: MLX-optimized Whisper (Apple Neural Engine / MPS backend)
# Downloads automatically from HuggingFace Hub on first run (~75MB for base.en)
WHISPER_MODEL_REPO = os.getenv("WHISPER_MODEL_REPO", "mlx-community/whisper-base.en-mlx")

# Fallback: openai-whisper model name (used if mlx-whisper is unavailable)
WHISPER_MODEL_FALLBACK = os.getenv("WHISPER_MODEL_FALLBACK", "base.en")

# ── Ollama LLM ────────────────────────────────────────────────────────────────
# qwen2.5:0.5b is ~390MB, fast on M1, and instruction-following capable.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")

# ── Application ───────────────────────────────────────────────────────────────
APP_NAME    = "Mercury"
APP_VERSION = "1.0.0"
APP_EMOJI   = "☿"

# ── Intent Registry ───────────────────────────────────────────────────────────
VALID_INTENTS = {"CREATE_FILE", "WRITE_CODE", "SUMMARIZE", "CHAT"}

# Intents that require human-in-the-loop confirmation before execution
CONFIRMATION_REQUIRED = {"CREATE_FILE", "WRITE_CODE"}

# Map programming language → file extension
LANG_EXTENSION_MAP: dict[str, str] = {
    "python":     "py",
    "javascript": "js",
    "typescript": "ts",
    "java":       "java",
    "c":          "c",
    "cpp":        "cpp",
    "rust":       "rs",
    "go":         "go",
    "shell":      "sh",
    "bash":       "sh",
    "html":       "html",
    "css":        "css",
    "sql":        "sql",
    "markdown":   "md",
}
