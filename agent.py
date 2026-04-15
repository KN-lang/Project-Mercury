"""
agent.py — Project Mercury
Intent classification and LLM-powered response generation via Ollama.

Intent categories:
    CREATE_FILE  — create a text or data file
    WRITE_CODE   — generate and save source code
    SUMMARIZE    — condense a body of text
    CHAT         — general conversation / fallback
"""
import json
import logging
import re
from typing import Optional

import ollama

from config import OLLAMA_MODEL, OLLAMA_HOST, VALID_INTENTS, LANG_EXTENSION_MAP

logger = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────────────

_INTENT_SYSTEM = """\
You are Mercury, a precise AI classification engine.
Classify the user's request into EXACTLY one of these intents:

  CREATE_FILE — user wants to create or write a text/data file
  WRITE_CODE  — user wants to generate and save source code
  SUMMARIZE   — user wants a summary of some text
  CHAT        — general conversation, greetings, questions

Reply with ONLY valid JSON in this exact schema. No prose, no markdown fences.
{
  "intent":     "INTENT_NAME",
  "filename":   "suggested_filename_or_null",
  "language":   "programming_language_or_null",
  "content":    "literal_file_content_or_null",
  "task":       "what_the_code_should_do_or_null",
  "text":       "text_to_summarise_or_null",
  "confidence": 0.95
}"""

_CHAT_SYSTEM = """\
You are Mercury, an intelligent and friendly AI voice agent.
Be concise, helpful, and warm. Always refer to yourself as Mercury.
If you don't know something, say so honestly."""

_SUMMARIZE_SYSTEM = """\
You are Mercury. Produce a clear, well-structured summary of the text provided.
Use bullet points for key facts. Be thorough yet concise."""

_CODE_SYSTEM = """\
You are Mercury, an expert software engineer.
Write clean, working, well-commented code for the given task.
Output ONLY the raw code — no markdown fences, no explanation."""


class MercuryAgent:
    """Connects to the local Ollama server and drives Mercury's reasoning."""

    def __init__(self) -> None:
        self.model = OLLAMA_MODEL
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Raise ConnectionError if Ollama is not reachable."""
        try:
            ollama.list()
        except Exception as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {OLLAMA_HOST}.\n"
                "Make sure Ollama is running: `ollama serve`\n"
                f"Detail: {exc}"
            ) from exc

    # ── Public API ─────────────────────────────────────────────────────────────

    def classify_intent(self, transcription: str) -> dict:
        """
        Ask the LLM to classify *transcription* and return a structured dict:
            {intent, filename, language, content, task, text, confidence}
        Falls back gracefully on JSON parse errors or network issues.
        """
        try:
            resp = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system",  "content": _INTENT_SYSTEM},
                    {"role": "user",    "content": f'Classify: "{transcription}"'},
                ],
                options={"temperature": 0.05},  # very low for deterministic classification
            )
            raw = resp["message"]["content"]
            return self._parse_intent(raw, transcription)
        except Exception as exc:
            logger.error(f"classify_intent error: {exc}", exc_info=True)
            return _fallback_intent(transcription)

    def chat(self, message: str, history: list[dict] | None = None) -> str:
        """Generate a conversational reply, optionally with recent history."""
        messages = [{"role": "system", "content": _CHAT_SYSTEM}]

        # Include last 5 CHAT exchanges for short-term memory
        if history:
            for entry in history[-10:]:
                if entry.get("intent") == "CHAT":
                    if t := entry.get("transcription"):
                        messages.append({"role": "user",      "content": t})
                    if o := entry.get("output"):
                        messages.append({"role": "assistant", "content": o})

        messages.append({"role": "user", "content": message})

        resp = ollama.chat(model=self.model, messages=messages)
        return resp["message"]["content"].strip()

    def summarize(self, text: str) -> str:
        """Return an LLM-generated summary of *text*."""
        resp = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": _SUMMARIZE_SYSTEM},
                {"role": "user",   "content": f"Summarize this:\n\n{text}"},
            ],
            options={"temperature": 0.3},
        )
        return resp["message"]["content"].strip()

    def generate_code(self, task: str, language: str = "python") -> str:
        """Ask the LLM to write code for *task* in *language*."""
        resp = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": _CODE_SYSTEM},
                {"role": "user",   "content": f"Write {language} code to: {task}"},
            ],
            options={"temperature": 0.2},
        )
        code = resp["message"]["content"].strip()

        # Strip accidental markdown fences if the model forgets instructions
        code = re.sub(r"^```[\w]*\n?", "", code)
        code = re.sub(r"\n?```$", "", code)
        return code.strip()

    # ── Intent parsing ────────────────────────────────────────────────────────

    def _parse_intent(self, raw: str, original: str) -> dict:
        """
        Robustly extract intent JSON from the LLM's response.
        Strategy:
          1. Direct json.loads()
          2. Regex-extract the first {...} block
          3. Keyword-matching fallback
        """
        # 1. Direct parse
        try:
            data = json.loads(raw.strip())
            if "intent" in data and data["intent"] in VALID_INTENTS:
                return data
        except json.JSONDecodeError:
            pass

        # 2. Extract first JSON object from surrounding text
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                if "intent" in data and data["intent"] in VALID_INTENTS:
                    return data
            except json.JSONDecodeError:
                pass

        # 3. Keyword fallback on both the LLM output and original transcript
        combined = (raw + " " + original).lower()
        if any(kw in combined for kw in ["create_file", "create file", "new file", "make a file", "write a file"]):
            return {"intent": "CREATE_FILE", "confidence": 0.6, "filename": None, "content": ""}
        if any(kw in combined for kw in ["write_code", "write code", "python script", "javascript", "save code", "program"]):
            return {"intent": "WRITE_CODE", "confidence": 0.6, "language": "python", "task": original}
        if any(kw in combined for kw in ["summarize", "summarise", "summary", "tldr"]):
            return {"intent": "SUMMARIZE", "confidence": 0.6, "text": original}

        return _fallback_intent(original)


# ── Module-level helpers ──────────────────────────────────────────────────────

def _fallback_intent(transcription: str) -> dict:
    """Default to CHAT so Mercury always responds gracefully."""
    return {
        "intent":     "CHAT",
        "confidence": 0.5,
        "filename":   None,
        "language":   None,
        "content":    None,
        "task":       None,
        "text":       None,
    }
