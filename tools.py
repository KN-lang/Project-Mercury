"""
tools.py — Project Mercury
All file-system operations for Mercury.

CRITICAL SAFETY: Every write operation is sandboxed to the output/ directory.
Path traversal attempts are detected and blocked before any I/O occurs.
"""
import logging
import os
import re
from pathlib import Path
from typing import Tuple

from config import OUTPUT_DIR, LANG_EXTENSION_MAP

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised when a path traversal or unsafe filename is detected."""


class ToolExecutor:
    """
    Executes Mercury's file-system tools:
        • create_file  — creates a plaintext file in output/
        • write_code   — saves generated code to a file in output/
        • list_files   — returns names of all files in output/
    """

    def __init__(self) -> None:
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)

    # ── Public tools ───────────────────────────────────────────────────────────

    def create_file(self, filename: str, content: str = "") -> Tuple[bool, str]:
        """Create (or overwrite) a text file in output/."""
        try:
            target = self._safe_path(filename)
            target.write_text(content, encoding="utf-8")
            rel = target.relative_to(self.output_dir.parent)
            return True, (
                f"✅ File created: `{rel}`\n\n"
                f"**Size:** {target.stat().st_size} bytes"
            )
        except SecurityError as e:
            logger.warning(f"SecurityError in create_file: {e}")
            return False, f"🚫 Security violation blocked: {e}"
        except Exception as e:
            logger.error(f"create_file error: {e}", exc_info=True)
            return False, f"❌ Failed to create file: {e}"

    def write_code(self, filename: str, code: str, language: str = "") -> Tuple[bool, str]:
        """Write generated code to a file in output/."""
        try:
            target = self._safe_path(filename)
            target.write_text(code, encoding="utf-8")
            rel   = target.relative_to(self.output_dir.parent)
            lines = code.count("\n") + 1 if code else 0
            return True, (
                f"✅ Code saved: `{rel}`\n\n"
                f"**Language:** {language or _detect_lang(filename)}\n"
                f"**Lines:** {lines}"
            )
        except SecurityError as e:
            logger.warning(f"SecurityError in write_code: {e}")
            return False, f"🚫 Security violation blocked: {e}"
        except Exception as e:
            logger.error(f"write_code error: {e}", exc_info=True)
            return False, f"❌ Failed to write code: {e}"

    def list_files(self) -> list[str]:
        """Return filenames currently in output/ (excluding .gitkeep)."""
        if not self.output_dir.exists():
            return []
        return sorted(
            f.name
            for f in self.output_dir.iterdir()
            if f.is_file() and f.name not in {".gitkeep", ".DS_Store"}
        )

    # ── Safety ────────────────────────────────────────────────────────────────

    def _safe_path(self, filename: str) -> Path:
        """
        Sanitise *filename* and resolve it inside output/.
        Raises SecurityError if the result escapes the sandbox.
        """
        # Strip any directory components the LLM may have hallucinated
        safe_name = Path(filename).name
        if not safe_name:
            raise SecurityError("Empty filename")

        # Remove shell-dangerous characters
        safe_name = re.sub(r'[<>:"|?*\x00-\x1f\\]', "_", safe_name)

        target   = (self.output_dir / safe_name).resolve()
        boundary = self.output_dir.resolve()

        if not str(target).startswith(str(boundary) + os.sep) and target != boundary:
            raise SecurityError(f"Path '{filename}' escapes the output/ sandbox")

        return target


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_lang(filename: str) -> str:
    ext = Path(filename).suffix.lstrip(".").lower()
    for lang, e in LANG_EXTENSION_MAP.items():
        if e == ext:
            return lang.capitalize()
    return ext.upper() or "Unknown"


def lang_to_extension(language: str) -> str:
    """Return the canonical file extension for a language name."""
    return LANG_EXTENSION_MAP.get(language.lower(), "txt")
