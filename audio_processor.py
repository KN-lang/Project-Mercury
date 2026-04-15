"""
audio_processor.py — Project Mercury
Speech-to-Text using MLX Whisper (Apple Silicon / MPS primary)
with a transparent fallback to openai-whisper on CPU.
"""
import io
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Backend detection ─────────────────────────────────────────────────────────
try:
    import mlx_whisper  # type: ignore
    _MLX_OK = True
    logger.info("AudioProcessor: MLX Whisper backend loaded ✓ (Apple Silicon)")
except ImportError:
    _MLX_OK = False
    logger.warning("AudioProcessor: mlx-whisper unavailable, trying openai-whisper …")

_OW_MODEL = None
if not _MLX_OK:
    try:
        import whisper as _ow  # type: ignore
        _OW_MODEL_NAME = None   # lazy-load on first use
        _OW_AVAILABLE  = True
        logger.info("AudioProcessor: openai-whisper backend available ✓")
    except ImportError:
        _OW_AVAILABLE = False
        logger.error("AudioProcessor: NO Whisper backend found! STT will fail.")
else:
    _OW_AVAILABLE = False

from config import WHISPER_MODEL_REPO, WHISPER_MODEL_FALLBACK


class AudioProcessor:
    """
    Wraps Whisper STT in a backend-agnostic interface.

    Accepts audio as:
        • bytes / bytearray
        • file-like object (BytesIO, UploadedFile, etc.)
        • str or Path (local filesystem path)
    """

    def __init__(self) -> None:
        self.use_mlx = _MLX_OK
        self._ow_model = None

        if not self.use_mlx:
            if not _OW_AVAILABLE:
                raise RuntimeError(
                    "No Whisper backend is available.\n"
                    "Run: pip install mlx-whisper   (Apple Silicon)\n"
                    "  or pip install openai-whisper (CPU fallback)"
                )
            logger.info(f"Loading openai-whisper model '{WHISPER_MODEL_FALLBACK}' …")
            import whisper as _ow
            self._ow_model = _ow.load_model(WHISPER_MODEL_FALLBACK)

    @property
    def backend_name(self) -> str:
        return "MLX Whisper (Apple Silicon)" if self.use_mlx else "OpenAI Whisper (CPU)"

    # ── Public API ─────────────────────────────────────────────────────────────

    def transcribe(self, source) -> str:
        """
        Transcribe audio from various source types.
        Returns the transcribed string (stripped) or "" on silence/failure.
        """
        if isinstance(source, (bytes, bytearray)):
            return self._from_bytes(bytes(source))
        if hasattr(source, "read"):          # BytesIO, UploadedFile …
            return self._from_bytes(source.read())
        if isinstance(source, (str, Path)):
            return self._from_file(str(source))
        raise TypeError(f"AudioProcessor.transcribe: unsupported source type {type(source)}")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _from_bytes(self, audio_bytes: bytes) -> str:
        """Save bytes to a temp WAV file, transcribe, then delete."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="mercury_audio_") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            return self._from_file(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _from_file(self, file_path: str) -> str:
        """Transcribe an audio file; return clean text or ""."""
        try:
            if self.use_mlx:
                result = mlx_whisper.transcribe(
                    file_path,
                    path_or_hf_repo=WHISPER_MODEL_REPO,
                    verbose=False,
                    language="en",
                )
            else:
                result = self._ow_model.transcribe(file_path, language="en")

            text = result.get("text", "").strip()
            return text

        except Exception as exc:
            logger.error(f"Transcription failed for '{file_path}': {exc}", exc_info=True)
            return ""
