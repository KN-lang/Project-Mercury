"""
audio_processor.py — Project Mercury
Speech-to-Text using MLX Whisper (Apple Silicon / MPS primary)
with a transparent fallback to openai-whisper on CPU.

KEY FIX: mlx_whisper's file loader internally calls ffmpeg, which may not
be installed. We side-step this by decoding the WAV with soundfile/numpy
and feeding a float32 numpy array directly to mlx_whisper.transcribe().
This code path NEVER touches ffmpeg.
"""
import io
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

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

MLX_WHISPER_SAMPLE_RATE = 16_000  # Whisper always expects 16 kHz


class AudioProcessor:
    """
    Wraps Whisper STT in a backend-agnostic interface.

    Accepts audio as:
        • bytes / bytearray  (WAV bytes — sent from the browser)
        • file-like object (BytesIO, UploadedFile, etc.)
        • str or Path (local filesystem path)

    The browser already encodes audio to 16 kHz mono WAV via blobToWav(),
    so we read it with soundfile → numpy and pass the float32 array directly
    to mlx_whisper, completely bypassing any ffmpeg dependency.
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

    def transcribe(self, source, fmt: str = "wav") -> str:
        """
        Transcribe audio from various source types.
        fmt: audio format hint (wav/webm/mp3/ogg) — only used for temp-file
             naming when falling back to the file-based openai-whisper path.
        Returns the transcribed string (stripped) or "" on silence/failure.
        """
        if isinstance(source, (bytes, bytearray)):
            return self._from_bytes(bytes(source), fmt=fmt)
        if hasattr(source, "read"):          # BytesIO, UploadedFile …
            return self._from_bytes(source.read(), fmt=fmt)
        if isinstance(source, (str, Path)):
            return self._from_file(str(source))
        raise TypeError(f"AudioProcessor.transcribe: unsupported source type {type(source)}")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _wav_bytes_to_numpy(self, audio_bytes: bytes) -> np.ndarray:
        """
        Decode WAV bytes → float32 numpy array at 16 kHz mono.
        Uses soundfile (libsndfile), which requires NO external binaries.
        """
        buf = io.BytesIO(audio_bytes)
        data, sample_rate = sf.read(buf, dtype="float32", always_2d=False)

        # Mix down to mono if stereo
        if data.ndim == 2:
            data = data.mean(axis=1)

        # Resample to 16 kHz if necessary (simple linear interpolation)
        if sample_rate != MLX_WHISPER_SAMPLE_RATE:
            target_len = int(len(data) * MLX_WHISPER_SAMPLE_RATE / sample_rate)
            data = np.interp(
                np.linspace(0, len(data) - 1, target_len),
                np.arange(len(data)),
                data,
            ).astype(np.float32)

        logger.debug(
            f"Decoded WAV: {len(data)} frames @ {MLX_WHISPER_SAMPLE_RATE} Hz "
            f"({len(data)/MLX_WHISPER_SAMPLE_RATE:.2f}s)"
        )
        return data

    def _from_bytes(self, audio_bytes: bytes, fmt: str = "wav") -> str:
        """
        For MLX backend: decode WAV → numpy, pass array directly (no ffmpeg).
        For openai-whisper fallback: write to a temp file (openai-whisper can
        read WAV natively without ffmpeg for WAV files).
        """
        if self.use_mlx:
            try:
                audio_np = self._wav_bytes_to_numpy(audio_bytes)
                return self._transcribe_numpy(audio_np)
            except Exception as exc:
                logger.error(f"MLX numpy-path transcription failed: {exc}", exc_info=True)
                return ""
        else:
            # openai-whisper can read WAV without ffmpeg
            safe_fmt = fmt.split(";")[0].strip().lower()
            if safe_fmt in ("mpeg", "mp4"):
                safe_fmt = "mp3"
            suffix = f".{safe_fmt}" if safe_fmt else ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="mercury_audio_") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                return self._from_file(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _transcribe_numpy(self, audio_np: np.ndarray) -> str:
        """Pass a float32 numpy array directly to mlx_whisper — zero ffmpeg calls."""
        logger.debug(f"Transcribing numpy array: {audio_np.shape}, dtype={audio_np.dtype}")
        try:
            result = mlx_whisper.transcribe(
                audio_np,                        # ← numpy array, not a file path
                path_or_hf_repo=WHISPER_MODEL_REPO,
                verbose=False,
                language="en",
            )
            text = result.get("text", "").strip()
            logger.info(f"MLX transcription result: '{text[:80]}'")
            return text
        except Exception as exc:
            logger.error(f"mlx_whisper.transcribe (numpy) failed: {exc}", exc_info=True)
            return ""

    def _from_file(self, file_path: str) -> str:
        """Transcribe an audio file; return clean text or ""."""
        try:
            if self.use_mlx:
                # Read file → numpy to avoid ffmpeg dependency
                with open(file_path, "rb") as f:
                    audio_bytes = f.read()
                audio_np = self._wav_bytes_to_numpy(audio_bytes)
                return self._transcribe_numpy(audio_np)
            else:
                logger.debug(f"Transcribing '{file_path}' using OpenAI Whisper backend...")
                result = self._ow_model.transcribe(file_path, language="en")
                text = result.get("text", "").strip()
                return text

        except Exception as exc:
            logger.error(f"Transcription failed for '{file_path}': {exc}", exc_info=True)
            return ""
