"""
Speech-to-Text wrapper using OpenAI Whisper (local, CPU/GPU).

Public API:
    transcribe(audio_path: str, *, language: str | None = None) -> str

Design notes:
- Model is loaded lazily and cached after the first call.
- The model size is controlled by config (whisper_model_size).
- Returns an empty string (not an exception) when the audio file is silent/empty,
  so the caller can decide how to handle it.
- All transcription happens in-process; no network calls are made.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------

_whisper_model = None
_loaded_model_size: Optional[str] = None


def _get_model(model_size: str):
    """Load and cache the Whisper model. Importing whisper is deferred here."""
    global _whisper_model, _loaded_model_size

    if _whisper_model is not None and _loaded_model_size == model_size:
        return _whisper_model

    try:
        import whisper  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "openai-whisper is not installed. Run: pip install openai-whisper"
        ) from exc

    logger.info("Loading Whisper model '%s' (first call – this may take a moment)…", model_size)
    _whisper_model = whisper.load_model(model_size)
    _loaded_model_size = model_size
    logger.info("Whisper model '%s' loaded.", model_size)
    return _whisper_model


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def transcribe(
    audio_path: str,
    *,
    language: Optional[str] = None,
    model_size: str = "base",
) -> str:
    """
    Transcribe an audio file to text using Whisper.

    Args:
        audio_path:  Path to the audio file (.wav, .mp3, .m4a, .ogg, …).
        language:    Optional BCP-47 language code (e.g. "en", "nl").
                     When None, Whisper auto-detects the language.
        model_size:  Whisper model size to use.
                     Defaults to "base"; override via the `WHISPER_MODEL_SIZE`
                     environment variable or pass explicitly.

    Returns:
        Transcribed text string, stripped of leading/trailing whitespace.
        Returns "" if the file produces no speech output.

    Raises:
        FileNotFoundError: if the audio file does not exist.
        RuntimeError:      if openai-whisper is not installed.
    """
    # Try to read model size from config if available
    try:
        from config import settings  # type: ignore
        effective_model_size = getattr(settings, "whisper_model_size", model_size)
    except Exception:
        effective_model_size = os.environ.get("WHISPER_MODEL_SIZE", model_size)

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path!r}")

    model = _get_model(effective_model_size)

    transcribe_kwargs: dict = {"fp16": False}
    if language:
        transcribe_kwargs["language"] = language

    logger.debug("Transcribing: %s", audio_path)
    result = model.transcribe(audio_path, **transcribe_kwargs)

    text: str = result.get("text", "") or ""
    return text.strip()
