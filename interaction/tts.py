"""
Text-to-Speech for the Adversarial Presentation Agent.

Primary:  macOS `say` command (zero dependencies, built-in on every Mac)
Fallback: pyttsx3 (cross-platform, pip install pyttsx3)

Public API:
    speak(text: str) -> None   — speaks text, blocks until done
    tts_available() -> bool    — True if any TTS backend is found
"""

from __future__ import annotations

import logging
import os
import subprocess
import shutil

logger = logging.getLogger(__name__)

# macOS voice — change to any voice from `say -v ?`
_MACOS_VOICE = "Samantha"


def _speak_macos(text: str) -> None:
    """Speak using macOS built-in `say` command."""
    subprocess.run(
        ["say", "-v", _MACOS_VOICE, text],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _speak_pyttsx3(text: str) -> None:
    """Speak using pyttsx3 (cross-platform fallback)."""
    import pyttsx3  # type: ignore
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    engine.say(text)
    engine.runAndWait()


def tts_available() -> bool:
    """Return True if any TTS backend is available."""
    if shutil.which("say"):
        return True
    try:
        import pyttsx3  # noqa: F401
        return True
    except ImportError:
        return False


def speak(text: str) -> None:
    """
    Speak text aloud. Blocks until speech finishes.
    Silently skips if no TTS backend is available.
    """
    if not text or not text.strip():
        return

    # Try macOS say first
    if shutil.which("say"):
        try:
            _speak_macos(text)
            return
        except Exception as e:
            logger.warning("macOS say failed: %s", e)

    # Try pyttsx3
    try:
        _speak_pyttsx3(text)
        return
    except ImportError:
        pass
    except Exception as e:
        logger.warning("pyttsx3 failed: %s", e)

    # Silent fallback — just log
    logger.info("TTS unavailable — would have spoken: %s", text[:80])
