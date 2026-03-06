"""
Text-to-Speech for the Adversarial Presentation Agent.

Backend selection (tried in order):
  1. macOS   — `say` command (built-in, zero deps)
  2. Windows — `powershell` SAPI via PowerShell (built-in, zero deps)
  3. Linux   — `espeak` or `espeak-ng` (common on Ubuntu/Debian; apt install espeak)
  4. pyttsx3 — cross-platform Python wrapper (pip install pyttsx3)

All backends are tried silently. If none is available the call is a no-op.

Public API:
    speak(text: str) -> None   — speaks text, blocks until done
    tts_available() -> bool    — True if any TTS backend is found
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess

logger = logging.getLogger(__name__)

_SYSTEM = platform.system()  # "Darwin" | "Windows" | "Linux"


# ---------------------------------------------------------------------------
# Platform-specific backends
# ---------------------------------------------------------------------------

def _speak_macos(text: str) -> None:
    """macOS built-in `say` command."""
    subprocess.run(
        ["say", text],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _speak_windows(text: str) -> None:
    """Windows built-in SAPI via PowerShell — no extra installs needed."""
    # Escape single quotes so the inline PS string is safe
    safe = text.replace("'", "''")
    script = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')"
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _speak_espeak(text: str) -> None:
    """Linux espeak / espeak-ng (install: sudo apt install espeak-ng)."""
    cmd = "espeak-ng" if shutil.which("espeak-ng") else "espeak"
    subprocess.run(
        [cmd, text],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _speak_pyttsx3(text: str) -> None:
    """Cross-platform Python TTS wrapper (pip install pyttsx3)."""
    import pyttsx3  # type: ignore
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    engine.say(text)
    engine.runAndWait()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tts_available() -> bool:
    """Return True if any TTS backend is available on this platform."""
    if _SYSTEM == "Darwin" and shutil.which("say"):
        return True
    if _SYSTEM == "Windows" and shutil.which("powershell"):
        return True
    if _SYSTEM == "Linux" and (shutil.which("espeak-ng") or shutil.which("espeak")):
        return True
    try:
        import pyttsx3  # noqa: F401
        return True
    except ImportError:
        pass
    return False


def speak(text: str) -> None:
    """
    Speak text aloud using the best available backend.
    Blocks until speech finishes. Silently skips if no backend is available.
    """
    if not text or not text.strip():
        return

    # 1. macOS
    if _SYSTEM == "Darwin" and shutil.which("say"):
        try:
            _speak_macos(text)
            return
        except Exception as e:
            logger.warning("macOS say failed: %s", e)

    # 2. Windows PowerShell SAPI
    if _SYSTEM == "Windows" and shutil.which("powershell"):
        try:
            _speak_windows(text)
            return
        except Exception as e:
            logger.warning("Windows SAPI failed: %s", e)

    # 3. Linux espeak / espeak-ng
    if _SYSTEM == "Linux" and (shutil.which("espeak-ng") or shutil.which("espeak")):
        try:
            _speak_espeak(text)
            return
        except Exception as e:
            logger.warning("espeak failed: %s", e)

    # 4. pyttsx3 (any platform, pip install pyttsx3)
    try:
        _speak_pyttsx3(text)
        return
    except ImportError:
        pass
    except Exception as e:
        logger.warning("pyttsx3 failed: %s", e)

    logger.info("TTS unavailable — would have spoken: %s", text[:80])
