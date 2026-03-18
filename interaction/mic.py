"""
Microphone recording for the Adversarial Presentation Agent.

Records audio from the default input device until the user presses Enter.
Writes a 16 kHz mono WAV using only stdlib `wave` (no scipy needed).

Works on macOS, Windows, and Linux without OS-specific code.
On Linux, sounddevice requires PortAudio: sudo apt install libportaudio2

Public API:
    mic_available() -> bool       — True if sounddevice + numpy are importable
    request_permission() -> bool  — warm-up open to trigger OS permission prompts
    record(...)          -> str | None
"""

from __future__ import annotations

import cmd
import logging
import os
import platform
import tempfile
import threading
import time
import wave

logger = logging.getLogger(__name__)

_SYSTEM = platform.system()  # "Darwin" | "Windows" | "Linux"


def mic_available() -> bool:
    """Return True if recording dependencies are installed."""
    try:
        import sounddevice  # noqa: F401
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def request_permission(sample_rate: int = 16000) -> bool:
    """
    Open the mic briefly to trigger OS-level permission prompts (macOS, Windows).
    Returns True if the stream opened successfully (permission granted or not required).
    Safe to call on all platforms; is a no-op on Linux where no prompt is needed.
    """
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        return False

    try:
        frames = []
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
            for _ in range(5):
                chunk, _ = stream.read(int(sample_rate * 0.1))
                frames.append(chunk.copy())
                time.sleep(0.1)
        return True
    except Exception as e:
        logger.warning("Mic permission check failed: %s", e)
        return False


def _mic_troubleshooting_hint(rms: float) -> str:
    """Return a platform-appropriate hint for low-signal mic issues."""
    if _SYSTEM == "Darwin":
        return (
            f"  ⚠  Signal too low (RMS={rms:.5f}).\n"
            "     macOS: System Settings → Privacy & Security → Microphone → enable Terminal/iTerm\n"
            "     Then check input device: python3 -c \"import sounddevice; print(sounddevice.query_devices())\""
        )
    if _SYSTEM == "Windows":
        return (
            f"  ⚠  Signal too low (RMS={rms:.5f}).\n"
            "     Windows: Settings → System → Sound → Input → choose your microphone\n"
            "     Allow microphone access: Settings → Privacy → Microphone → enable for this app\n"
            "     Then check input device: python -c \"import sounddevice; print(sounddevice.query_devices())\""
        )
    # Linux
    return (
        f"  ⚠  Signal too low (RMS={rms:.5f}).\n"
        "     Linux: check PulseAudio/PipeWire input is not muted (pavucontrol)\n"
        "     Then check input device: python3 -c \"import sounddevice; print(sounddevice.query_devices())\"\n"
        "     If sounddevice fails to open: sudo apt install libportaudio2"
    )


def record(
    prompt: str = "",
    sample_rate: int = 16000,
    min_duration_s: float = 0.5,
    silence_threshold: float = 0.0005,
) -> str | None:
    """
    Record from the microphone until the user presses Enter.

    Returns:
        Path to a temp WAV file  — audio was captured
        "/end"                   — user typed /end before recording
        None                     — capture was silent / too short
    """
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        logger.error(
            "sounddevice not installed.\n"
            "  Install: pip install sounddevice numpy\n"
            "  Linux also needs: sudo apt install libportaudio2"
        )
        return None

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if prompt:
        print(prompt, flush=True)
    cmd = input().strip().lower()
    if cmd in {"/end", "end", "quit", "exit"}:
        return "/end"
    if cmd in {"/reset", "reset", "new", "new_user"}:
        return "/reset"

    print("  ◉  Recording…  (press Enter to stop)", flush=True)

    chunks: list = []
    stop_event = threading.Event()

    def _wait():
        input()
        stop_event.set()

    t = threading.Thread(target=_wait, daemon=True)
    t.start()

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
        while not stop_event.is_set():
            chunk, _ = stream.read(int(sample_rate * 0.1))
            chunks.append(chunk.copy())

    if not chunks:
        logger.warning("No audio captured.")
        return None

    audio = np.concatenate(chunks, axis=0).flatten()
    duration = len(audio) / sample_rate
    rms = float(np.sqrt(np.mean(audio ** 2)))

    level_bar = "█" * min(30, int(rms * 300))
    print(f"  Level: {rms:.4f}  {level_bar}", flush=True)

    if duration < min_duration_s:
        print("  ⚠  Recording too short — try again.", flush=True)
        return None

    if rms < silence_threshold:
        print(_mic_troubleshooting_hint(rms), flush=True)
        return None

    # Write WAV with stdlib wave (no scipy dependency)
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype("int16")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    return tmp.name
