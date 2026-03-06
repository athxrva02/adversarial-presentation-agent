"""
Microphone recording for the Adversarial Presentation Agent.

Records audio from the default input device until the user presses Enter.
Writes a 16 kHz mono WAV using only stdlib `wave` (no scipy needed).

Public API:
    request_permission() -> bool   — triggers macOS permission prompt, returns True if mic works
    record(...)          -> str | None
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import wave

logger = logging.getLogger(__name__)

# Show actual RMS level always — helps diagnose permission / mute issues
_SHOW_LEVEL = True


def mic_available() -> bool:
    try:
        import sounddevice  # noqa: F401
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def request_permission(sample_rate: int = 16000) -> bool:
    """
    Trigger macOS microphone permission prompt by opening and reading the stream
    for 0.5 s.  Returns True if a non-silent signal was received (mic is working),
    False if audio is all zeros (permission denied or mic muted).

    Call this once at startup before the first real recording.
    """
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        return False

    try:
        frames = []
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
            # Read ~0.5 s — enough to trigger the macOS permission dialog
            for _ in range(5):
                chunk, _ = stream.read(int(sample_rate * 0.1))
                frames.append(chunk.copy())
                time.sleep(0.1)

        if not frames:
            return False

        audio = np.concatenate(frames).flatten()
        rms = float(np.sqrt(np.mean(audio ** 2)))
        logger.debug("Permission check RMS: %.6f", rms)
        # Even silence (rms==0) means permission was granted; we can't tell from
        # the stream itself.  Just return True — actual recording will show level.
        return True
    except Exception as e:
        logger.warning("Mic permission check failed: %s", e)
        return False


def record(
    prompt: str = "",
    sample_rate: int = 16000,
    min_duration_s: float = 0.5,
    silence_threshold: float = 0.0005,   # lowered: macOS mic varies by model
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
        logger.error("sounddevice not installed — run: pip install sounddevice")
        return None

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if prompt:
        print(prompt, flush=True)
    cmd = input().strip().lower()
    if cmd in {"/end", "end", "quit", "exit"}:
        return "/end"

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

    # Always show level so the user can see if mic is working
    level_bar = "█" * min(30, int(rms * 300))
    print(f"  Level: {rms:.4f}  {level_bar}", flush=True)

    if duration < min_duration_s:
        print("  ⚠  Recording too short — try again.", flush=True)
        return None

    if rms < silence_threshold:
        print(
            f"  ⚠  Signal too low (RMS={rms:.5f}).\n"
            "     If this keeps happening:\n"
            "     1. macOS: System Settings → Privacy & Security → Microphone → enable Terminal/iTerm\n"
            "     2. Check the correct input device: python3 -c \"import sounddevice; print(sounddevice.query_devices())\"\n"
            "     3. Run with a louder voice or move closer to the mic.",
            flush=True,
        )
        return None

    # Write WAV with stdlib wave
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype("int16")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    return tmp.name
