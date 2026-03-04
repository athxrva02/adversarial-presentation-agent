"""Working Memory — sliding window buffer of recent turns (in-memory only)."""
from __future__ import annotations

from config import settings


class WorkingMemory:
    """Fixed-size FIFO buffer of conversation turns."""

    def __init__(self, window: int | None = None) -> None:
        self._window = window or settings.working_memory_window
        self._buffer: list[dict] = []

    def add(self, turn: dict) -> None:
        self._buffer.append(turn)
        if len(self._buffer) > self._window:
            self._buffer = self._buffer[-self._window:]

    def get_turns(self) -> list[dict]:
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer = []
