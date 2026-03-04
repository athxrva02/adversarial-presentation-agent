"""Tests for memory.working.WorkingMemory — pure in-memory sliding window."""
from memory.working import WorkingMemory


def test_empty_window_returns_empty():
    wm = WorkingMemory(window=20)
    assert wm.get_turns() == []


def test_add_and_get_within_window():
    wm = WorkingMemory(window=20)
    for i in range(5):
        wm.add({"role": "user", "content": f"turn {i}"})
    assert len(wm.get_turns()) == 5


def test_window_evicts_oldest():
    wm = WorkingMemory(window=3)
    for i in range(5):
        wm.add({"role": "user", "content": f"turn {i}"})
    turns = wm.get_turns()
    assert len(turns) == 3
    assert turns[0]["content"] == "turn 2"
    assert turns[-1]["content"] == "turn 4"


def test_clear_resets_buffer():
    wm = WorkingMemory(window=20)
    wm.add({"role": "user", "content": "hello"})
    wm.clear()
    assert wm.get_turns() == []


def test_boundary_exact_window():
    wm = WorkingMemory(window=3)
    for i in range(3):
        wm.add({"role": "user", "content": f"turn {i}"})
    assert len(wm.get_turns()) == 3
