# tests/test_summarise_node_live.py
import json
import pytest
from urllib.request import urlopen, Request
from urllib.error import URLError

from reasoning.nodes.summarise import run


def _ollama_reachable() -> bool:
    try:
        req = Request("http://localhost:11434/api/tags", headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except URLError:
        return False
    except Exception:
        return False


def test_summarise_node_live(live_enabled):
    if not live_enabled:
        pytest.skip("Live test disabled. Run with: pytest --live")
    if not _ollama_reachable():
        pytest.skip("Ollama not reachable at http://localhost:11434")

    state = {
        "session_id": "live_sess",
        "turns": [
            {"role": "assistant", "content": "State your main contribution."},
            {"role": "user", "content": "We improve accuracy by 15% compared to a baseline."},
            {"role": "assistant", "content": "Which baseline and what metric?"},
            {"role": "user", "content": "Logistic regression baseline; accuracy metric on held-out test set."},
        ],
        "claims": [],
        "memory_bundle": None,
        "duration_seconds": 60.0,
    }

    result = run(state)
    assert "session_summary" in result

    summary = result["session_summary"]
    assert summary.session_id == "live_sess"
    assert isinstance(summary.strengths, list)
    assert isinstance(summary.weaknesses, list)
    assert summary.claims_count == 0
    assert isinstance(summary.contradictions_detected, int)

    # Extra info stored separately
    assert "score_breakdown" in result
    assert "overall_notes" in result["score_breakdown"]