# tests/test_score_node_live.py
import pytest
from datetime import datetime
from urllib.request import urlopen, Request
from urllib.error import URLError

from reasoning.nodes.score import run
from reasoning.prompts.scoring import RUBRIC_WEIGHTS
from storage.schemas import SessionRecord


def _ollama_reachable() -> bool:
    try:
        req = Request("http://localhost:11434/api/tags", headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except URLError:
        return False
    except Exception:
        return False


def test_score_node_live(live_enabled):
    if not live_enabled:
        pytest.skip("Live test disabled. Run with: pytest --live")
    if not _ollama_reachable():
        pytest.skip("Ollama not reachable at http://localhost:11434")

    summary = SessionRecord(
        session_id="live_sess",
        timestamp=datetime.now(),
        duration_seconds=60.0,
        overall_score=None,
        strengths=["Stated main claim clearly."],
        weaknesses=["Initially missing baseline/metric detail."],
        claims_count=2,
        contradictions_detected=0,
    )

    state = {
        "session_id": "live_sess",
        "session_summary": summary,
        "turns": [
            {"role": "agent", "content": "State your main contribution."},
            {"role": "user", "content": "We improve accuracy by 15% compared to baseline."},
            {"role": "agent", "content": "Which baseline and what metric?"},
            {"role": "user", "content": "Logistic regression baseline; accuracy on held-out test set."},
        ],
        "score_breakdown": {},
    }

    result = run(state)
    assert "session_summary" in result
    scored = result["session_summary"]
    assert scored.session_id == "live_sess"
    assert scored.overall_score is not None
    assert 0.0 <= float(scored.overall_score) <= 100.0

    bd = result.get("score_breakdown", {})
    assert "rubric_scores" in bd
    assert "rubric_reasoning" in bd
    assert "notes" in bd

    # All 8 dimensions should have scores in 1-5
    for dim in RUBRIC_WEIGHTS:
        assert dim in bd["rubric_scores"], f"Missing dimension: {dim}"
        assert 1 <= bd["rubric_scores"][dim] <= 5, f"{dim} score out of range: {bd['rubric_scores'][dim]}"
