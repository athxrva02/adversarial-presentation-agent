# tests/test_summarise_node_mocked.py
from unittest.mock import patch

from reasoning.nodes.summarise import run


def test_summarise_node_mocked():
    state = {
        "session_id": "s1",
        "turns": [
            {"role": "user", "content": "Our approach improves accuracy by 15%."},
            {"role": "agent", "content": "What baseline did you compare against?"},
            {"role": "user", "content": "Baseline is logistic regression on the same dataset."},
        ],
        "claims": [],  # can be empty; node should still work
        "memory_bundle": None,
        "duration_seconds": 123.0,
    }

    fake_llm_out = {
        "strengths": ["Clear quantitative claim stated up front."],
        "weaknesses": ["Did not specify evaluation setup initially."],
        "key_claims": ["Method improves accuracy by 15% vs baseline."],
        "open_issues": ["Clarify dataset and evaluation protocol."],
        "contradictions_detected": 0,
        "overall_notes": "Good start but needs clearer experimental detail."
    }

    with patch("reasoning.nodes.summarise.call_llm_structured", return_value=fake_llm_out):
        result = run(state)

    assert "session_summary" in result
    summary = result["session_summary"]
    assert summary.session_id == "s1"
    assert summary.duration_seconds == 123.0
    assert summary.overall_score is None
    assert summary.claims_count == 0
    assert summary.contradictions_detected == 0
    assert len(summary.strengths) == 1
    assert len(summary.weaknesses) == 1

    assert "score_breakdown" in result
    assert "open_issues" in result["score_breakdown"]