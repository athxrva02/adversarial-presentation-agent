# tests/test_score_node_mocked.py
from unittest.mock import patch
from datetime import datetime

from reasoning.nodes.score import run
from storage.schemas import SessionRecord


def test_score_node_mocked():
    summary = SessionRecord(
        session_id="s1",
        timestamp=datetime.now(),
        duration_seconds=60.0,
        overall_score=None,
        strengths=["Clear main claim."],
        weaknesses=["No baseline specified initially."],
        claims_count=2,
        contradictions_detected=0,
    )

    state = {
        "session_id": "s1",
        "session_summary": summary,
        "turns": [
            {"role": "agent", "content": "What is your contribution?"},
            {"role": "user", "content": "We improve accuracy by 15%."},
            {"role": "agent", "content": "Compared to what baseline and which metric?"},
            {"role": "user", "content": "Logistic regression baseline; accuracy on test set."},
        ],
        "score_breakdown": {"overall_notes": "pre-existing notes"},
    }

    fake_llm_out = {
        "overall_score": 72,
        "rubric": {
            "clarity_structure": 70,
            "evidence_specificity": 75,
            "definition_precision": 60,
            "logical_coherence": 80,
            "handling_adversarial_questions": 75,
        },
        "notes": {
            "top_strengths": ["Clear quantitative improvement."],
            "top_weaknesses": ["Define evaluation setup earlier."],
            "most_important_next_step": "State baseline/metric immediately when quoting gains.",
        },
    }

    with patch("reasoning.nodes.score.call_llm_structured", return_value=fake_llm_out):
        result = run(state)

    assert "session_summary" in result
    assert result["session_summary"].overall_score == 72.0

    assert "score_breakdown" in result
    bd = result["score_breakdown"]
    assert bd["overall_score"] == 72.0
    assert "rubric" in bd and "notes" in bd
    # Ensure merge preserved prior keys
    assert bd["overall_notes"] == "pre-existing notes"