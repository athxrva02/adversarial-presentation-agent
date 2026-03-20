# tests/test_score_node_mocked.py
from unittest.mock import patch
from datetime import datetime

from reasoning.nodes.score import run, compute_overall_score
from reasoning.prompts.scoring import RUBRIC_WEIGHTS
from storage.schemas import SessionRecord


def _make_rubric(**overrides):
    """Build a full rubric dict with CoT reasoning, defaulting all scores to 3."""
    defaults = {dim: {"reasoning": f"test reasoning for {dim}", "score": 3} for dim in RUBRIC_WEIGHTS}
    defaults.update(overrides)
    return defaults


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
        "rubric": _make_rubric(
            clarity_structure={"reasoning": "Clear arguments.", "score": 4},
            evidence_specificity={"reasoning": "Good metrics.", "score": 4},
            logical_coherence={"reasoning": "Solid logic.", "score": 4},
        ),
        "notes": {
            "top_strengths": ["Clear quantitative improvement."],
            "top_weaknesses": ["Define evaluation setup earlier."],
            "most_important_next_step": "State baseline/metric immediately when quoting gains.",
        },
    }

    with patch("reasoning.nodes.score.call_llm_structured", return_value=fake_llm_out):
        result = run(state)

    assert "session_summary" in result
    # overall_score is computed deterministically, not from LLM
    assert result["session_summary"].overall_score is not None
    assert 0.0 <= result["session_summary"].overall_score <= 100.0

    assert "score_breakdown" in result
    bd = result["score_breakdown"]
    assert bd["overall_score"] == result["session_summary"].overall_score
    assert "rubric_scores" in bd
    assert "rubric_reasoning" in bd
    assert "rubric_weights" in bd
    assert "notes" in bd
    # Ensure merge preserved prior keys
    assert bd["overall_notes"] == "pre-existing notes"
    # Verify individual dimension scores
    assert bd["rubric_scores"]["clarity_structure"] == 4
    assert bd["rubric_scores"]["evidence_specificity"] == 4
    assert bd["rubric_scores"]["logical_coherence"] == 4
    # Default dimensions should be 3
    assert bd["rubric_scores"]["definition_precision"] == 3
    # Verify reasoning is extracted
    assert bd["rubric_reasoning"]["clarity_structure"] == "Clear arguments."


def test_score_node_handles_plain_number_rubric():
    """Backward compat: if LLM returns plain numbers instead of CoT dicts."""
    summary = SessionRecord(
        session_id="s2",
        timestamp=datetime.now(),
        duration_seconds=30.0,
        overall_score=None,
        strengths=[],
        weaknesses=[],
        claims_count=0,
        contradictions_detected=0,
    )

    state = {
        "session_id": "s2",
        "session_summary": summary,
        "turns": [],
        "score_breakdown": {},
    }

    fake_llm_out = {
        "rubric": {dim: 4 for dim in RUBRIC_WEIGHTS},
        "notes": {"top_strengths": [], "top_weaknesses": [], "most_important_next_step": ""},
    }

    with patch("reasoning.nodes.score.call_llm_structured", return_value=fake_llm_out):
        result = run(state)

    # All scores are 4, so overall = (4-1)*25 = 75.0
    assert result["session_summary"].overall_score == 75.0
    assert all(v == 4 for v in result["score_breakdown"]["rubric_scores"].values())


def test_compute_overall_score_deterministic():
    """Verify the weighted computation maps correctly from 1-5 to 0-100."""
    # All 1s -> 0.0
    rubric_all_1 = {dim: {"reasoning": "", "score": 1} for dim in RUBRIC_WEIGHTS}
    assert compute_overall_score(rubric_all_1) == 0.0

    # All 5s -> 100.0
    rubric_all_5 = {dim: {"reasoning": "", "score": 5} for dim in RUBRIC_WEIGHTS}
    assert compute_overall_score(rubric_all_5) == 100.0

    # All 3s -> 50.0
    rubric_all_3 = {dim: {"reasoning": "", "score": 3} for dim in RUBRIC_WEIGHTS}
    assert compute_overall_score(rubric_all_3) == 50.0


def test_score_clamping():
    """Scores outside 1-5 range are clamped."""
    from reasoning.nodes.score import _clamp_score
    assert _clamp_score(0) == 1
    assert _clamp_score(-5) == 1
    assert _clamp_score(6) == 5
    assert _clamp_score(100) == 5
    assert _clamp_score(3.7) == 4  # rounds
    assert _clamp_score("invalid") == 1


def test_rubric_weights_sum_to_one():
    """Weights must sum to 1.0 for correct scaling."""
    total = sum(RUBRIC_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"


def test_all_dimensions_present():
    """All 8 rubric dimensions have weights defined."""
    expected = {
        "clarity_structure",
        "evidence_specificity",
        "definition_precision",
        "logical_coherence",
        "handling_adversarial_questions",
        "depth_of_understanding",
        "concession_and_qualification",
        "recovery_from_challenge",
    }
    assert set(RUBRIC_WEIGHTS.keys()) == expected
