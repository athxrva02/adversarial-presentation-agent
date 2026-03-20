# tests/test_graph_runner_mocked.py
from unittest.mock import patch
from datetime import datetime

from reasoning.graph import SessionRunner


def test_session_runner_end_to_end_mocked():
    r = SessionRunner(session_id="s_mock")

    fake_classification = {
        "response_class": "strong",
        "alignment": "supported",
        "confidence": 0.9,
        "reasoning": "Clear claim.",
    }
    fake_summary = {
        "strengths": ["Clear main claim."],
        "weaknesses": ["Missing evaluation detail initially."],
        "key_claims": ["Improves accuracy by 15%."],
        "open_issues": ["Define baseline and metric earlier."],
        "contradictions_detected": 0,
        "overall_notes": "Good start; needs more experimental detail.",
    }
    fake_score = {
        "rubric": {
            "clarity_structure": {"reasoning": "Clear arguments.", "score": 4},
            "evidence_specificity": {"reasoning": "Good metrics.", "score": 3},
            "definition_precision": {"reasoning": "Terms defined.", "score": 3},
            "logical_coherence": {"reasoning": "Solid logic.", "score": 4},
            "handling_adversarial_questions": {"reasoning": "Handled well.", "score": 4},
            "depth_of_understanding": {"reasoning": "Good depth.", "score": 3},
            "concession_and_qualification": {"reasoning": "Some concession.", "score": 3},
            "recovery_from_challenge": {"reasoning": "Recovered.", "score": 3},
        },
        "notes": {
            "top_strengths": ["Clear quantitative improvement."],
            "top_weaknesses": ["Baseline/metric mentioned late."],
            "most_important_next_step": "State baseline and metric immediately.",
        },
    }

    with patch("reasoning.nodes.classify.call_llm_structured", return_value=fake_classification), \
         patch("reasoning.nodes.classify.call_llm_text", side_effect=lambda _sys, user, **kw: user), \
         patch("reasoning.nodes.generate_question.call_llm_text", return_value="Which baseline and metric did you use?"), \
         patch("reasoning.nodes.summarise.call_llm_structured", return_value=fake_summary), \
         patch("reasoning.nodes.score.call_llm_structured", return_value=fake_score):

        q = r.handle_user_input("We improve accuracy by 15% compared to baseline.")
        assert q.endswith("?")
        assert "baseline" in q.lower()

        session_record = r.end_session()
        assert session_record is not None
        assert session_record.session_id == "s_mock"
        # overall_score is computed deterministically from rubric weights, not from LLM
        assert session_record.overall_score is not None
        assert 0.0 <= session_record.overall_score <= 100.0
        assert session_record.claims_count >= 1  # classify node appends a ClaimRecord

        # turn history should contain user + agent
        assert len(r.state["turns"]) >= 2