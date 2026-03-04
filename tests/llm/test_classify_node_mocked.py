# tests/test_classify_node_mocked.py
from unittest.mock import patch

from reasoning.nodes.classify import run


def test_classify_node_with_mocked_llm(sample_classification_json):
    state = {
        "user_input": "Our method improves accuracy by 15%.",
        "turn_number": 1,
        "session_id": "s1",
        "memory_bundle": None,
    }

    with patch("reasoning.nodes.classify.call_llm_structured", return_value=sample_classification_json):
        result = run(state)

    assert "classification" in result
    assert result["classification"].response_class.value == "strong"
    assert result["classification"].alignment.value == "supported"

    assert "claims" in result
    assert len(result["claims"]) == 1
    claim = result["claims"][0]
    assert claim.session_id == "s1"
    assert claim.turn_number == 1
    assert "accuracy" in claim.claim_text.lower()