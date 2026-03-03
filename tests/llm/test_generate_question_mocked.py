# tests/test_generate_question_node_mocked.py
from unittest.mock import patch

from reasoning.nodes.generate_question import run


def test_generate_question_node_with_mocked_llm():
    state = {
        "user_input": "Our approach improves accuracy by 15% compared to baseline.",
        "turn_number": 1,
        "session_id": "s1",
        "memory_bundle": None,
        "classification": None,
    }

    with patch("reasoning.nodes.generate_question.call_llm_text", return_value="What baseline did you compare against, and how was accuracy measured?"):
        result = run(state)

    assert "agent_response" in result
    assert "baseline" in result["agent_response"].lower()
    assert result["agent_response"].strip().endswith("?")