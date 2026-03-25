"""Tests for reasoning.nodes.retrieve — mocked MemoryModule."""
from unittest.mock import MagicMock

from reasoning.nodes.retrieve import run
from storage.schemas import MemoryBundle


def _empty_bundle():
    return MemoryBundle(
        document_context=[],
        episodic_claims=[],
        episodic_sessions=[],
        semantic_patterns=[],
        common_ground=[],
    )


def test_retrieve_returns_memory_bundle():
    mock_module = MagicMock()
    mock_module.retrieve.return_value = _empty_bundle()

    state = {
        "user_input": "We improve accuracy by 15%.",
        "_memory_module": mock_module,
    }
    result = run(state)
    assert "memory_bundle" in result
    assert isinstance(result["memory_bundle"], MemoryBundle)
    mock_module.retrieve.assert_called_once()


def test_retrieve_without_module_returns_none():
    state = {"user_input": "test", "_memory_module": None}
    result = run(state)
    assert result["memory_bundle"] is None


def test_retrieve_passes_user_input_as_query():
    mock_module = MagicMock()
    mock_module.retrieve.return_value = _empty_bundle()

    state = {
        "user_input": "Our baseline is ResNet-50.",
        "_memory_module": mock_module,
    }
    run(state)
    call_args = mock_module.retrieve.call_args
    query = call_args.kwargs["query"]
    assert "LATEST_ANSWER: Our baseline is ResNet-50." in query


def test_retrieve_builds_broader_query_from_turn_history():
    mock_module = MagicMock()
    mock_module.retrieve.return_value = _empty_bundle()

    state = {
        "user_input": "We improved adoption.",
        "turns": [
            {"role": "user", "content": "My presentation is about TravelGo and solo travel."},
            {"role": "agent", "content": "What evidence supports the accessibility claim?"},
        ],
        "claims": [],
        "_memory_module": mock_module,
    }
    run(state)
    query = mock_module.retrieve.call_args.kwargs["query"]
    assert "LATEST_ANSWER: We improved adoption." in query
    assert "PREVIOUS_QUESTION: What evidence supports the accessibility claim?" in query
    assert "PRESENTATION_CONTEXT: My presentation is about TravelGo and solo travel." in query
