# tests/test_generate_question_node_mocked.py
from unittest.mock import MagicMock, patch

from reasoning.nodes.generate_question import run
from storage.schemas import MemoryBundle, DocumentChunk


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


def test_generate_question_node_can_choose_document_driven_mode():
    # Context expansion change: even turns can introduce untouched document chunks.
    memory_bundle = MemoryBundle(
        document_context=[],
        episodic_claims=[],
        episodic_sessions=[],
        semantic_patterns=[],
        common_ground=[],
    )
    mock_module = MagicMock()
    mock_module.get_document_question_candidates.return_value = [
        DocumentChunk(
            chunk_id="dc_doc_1",
            slide_number=4,
            chunk_type="evidence",
            text="User interviews identified trust and safety as the biggest barrier.",
            position_in_pdf=40,
        ),
        DocumentChunk(
            chunk_id="dc_doc_2",
            slide_number=5,
            chunk_type="definition",
            text="Accessibility is defined as reducing planning friction for solo travelers.",
            position_in_pdf=50,
        ),
    ]

    state = {
        "user_input": "We improve accuracy by 15%.",
        "turn_number": 2,
        "session_id": "s1",
        "memory_bundle": memory_bundle,
        "classification": None,
        "turns": [
            {"role": "user", "content": "My topic is TravelGo."},
            {"role": "agent", "content": "What baseline did you compare against?"},
        ],
        "_memory_module": mock_module,
        "asked_document_chunk_ids": [],
        "question_modes": ["answer_driven"],
        "question_focus_history": ["evidence"],
        "document_coverage_keys": [],
    }

    with patch(
        "reasoning.nodes.generate_question.call_llm_text",
        return_value="Your document highlights trust and safety barriers; how does TravelGo address those concretely?",
    ):
        result = run(state)

    assert result["agent_response"].strip().endswith("?")
    assert result["question_modes"] == ["document_driven"]
    assert result["asked_document_chunk_ids"] == ["dc_doc_1", "dc_doc_2"]
    assert result["question_focus_history"] == ["definition"]
    assert "type:evidence" in result["document_coverage_keys"]
    assert "slide:4" in result["document_coverage_keys"]
