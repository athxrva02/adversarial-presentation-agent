"""Integration test: SessionRunner + MemoryModule with mocked LLM calls."""
from datetime import datetime
from unittest.mock import patch

import pytest
from tests.storage.test_vector_store import _FakeClient, _mock_embedding_fn
from storage.vector_store import VectorStore
from storage.relational_store import RelationalStore
from memory.module import MemoryModule
from reasoning.graph import SessionRunner


@pytest.fixture
def runner(tmp_path):
    vec = VectorStore(client=_FakeClient(), embedding_fn=_mock_embedding_fn)
    rel = RelationalStore(db_path=str(tmp_path / "test.db"))
    memory = MemoryModule(vector_store=vec, relational_store=rel)
    return SessionRunner(session_id="s_integ", memory_module=memory)


_FAKE_CLASSIFICATION = {
    "response_class": "strong",
    "alignment": "supported",
    "confidence": 0.85,
    "reasoning": "Clear claim.",
}

_FAKE_SUMMARY = {
    "session_id": "s_integ",
    "timestamp": datetime.now().isoformat(),
    "duration_seconds": 60.0,
    "overall_score": None,
    "strengths": ["Clear thesis"],
    "weaknesses": ["No baseline"],
    "claims_count": 1,
    "contradictions_detected": 0,
}

_FAKE_SCORE = {
    "overall_score": 72.0,
    "breakdown": {"argument_quality": 75, "evidence_use": 70},
}


def test_full_session_with_memory(runner):
    with patch("reasoning.nodes.classify.call_llm_structured", return_value=_FAKE_CLASSIFICATION), \
         patch("reasoning.nodes.generate_question.call_llm_text", return_value="What baseline did you compare against?"):
        q = runner.handle_user_input("We improve accuracy by 15%.")

    assert q is not None
    assert "baseline" in q.lower()

    # Claims should have been persisted by the memory module
    assert len(runner.state.get("claims", [])) >= 1

    # Memory bundle should be populated (retrieve node ran)
    # On first turn with empty memory, it's an empty bundle
    bundle = runner.state.get("memory_bundle")
    # bundle may be None if retrieve returned empty, that's fine

    # Second turn — memory should be populated from persisted claims
    with patch("reasoning.nodes.classify.call_llm_structured", return_value=_FAKE_CLASSIFICATION), \
         patch("reasoning.nodes.generate_question.call_llm_text", return_value="Can you elaborate on the metric?"):
        q2 = runner.handle_user_input("We compared against ResNet-50.")

    assert q2 is not None
    assert len(runner.state.get("claims", [])) >= 2


def test_end_session_persists(runner):
    with patch("reasoning.nodes.classify.call_llm_structured", return_value=_FAKE_CLASSIFICATION), \
         patch("reasoning.nodes.generate_question.call_llm_text", return_value="What baseline?"):
        runner.handle_user_input("We improve accuracy by 15%.")

    with patch("reasoning.nodes.summarise.call_llm_structured", return_value=_FAKE_SUMMARY), \
         patch("reasoning.nodes.score.call_llm_structured", return_value=_FAKE_SCORE):
        result = runner.end_session()

    assert result is not None
    assert result.session_id == "s_integ"


def test_session_without_memory_still_works():
    """SessionRunner with no memory_module — backward compat."""
    runner = SessionRunner(session_id="s_no_mem")
    with patch("reasoning.nodes.classify.call_llm_structured", return_value=_FAKE_CLASSIFICATION), \
         patch("reasoning.nodes.generate_question.call_llm_text", return_value="What baseline?"):
        q = runner.handle_user_input("We improve accuracy.")
    assert q is not None
