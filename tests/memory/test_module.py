"""Tests for memory.module.MemoryModule — the orchestrator."""
from datetime import datetime

import pytest
from tests.storage.test_vector_store import _FakeClient, _mock_embedding_fn
from storage.vector_store import VectorStore
from storage.relational_store import RelationalStore
from memory.module import MemoryModule
from storage.schemas import (
    ClaimAlignment,
    ClaimRecord,
    CommonGroundEntry,
    DocumentChunk,
    MemoryBundle,
    SessionRecord,
)


@pytest.fixture
def module(tmp_path):
    vec = VectorStore(client=_FakeClient(), embedding_fn=_mock_embedding_fn)
    rel = RelationalStore(db_path=str(tmp_path / "test.db"))
    return MemoryModule(vector_store=vec, relational_store=rel)


def _session(sid: str = "s1") -> SessionRecord:
    return SessionRecord(
        session_id=sid,
        timestamp=datetime(2025, 6, 1),
        duration_seconds=60.0,
        overall_score=72.0,
        strengths=["Clear thesis"],
        weaknesses=["No baseline"],
        claims_count=2,
        contradictions_detected=0,
    )


def _claim(cid: str = "c0", sid: str = "s1") -> ClaimRecord:
    return ClaimRecord(
        claim_id=cid,
        session_id=sid,
        turn_number=0,
        claim_text="We improve accuracy by 15%.",
        alignment=ClaimAlignment.SUPPORTED,
        mapped_to_slide=None,
        timestamp=datetime(2025, 6, 1),
    )


def _chunk(cid: str = "dc0") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid,
        slide_number=1,
        chunk_type="claim",
        text="Our method outperforms baselines.",
        position_in_pdf=0,
    )


def test_retrieve_returns_memory_bundle(module):
    result = module.retrieve("test", stores=["document"])
    assert isinstance(result, MemoryBundle)


def test_empty_retrieve_returns_empty_bundle(module):
    bundle = module.retrieve("test", stores=["document", "episodic", "semantic", "common_ground"])
    assert bundle.document_context == []
    assert bundle.episodic_claims == []
    assert bundle.episodic_sessions == []
    assert bundle.semantic_patterns == []
    assert bundle.common_ground == []


def test_store_and_retrieve_document(module):
    module.store_document(_chunk("dc0"))
    bundle = module.retrieve("outperforms baselines", stores=["document"])
    assert len(bundle.document_context) == 1
    assert bundle.document_context[0].chunk_id == "dc0"


def test_store_and_retrieve_claim(module):
    module.store_session(_session("s1"), [_claim("c0", "s1")])
    bundle = module.retrieve("accuracy", stores=["episodic"])
    assert len(bundle.episodic_claims) == 1


def test_store_and_retrieve_session(module):
    module.store_session(_session("s1"), [_claim("c0", "s1")])
    bundle = module.retrieve("Clear thesis", stores=["episodic"])
    assert len(bundle.episodic_sessions) == 1


def test_store_common_ground(module):
    entry = CommonGroundEntry(
        cg_id="cg0",
        pdf_chunk_ref=None,
        original_text=None,
        negotiated_text="Agreed: baseline is ResNet-50.",
        proposed_by="agent",
        session_agreed="s1",
        version=1,
        timestamp=datetime(2025, 6, 1),
    )
    module.store_common_ground(entry)
    bundle = module.retrieve("baseline ResNet", stores=["common_ground"])
    assert len(bundle.common_ground) == 1


def test_retrieve_only_requested_stores(module):
    module.store_document(_chunk("dc0"))
    module.store_session(_session("s1"), [_claim("c0", "s1")])
    bundle = module.retrieve("test", stores=["document"])
    assert len(bundle.document_context) == 1
    assert bundle.episodic_claims == []


def test_retrieve_top_k(module):
    for i in range(10):
        module.store_document(_chunk(f"dc{i}"))
    bundle = module.retrieve("method", stores=["document"], top_k=3)
    assert len(bundle.document_context) == 3


def test_get_document_question_candidates(module):
    module.store_document(
        DocumentChunk(
            chunk_id="dc_def",
            slide_number=1,
            chunk_type="definition",
            text="Accessibility means reducing planning effort.",
            position_in_pdf=0,
        )
    )
    module.store_document(
        DocumentChunk(
            chunk_id="dc_evd",
            slide_number=2,
            chunk_type="evidence",
            text="Survey evidence highlights safety concerns.",
            position_in_pdf=10,
        )
    )

    results = module.get_document_question_candidates(limit=2)
    result_ids = {r.chunk_id for r in results}
    assert result_ids == {"dc_def", "dc_evd"}


def test_promote_patterns(module):
    for sid in ("s1", "s2"):
        module.store_session(
            SessionRecord(
                session_id=sid,
                timestamp=datetime(2025, 6, 1),
                duration_seconds=60.0,
                overall_score=None,
                strengths=[],
                weaknesses=[],
                claims_count=1,
                contradictions_detected=0,
            ),
            [
                ClaimRecord(
                    claim_id=f"{sid}-0",
                    session_id=sid,
                    turn_number=0,
                    claim_text="Contradicted claim",
                    alignment=ClaimAlignment.CONTRADICTED,
                    mapped_to_slide=None,
                    timestamp=datetime(2025, 6, 1),
                )
            ],
        )
    promoted = module.promote_patterns("s2")
    assert len(promoted) >= 1
