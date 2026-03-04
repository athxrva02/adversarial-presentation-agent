"""Tests for memory.retrieval — merge, dedup, recency ranking."""
from datetime import datetime

from memory.retrieval import merge_and_rank, _dedup_by_id, _recency_score
from storage.schemas import (
    ClaimAlignment,
    ClaimRecord,
    CommonGroundEntry,
    DocumentChunk,
    MemoryBundle,
    SemanticPattern,
    SessionRecord,
)


def _claim(claim_id: str, session_id: str = "s1") -> ClaimRecord:
    return ClaimRecord(
        claim_id=claim_id,
        session_id=session_id,
        turn_number=0,
        claim_text=f"Claim {claim_id}",
        alignment=ClaimAlignment.SUPPORTED,
        mapped_to_slide=None,
        timestamp=datetime(2025, 6, 1),
    )


def _chunk(chunk_id: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        slide_number=1,
        chunk_type="claim",
        text=f"Chunk {chunk_id}",
        position_in_pdf=0,
    )


def test_merge_empty_returns_empty_bundle():
    bundle = merge_and_rank([], [], [], [], [], top_k=5)
    assert isinstance(bundle, MemoryBundle)
    assert bundle.document_context == []
    assert bundle.episodic_claims == []


def test_dedup_by_id():
    items = [_claim("c1", "s1"), _claim("c1", "s2"), _claim("c2", "s1")]
    deduped = _dedup_by_id(items, "claim_id")
    assert len(deduped) == 2
    assert deduped[0].claim_id == "c1"
    assert deduped[1].claim_id == "c2"


def test_recency_score_recent_is_higher():
    order = {"s1": 2, "s2": 1, "s3": 0}  # s3 is most recent
    c_old = _claim("c1", "s1")
    c_new = _claim("c2", "s3")
    assert _recency_score(c_new, order) > _recency_score(c_old, order)


def test_recency_score_applies_decay():
    order = {"s1": 2, "s2": 0}
    c = _claim("c1", "s1")
    score = _recency_score(c, order)
    assert 0 < score < 1  # decayed
    c_recent = _claim("c2", "s2")
    assert _recency_score(c_recent, order) == 1.0  # age 0 → factor^0 = 1


def test_top_k_truncates():
    chunks = [_chunk(f"dc{i}") for i in range(10)]
    bundle = merge_and_rank(chunks, [], [], [], [], top_k=3)
    assert len(bundle.document_context) == 3


def test_merge_complete_bundle():
    chunks = [_chunk("dc0")]
    claims = [_claim("c0")]
    sessions = [
        SessionRecord(
            session_id="s1", timestamp=datetime(2025, 6, 1),
            duration_seconds=60.0, overall_score=None,
            strengths=[], weaknesses=[], claims_count=0,
            contradictions_detected=0,
        )
    ]
    patterns = [
        SemanticPattern(
            pattern_id="sp0", category="weakness", text="Pattern",
            confidence=0.7, direction="stable", first_seen="s1",
            last_updated="s1", session_count=1, status="active",
            evidence=[],
        )
    ]
    cg = [
        CommonGroundEntry(
            cg_id="cg0", pdf_chunk_ref=None, original_text=None,
            negotiated_text="Agreed", proposed_by="agent",
            session_agreed="s1", version=1, timestamp=datetime(2025, 6, 1),
        )
    ]
    bundle = merge_and_rank(chunks, claims, sessions, patterns, cg, top_k=5)
    assert len(bundle.document_context) == 1
    assert len(bundle.episodic_claims) == 1
    assert len(bundle.episodic_sessions) == 1
    assert len(bundle.semantic_patterns) == 1
    assert len(bundle.common_ground) == 1


def test_returns_memory_bundle_type():
    bundle = merge_and_rank([], [], [], [], [], top_k=5)
    assert isinstance(bundle, MemoryBundle)
