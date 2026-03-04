"""Tests for memory.episodic.EpisodicMemory."""
from datetime import datetime

import pytest
from memory.episodic import EpisodicMemory
from storage.schemas import ClaimAlignment, ClaimRecord, SessionRecord


@pytest.fixture
def ep_mem(vec_store, rel_store):
    return EpisodicMemory(vector_store=vec_store, relational_store=rel_store)


def _claim(i: int = 0, session_id: str = "s1") -> ClaimRecord:
    return ClaimRecord(
        claim_id=f"{session_id}-{i}",
        session_id=session_id,
        turn_number=i,
        claim_text=f"We improve accuracy by {i * 5}%.",
        alignment=ClaimAlignment.SUPPORTED,
        mapped_to_slide=None,
        timestamp=datetime(2025, 6, 1, 12, i, 0),
    )


def _session(session_id: str = "s1", score: float | None = 72.0) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        timestamp=datetime(2025, 6, 1, 12, 0, 0),
        duration_seconds=120.0,
        overall_score=score,
        strengths=["Clear thesis"],
        weaknesses=["No baseline"],
        claims_count=3,
        contradictions_detected=0,
    )


def test_store_and_retrieve_claim(ep_mem, rel_store):
    # Need session in relational store for FK
    rel_store.upsert_session(_session("s1"))
    ep_mem.store_claim(_claim(0, "s1"))
    results = ep_mem.retrieve_claims("accuracy", top_k=1)
    assert len(results) == 1
    assert results[0].claim_id == "s1-0"


def test_store_session_persists(ep_mem):
    session = _session("s1")
    claims = [_claim(i, "s1") for i in range(3)]
    ep_mem.store_session(session, claims)
    results = ep_mem.retrieve_sessions("Clear thesis", top_k=1)
    assert len(results) == 1
    assert results[0].session_id == "s1"


def test_retrieve_claims_top_k(ep_mem, rel_store):
    rel_store.upsert_session(_session("s1"))
    for i in range(4):
        ep_mem.store_claim(_claim(i, "s1"))
    results = ep_mem.retrieve_claims("accuracy", top_k=2)
    assert len(results) == 2


def test_retrieve_sessions_top_k(ep_mem):
    for i in range(3):
        ep_mem.store_session(_session(f"s{i}"), [_claim(0, f"s{i}")])
    results = ep_mem.retrieve_sessions("thesis", top_k=2)
    assert len(results) == 2


def test_get_claims_for_session(ep_mem, rel_store):
    rel_store.upsert_session(_session("s1"))
    rel_store.upsert_session(_session("s2"))
    for i in range(3):
        ep_mem.store_claim(_claim(i, "s1"))
    ep_mem.store_claim(_claim(0, "s2"))
    claims = rel_store.get_claims_for_session("s1")
    assert len(claims) == 3


def test_empty_retrieval(ep_mem):
    assert ep_mem.retrieve_claims("anything", top_k=5) == []
    assert ep_mem.retrieve_sessions("anything", top_k=5) == []
