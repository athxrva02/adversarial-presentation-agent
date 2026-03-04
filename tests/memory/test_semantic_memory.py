"""Tests for memory.semantic.SemanticMemory."""
from datetime import datetime

import pytest
from memory.semantic import SemanticMemory
from storage.schemas import (
    ClaimAlignment,
    ClaimRecord,
    SemanticPattern,
    SessionRecord,
)


@pytest.fixture
def sem_mem(vec_store, rel_store):
    return SemanticMemory(vector_store=vec_store, relational_store=rel_store)


def _pattern(i: int = 0, status: str = "active") -> SemanticPattern:
    return SemanticPattern(
        pattern_id=f"sp{i}",
        category="weakness",
        text=f"Tends to avoid baseline comparisons (variant {i}).",
        confidence=0.75,
        direction="stable",
        first_seen="s1",
        last_updated="s2",
        session_count=2,
        status=status,
        evidence=[f"cl{i}"],
    )


def test_store_and_retrieve(sem_mem):
    sem_mem.store_pattern(_pattern(0))
    results = sem_mem.retrieve("baseline comparisons", top_k=1)
    assert len(results) == 1
    assert results[0].pattern_id == "sp0"


def test_get_active(sem_mem):
    sem_mem.store_pattern(_pattern(0, status="active"))
    sem_mem.store_pattern(_pattern(1, status="resolved"))
    active = sem_mem.get_active()
    assert len(active) == 1
    assert active[0].status == "active"


def test_promote_at_threshold(sem_mem, rel_store):
    """Two sessions with 'weak' claims → should promote a pattern."""
    for sid in ("s1", "s2"):
        rel_store.upsert_session(
            SessionRecord(
                session_id=sid,
                timestamp=datetime(2025, 6, 1),
                duration_seconds=60.0,
                overall_score=None,
                strengths=[],
                weaknesses=[],
                claims_count=1,
                contradictions_detected=0,
            )
        )
        rel_store.insert_claim(
            ClaimRecord(
                claim_id=f"{sid}-0",
                session_id=sid,
                turn_number=0,
                claim_text="We have no baseline comparison.",
                alignment=ClaimAlignment.CONTRADICTED,
                mapped_to_slide=None,
                timestamp=datetime(2025, 6, 1),
            )
        )

    promoted = sem_mem.promote("s2")
    assert len(promoted) >= 1
    p = promoted[0]
    assert p.status == "active"
    assert p.session_count >= 2


def test_promote_below_threshold(sem_mem, rel_store):
    """Only one session → should not promote."""
    rel_store.upsert_session(
        SessionRecord(
            session_id="s1",
            timestamp=datetime(2025, 6, 1),
            duration_seconds=60.0,
            overall_score=None,
            strengths=[],
            weaknesses=[],
            claims_count=1,
            contradictions_detected=0,
        )
    )
    rel_store.insert_claim(
        ClaimRecord(
            claim_id="s1-0",
            session_id="s1",
            turn_number=0,
            claim_text="Some claim",
            alignment=ClaimAlignment.CONTRADICTED,
            mapped_to_slide=None,
            timestamp=datetime(2025, 6, 1),
        )
    )
    promoted = sem_mem.promote("s1")
    assert promoted == []


def test_promote_updates_existing(sem_mem, rel_store):
    """If pattern already exists, promote should increment session_count."""
    # Seed existing pattern
    sem_mem.store_pattern(
        SemanticPattern(
            pattern_id="sp_contradicted",
            category="contradicted",
            text="Existing pattern",
            confidence=0.7,
            direction="stable",
            first_seen="s1",
            last_updated="s1",
            session_count=2,
            status="active",
            evidence=["s1-0"],
        )
    )
    # Add two more sessions with contradicted claims
    for sid in ("s1", "s2", "s3"):
        rel_store.upsert_session(
            SessionRecord(
                session_id=sid,
                timestamp=datetime(2025, 6, 1),
                duration_seconds=60.0,
                overall_score=None,
                strengths=[],
                weaknesses=[],
                claims_count=1,
                contradictions_detected=0,
            )
        )
        rel_store.insert_claim(
            ClaimRecord(
                claim_id=f"{sid}-0",
                session_id=sid,
                turn_number=0,
                claim_text="Contradicted claim",
                alignment=ClaimAlignment.CONTRADICTED,
                mapped_to_slide=None,
                timestamp=datetime(2025, 6, 1),
            )
        )

    promoted = sem_mem.promote("s3")
    assert len(promoted) >= 1
    p = promoted[0]
    assert p.session_count >= 3


def test_promote_empty_store(sem_mem):
    assert sem_mem.promote("s1") == []
