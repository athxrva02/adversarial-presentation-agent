"""Tests for storage.relational_store.RelationalStore using tmp_path SQLite."""
import sqlite3
from datetime import datetime

import pytest
from storage.relational_store import RelationalStore
from storage.schemas import (
    ClaimAlignment,
    ClaimRecord,
    CommonGroundEntry,
    DocumentChunk,
    SemanticPattern,
    SessionRecord,
)


@pytest.fixture
def store(tmp_path):
    return RelationalStore(db_path=str(tmp_path / "test.db"))


# ---- Schema ----------------------------------------------------------------

def test_tables_created_on_init(store):
    conn = sqlite3.connect(store.db_path)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()
    expected = {
        "document_chunks",
        "session_records",
        "claim_records",
        "semantic_patterns",
        "pattern_evidence",
        "common_ground",
    }
    assert expected.issubset(tables)


# ---- DocumentChunk ---------------------------------------------------------

def test_upsert_and_get_document_chunk(store):
    chunk = DocumentChunk(
        chunk_id="dc1",
        slide_number=3,
        chunk_type="claim",
        text="Our approach outperforms baselines.",
        position_in_pdf=42,
        embedding_id="emb1",
    )
    store.upsert_document_chunk(chunk)
    got = store.get_document_chunk("dc1")
    assert got is not None
    assert got.chunk_id == "dc1"
    assert got.slide_number == 3
    assert got.text == "Our approach outperforms baselines."


def test_get_all_document_chunks(store):
    for i in range(3):
        store.upsert_document_chunk(
            DocumentChunk(
                chunk_id=f"dc{i}",
                slide_number=i,
                chunk_type="claim",
                text=f"Chunk {i}",
                position_in_pdf=i,
            )
        )
    assert len(store.get_all_document_chunks()) == 3


# ---- SessionRecord ---------------------------------------------------------

def test_upsert_and_get_session(store):
    session = SessionRecord(
        session_id="s1",
        timestamp=datetime(2025, 6, 1, 12, 0, 0),
        duration_seconds=120.5,
        overall_score=78.0,
        strengths=["Good structure", "Clear thesis"],
        weaknesses=["No baseline"],
        claims_count=3,
        contradictions_detected=1,
    )
    store.upsert_session(session)
    got = store.get_session("s1")
    assert got is not None
    assert got.session_id == "s1"
    assert got.strengths == ["Good structure", "Clear thesis"]
    assert got.weaknesses == ["No baseline"]
    assert got.overall_score == 78.0


def test_get_all_sessions(store):
    for i in range(2):
        store.upsert_session(
            SessionRecord(
                session_id=f"s{i}",
                timestamp=datetime.now(),
                duration_seconds=60.0,
                overall_score=None,
                strengths=[],
                weaknesses=[],
                claims_count=0,
                contradictions_detected=0,
            )
        )
    assert len(store.get_all_sessions()) == 2


# ---- ClaimRecord -----------------------------------------------------------

def _make_session(store, session_id="s1"):
    store.upsert_session(
        SessionRecord(
            session_id=session_id,
            timestamp=datetime.now(),
            duration_seconds=60.0,
            overall_score=None,
            strengths=[],
            weaknesses=[],
            claims_count=0,
            contradictions_detected=0,
        )
    )


def test_insert_and_get_claim(store):
    _make_session(store, "s1")
    claim = ClaimRecord(
        claim_id="cl1",
        session_id="s1",
        turn_number=1,
        claim_text="We improve accuracy by 15%.",
        alignment=ClaimAlignment.SUPPORTED,
        mapped_to_slide=2,
        timestamp=datetime.now(),
    )
    store.insert_claim(claim)
    got = store.get_claim("cl1")
    assert got is not None
    assert got.claim_text == "We improve accuracy by 15%."
    assert got.alignment == ClaimAlignment.SUPPORTED


def test_get_claims_for_session(store):
    _make_session(store, "s1")
    _make_session(store, "s2")
    for i in range(3):
        store.insert_claim(
            ClaimRecord(
                claim_id=f"cl_s1_{i}",
                session_id="s1",
                turn_number=i,
                claim_text=f"Claim {i}",
                alignment=ClaimAlignment.NOVEL,
                mapped_to_slide=None,
                timestamp=datetime.now(),
            )
        )
    store.insert_claim(
        ClaimRecord(
            claim_id="cl_s2_0",
            session_id="s2",
            turn_number=0,
            claim_text="Other session claim",
            alignment=ClaimAlignment.NOVEL,
            mapped_to_slide=None,
            timestamp=datetime.now(),
        )
    )
    claims = store.get_claims_for_session("s1")
    assert len(claims) == 3
    assert all(c.session_id == "s1" for c in claims)


# ---- SemanticPattern -------------------------------------------------------

def test_upsert_and_get_semantic_pattern(store):
    pattern = SemanticPattern(
        pattern_id="sp1",
        category="weakness",
        text="Tends to avoid providing baselines.",
        confidence=0.75,
        direction="stable",
        first_seen="s1",
        last_updated="s2",
        session_count=2,
        status="active",
        evidence=["cl1", "cl2"],
    )
    store.upsert_semantic_pattern(pattern)
    got = store.get_pattern("sp1")
    assert got is not None
    assert got.text == "Tends to avoid providing baselines."
    assert got.evidence == ["cl1", "cl2"]
    assert got.session_count == 2


def test_get_semantic_patterns_by_status(store):
    for i, status in enumerate(["active", "active", "resolved"]):
        store.upsert_semantic_pattern(
            SemanticPattern(
                pattern_id=f"sp{i}",
                category="weakness",
                text=f"Pattern {i}",
                confidence=0.5,
                direction="stable",
                first_seen="s1",
                last_updated="s1",
                session_count=1,
                status=status,
                evidence=[],
            )
        )
    active = store.get_semantic_patterns(status="active")
    assert len(active) == 2
    all_patterns = store.get_semantic_patterns()
    assert len(all_patterns) == 3


# ---- CommonGround ----------------------------------------------------------

def test_upsert_and_get_common_ground(store):
    # Insert referenced document chunk first (FK constraint)
    store.upsert_document_chunk(
        DocumentChunk(
            chunk_id="dc1", slide_number=1, chunk_type="claim",
            text="Original", position_in_pdf=0,
        )
    )
    entry = CommonGroundEntry(
        cg_id="cg1",
        pdf_chunk_ref="dc1",
        original_text="Original claim text",
        negotiated_text="Revised claim text",
        proposed_by="agent",
        session_agreed="s1",
        version=1,
        timestamp=datetime.now(),
    )
    store.upsert_common_ground(entry)
    got = store.get_common_ground("cg1")
    assert got is not None
    assert got.negotiated_text == "Revised claim text"
    assert got.version == 1


def test_get_all_common_ground(store):
    for i in range(2):
        store.upsert_common_ground(
            CommonGroundEntry(
                cg_id=f"cg{i}",
                pdf_chunk_ref=None,
                original_text=None,
                negotiated_text=f"Entry {i}",
                proposed_by="user",
                session_agreed="s1",
                version=1,
                timestamp=datetime.now(),
            )
        )
    assert len(store.get_all_common_ground()) == 2
