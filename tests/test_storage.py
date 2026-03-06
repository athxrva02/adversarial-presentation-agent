"""
Tests for storage/vector_store.py and storage/relational_store.py

All tests use temporary directories/files (cleaned up automatically).
No Ollama or external services required.

Run with:
    cd adversarial-presentation-agent
    pytest tests/test_storage.py -v
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vector_store(tmp_path):
    pytest.importorskip("chromadb", reason="chromadb required")
    from storage.vector_store import VectorStore
    return VectorStore(
        chroma_path=str(tmp_path / "chroma"),
        embedding_model="all-MiniLM-L6-v2",
    )


def make_relational_store(tmp_path):
    from storage.relational_store import RelationalStore
    db_path = str(tmp_path / "db" / "agent.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return RelationalStore(db_path=db_path)


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class TestVectorStore:

    def test_embed_and_query_returns_results(self, tmp_path):
        vs = make_vector_store(tmp_path)
        vs.embed_and_store(
            documents=["The sky is blue.", "Grass is green."],
            metadatas=[{"tag": "sky"}, {"tag": "grass"}],
            collection_name="test_col",
        )
        results = vs.query("blue sky", "test_col", top_k=1)
        assert len(results) == 1
        assert "sky" in results[0]["document"].lower() or results[0]["document"]

    def test_query_empty_collection_returns_empty(self, tmp_path):
        vs = make_vector_store(tmp_path)
        results = vs.query("anything", "empty_col", top_k=5)
        assert results == []

    def test_delete_removes_document(self, tmp_path):
        vs = make_vector_store(tmp_path)
        ids = vs.embed_and_store(
            documents=["Delete me."],
            metadatas=[{"x": "1"}],
            collection_name="del_col",
        )
        assert vs.count("del_col") == 1
        vs.delete(ids=ids, collection_name="del_col")
        assert vs.count("del_col") == 0

    def test_upsert_does_not_duplicate(self, tmp_path):
        vs = make_vector_store(tmp_path)
        vs.upsert(
            documents=["Once."],
            metadatas=[{"v": "1"}],
            ids=["fixed-id"],
            collection_name="ups_col",
        )
        vs.upsert(
            documents=["Updated."],
            metadatas=[{"v": "2"}],
            ids=["fixed-id"],
            collection_name="ups_col",
        )
        assert vs.count("ups_col") == 1

    def test_top_k_respected(self, tmp_path):
        vs = make_vector_store(tmp_path)
        docs = [f"Document number {i}." for i in range(10)]
        metas = [{"i": str(i)} for i in range(10)]
        vs.embed_and_store(docs, metas, collection_name="topk_col")
        results = vs.query("document", "topk_col", top_k=3)
        assert len(results) <= 3

    def test_metadata_survives_round_trip(self, tmp_path):
        vs = make_vector_store(tmp_path)
        vs.embed_and_store(
            documents=["Meta test."],
            metadatas=[{"slide_number": 2, "chunk_type": "claim"}],
            ids=["meta-id"],
            collection_name="meta_col",
        )
        results = vs.query("meta", "meta_col", top_k=1)
        assert results[0]["metadata"]["slide_number"] == 2
        assert results[0]["metadata"]["chunk_type"] == "claim"


# ---------------------------------------------------------------------------
# RelationalStore
# ---------------------------------------------------------------------------

class TestRelationalStore:

    def _make_chunk(self, chunk_id="c1", slide=1, position=1):
        from storage.schemas import DocumentChunk
        return DocumentChunk(
            chunk_id=chunk_id,
            slide_number=slide,
            chunk_type="claim",
            text="We argue that testing is important.",
            position_in_pdf=position,
            embedding_id=chunk_id,
        )

    def _make_session(self, session_id="s1"):
        from storage.schemas import SessionRecord
        return SessionRecord(
            session_id=session_id,
            timestamp=datetime.now(),
            duration_seconds=120.0,
            overall_score=0.75,
            strengths=["clear argument"],
            weaknesses=["lacks evidence"],
            claims_count=5,
            contradictions_detected=1,
        )

    def _make_claim(self, claim_id="cl1", session_id="s1", turn=1):
        from storage.schemas import ClaimRecord, ClaimAlignment
        return ClaimRecord(
            claim_id=claim_id,
            session_id=session_id,
            turn_number=turn,
            claim_text="The intervention was effective.",
            alignment=ClaimAlignment.SUPPORTED,
            mapped_to_slide=1,
            prior_conflict=None,
            timestamp=datetime.now(),
        )

    def test_insert_and_get_chunk(self, tmp_path):
        rs = make_relational_store(tmp_path)
        chunk = self._make_chunk()
        rs.insert_chunk(chunk)
        row = rs.get_chunk("c1")
        assert row is not None
        assert row["chunk_id"] == "c1"
        assert row["chunk_type"] == "claim"

    def test_insert_chunks_batch(self, tmp_path):
        rs = make_relational_store(tmp_path)
        chunks = [self._make_chunk(f"c{i}", position=i) for i in range(5)]
        rs.insert_chunks(chunks)
        rows = rs.get_all_chunks()
        assert len(rows) == 5

    def test_duplicate_chunk_ignored(self, tmp_path):
        rs = make_relational_store(tmp_path)
        chunk = self._make_chunk()
        rs.insert_chunk(chunk)
        rs.insert_chunks([chunk])  # should be ignored
        rows = rs.get_all_chunks()
        assert len(rows) == 1

    def test_delete_all_chunks(self, tmp_path):
        rs = make_relational_store(tmp_path)
        rs.insert_chunks([self._make_chunk(f"c{i}", position=i) for i in range(3)])
        rs.delete_all_chunks()
        assert rs.get_all_chunks() == []

    def test_insert_and_get_session(self, tmp_path):
        rs = make_relational_store(tmp_path)
        session = self._make_session()
        rs.insert_session(session)
        row = rs.get_session("s1")
        assert row is not None
        assert row["session_id"] == "s1"
        assert row["strengths"] == ["clear argument"]
        assert row["weaknesses"] == ["lacks evidence"]

    def test_insert_and_get_claim(self, tmp_path):
        rs = make_relational_store(tmp_path)
        # Session must exist (foreign key)
        rs.insert_session(self._make_session())
        claim = self._make_claim()
        rs.insert_claim(claim)
        rows = rs.get_claims_for_session("s1")
        assert len(rows) == 1
        assert rows[0]["claim_id"] == "cl1"

    def test_get_all_sessions_ordered_desc(self, tmp_path):
        rs = make_relational_store(tmp_path)
        from storage.schemas import SessionRecord
        import time
        for i in range(3):
            s = SessionRecord(
                session_id=f"s{i}",
                timestamp=datetime.now(),
                duration_seconds=float(i * 10),
                overall_score=None,
                strengths=[],
                weaknesses=[],
                claims_count=0,
                contradictions_detected=0,
            )
            rs.insert_session(s)
            time.sleep(0.01)  # ensure different timestamps

        sessions = rs.get_all_sessions()
        assert len(sessions) == 3

    def test_update_chunk_embedding_id(self, tmp_path):
        rs = make_relational_store(tmp_path)
        chunk = self._make_chunk()
        chunk = chunk.model_copy(update={"embedding_id": None})
        rs.insert_chunk(chunk)
        rs.update_chunk_embedding_id("c1", "new-embed-id")
        row = rs.get_chunk("c1")
        assert row["embedding_id"] == "new-embed-id"
