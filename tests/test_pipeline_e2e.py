"""
End-to-end pipeline test: PDF upload → chunk → embed → store → query.

This is the Member C integration test that validates the full pipeline
described in the technical plan.

Requirements:
- PyMuPDF  (pip install PyMuPDF)
- chromadb  (pip install chromadb)
- sentence-transformers  (pip install sentence-transformers)

Runs against temporary on-disk stores (cleaned up after each test).

Run with:
    cd adversarial-presentation-agent
    pytest tests/test_pipeline_e2e.py -v
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_stores(tmp_path):
    """
    Provide temporary ChromaDB and SQLite paths, patch config, and clean up.
    """
    chroma_dir = str(tmp_path / "chroma")
    sqlite_path = str(tmp_path / "db" / "agent.db")
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)

    # Patch config settings before any store is created
    from unittest.mock import patch
    settings_patch = {
        "chroma_path": chroma_dir,
        "sqlite_path": sqlite_path,
        "embedding_model": "all-MiniLM-L6-v2",
        "max_chunk_tokens": 128,
        "chunk_overlap_tokens": 16,
        "retrieval_top_k": 5,
    }

    # Reset singletons between tests
    import storage.vector_store as vs_mod
    import storage.relational_store as rs_mod
    import memory.document as dm_mod

    vs_mod._default_store = None
    rs_mod._default_store = None

    with patch.multiple("config.settings", **settings_patch):
        yield {
            "chroma_path": chroma_dir,
            "sqlite_path": sqlite_path,
        }

    # Teardown singletons
    vs_mod._default_store = None
    rs_mod._default_store = None


def _make_test_pdf(tmp_path, pages: list[str]) -> str:
    """Create a real PDF file from a list of page text strings."""
    fitz = pytest.importorskip("fitz", reason="PyMuPDF required")
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    pdf_path = str(tmp_path / "test.pdf")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestPDFIngestionPipeline:

    def test_full_pipeline_upload_chunk_embed_store_query(self, tmp_stores, tmp_path):
        """
        Full pipeline: parse PDF → embed → store → query → get back DocumentChunks.
        """
        pytest.importorskip("fitz", reason="PyMuPDF required")
        pytest.importorskip("chromadb", reason="chromadb required")

        from interaction.pdf_parser import ingest_pdf
        from memory.document import DocumentMemory
        from storage.vector_store import VectorStore
        from storage.relational_store import RelationalStore

        vs = VectorStore(
            chroma_path=tmp_stores["chroma_path"],
            embedding_model="all-MiniLM-L6-v2",
        )
        rs = RelationalStore(db_path=tmp_stores["sqlite_path"])
        doc_mem = DocumentMemory(vector_store=vs, relational_store=rs)

        # Step 1: ingest PDF
        pdf_path = _make_test_pdf(tmp_path, [
            "We argue that renewable energy adoption reduces carbon emissions significantly.",
            "According to studies, solar power costs have decreased by 90% since 2010.",
            "Therefore, we conclude that policy support is essential for the energy transition.",
        ])
        chunks = ingest_pdf(pdf_path)
        assert len(chunks) > 0, "PDF parser must produce at least one chunk"

        # Step 2: store (embed + persist)
        doc_mem.store(chunks)

        # Step 3: verify count
        count = doc_mem.count()
        assert count > 0, "Vector store must have entries after storing chunks"

        # Step 4: query
        results = doc_mem.retrieve("renewable energy policy", top_k=3)
        assert len(results) > 0, "Query must return results"
        for r in results:
            assert r.text.strip(), "Returned chunk must have non-empty text"
            assert r.chunk_id, "Returned chunk must have a chunk_id"

    def test_retrieved_chunks_are_relevant(self, tmp_stores, tmp_path):
        """
        The most similar chunk should be semantically close to the query.
        """
        pytest.importorskip("fitz", reason="PyMuPDF required")
        pytest.importorskip("chromadb", reason="chromadb required")

        from interaction.pdf_parser import ingest_pdf
        from memory.document import DocumentMemory
        from storage.vector_store import VectorStore
        from storage.relational_store import RelationalStore

        vs = VectorStore(chroma_path=tmp_stores["chroma_path"], embedding_model="all-MiniLM-L6-v2")
        rs = RelationalStore(db_path=tmp_stores["sqlite_path"])
        doc_mem = DocumentMemory(vector_store=vs, relational_store=rs)

        pdf_path = _make_test_pdf(tmp_path, [
            "Deep learning models have revolutionised computer vision tasks.",
            "The French Revolution began in 1789 and transformed European politics.",
            "Photosynthesis converts sunlight into glucose in plant cells.",
        ])
        chunks = ingest_pdf(pdf_path)
        doc_mem.store(chunks)

        results = doc_mem.retrieve("neural networks and image recognition", top_k=1)
        assert results, "Should return at least one result"
        top = results[0]
        assert "deep learning" in top.text.lower() or "vision" in top.text.lower(), (
            f"Top result should be about deep learning/vision, got: {top.text!r}"
        )

    def test_relational_store_persists_chunks(self, tmp_stores, tmp_path):
        """
        SQLite should contain the same chunks as the vector store after ingestion.
        """
        pytest.importorskip("fitz", reason="PyMuPDF required")
        pytest.importorskip("chromadb", reason="chromadb required")

        from interaction.pdf_parser import ingest_pdf
        from memory.document import DocumentMemory
        from storage.vector_store import VectorStore
        from storage.relational_store import RelationalStore

        vs = VectorStore(chroma_path=tmp_stores["chroma_path"], embedding_model="all-MiniLM-L6-v2")
        rs = RelationalStore(db_path=tmp_stores["sqlite_path"])
        doc_mem = DocumentMemory(vector_store=vs, relational_store=rs)

        pdf_path = _make_test_pdf(tmp_path, [
            "We claim that testing is an important part of software engineering.",
        ])
        chunks = ingest_pdf(pdf_path)
        doc_mem.store(chunks)

        db_rows = rs.get_all_chunks()
        assert len(db_rows) == len(chunks), (
            f"SQLite should have {len(chunks)} rows, got {len(db_rows)}"
        )
        chunk_ids_in_db = {r["chunk_id"] for r in db_rows}
        chunk_ids_from_parser = {c.chunk_id for c in chunks}
        assert chunk_ids_from_parser == chunk_ids_in_db

    def test_clear_removes_all_chunks(self, tmp_stores, tmp_path):
        pytest.importorskip("fitz", reason="PyMuPDF required")
        pytest.importorskip("chromadb", reason="chromadb required")

        from interaction.pdf_parser import ingest_pdf
        from memory.document import DocumentMemory
        from storage.vector_store import VectorStore
        from storage.relational_store import RelationalStore

        vs = VectorStore(chroma_path=tmp_stores["chroma_path"], embedding_model="all-MiniLM-L6-v2")
        rs = RelationalStore(db_path=tmp_stores["sqlite_path"])
        doc_mem = DocumentMemory(vector_store=vs, relational_store=rs)

        pdf_path = _make_test_pdf(tmp_path, ["We argue that cleanup is essential."])
        chunks = ingest_pdf(pdf_path)
        doc_mem.store(chunks)

        assert doc_mem.count() > 0
        doc_mem.clear()
        assert doc_mem.count() == 0
        assert rs.get_all_chunks() == []

    def test_reingest_same_pdf_is_idempotent(self, tmp_stores, tmp_path):
        """
        Ingesting the same PDF twice should not duplicate chunks.
        """
        pytest.importorskip("fitz", reason="PyMuPDF required")
        pytest.importorskip("chromadb", reason="chromadb required")

        from interaction.pdf_parser import ingest_pdf
        from memory.document import DocumentMemory
        from storage.vector_store import VectorStore
        from storage.relational_store import RelationalStore

        vs = VectorStore(chroma_path=tmp_stores["chroma_path"], embedding_model="all-MiniLM-L6-v2")
        rs = RelationalStore(db_path=tmp_stores["sqlite_path"])
        doc_mem = DocumentMemory(vector_store=vs, relational_store=rs)

        pdf_path = _make_test_pdf(tmp_path, ["We argue that idempotency is a virtue."])
        chunks = ingest_pdf(pdf_path)

        doc_mem.store(chunks)
        count_after_first = doc_mem.count()

        doc_mem.store(chunks)  # Second ingest (same chunk IDs)
        count_after_second = doc_mem.count()

        assert count_after_first == count_after_second, (
            "Re-ingesting the same PDF should not create duplicate embeddings"
        )

    def test_multipage_pdf_all_slides_stored(self, tmp_stores, tmp_path):
        pytest.importorskip("fitz", reason="PyMuPDF required")
        pytest.importorskip("chromadb", reason="chromadb required")

        from interaction.pdf_parser import ingest_pdf
        from memory.document import DocumentMemory
        from storage.vector_store import VectorStore
        from storage.relational_store import RelationalStore

        vs = VectorStore(chroma_path=tmp_stores["chroma_path"], embedding_model="all-MiniLM-L6-v2")
        rs = RelationalStore(db_path=tmp_stores["sqlite_path"])
        doc_mem = DocumentMemory(vector_store=vs, relational_store=rs)

        pdf_path = _make_test_pdf(tmp_path, [
            "Slide one: we claim that AI is transforming healthcare.",
            "Slide two: evidence shows a 20% reduction in diagnostic errors.",
            "Slide three: therefore we conclude investment is warranted.",
        ])
        chunks = ingest_pdf(pdf_path)
        doc_mem.store(chunks)

        db_rows = rs.get_all_chunks()
        slide_numbers = {r["slide_number"] for r in db_rows if r["slide_number"]}
        assert len(slide_numbers) >= 2, "Chunks from multiple slides should be stored"
