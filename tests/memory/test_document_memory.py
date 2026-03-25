"""Tests for memory.document.DocumentMemory."""
import pytest
from memory.document import DocumentMemory
from storage.schemas import DocumentChunk


@pytest.fixture
def doc_mem(vec_store, rel_store):
    return DocumentMemory(vector_store=vec_store, relational_store=rel_store)


def _chunk(i: int = 0) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"dc{i}",
        slide_number=i + 1,
        chunk_type="claim",
        text=f"Our method improves metric {i} significantly.",
        position_in_pdf=i * 10,
    )


def test_store_and_retrieve(doc_mem):
    doc_mem.store(_chunk(0))
    results = doc_mem.retrieve("method improves", top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "dc0"


def test_retrieve_top_k(doc_mem):
    for i in range(5):
        doc_mem.store(_chunk(i))
    results = doc_mem.retrieve("method", top_k=2)
    assert len(results) == 2


def test_retrieve_empty_returns_empty(doc_mem):
    assert doc_mem.retrieve("anything", top_k=5) == []


def test_fields_preserved(doc_mem):
    chunk = _chunk(7)
    doc_mem.store(chunk)
    got = doc_mem.retrieve("metric 7", top_k=1)
    assert len(got) == 1
    assert got[0].slide_number == 8
    assert got[0].chunk_type == "claim"
    assert got[0].position_in_pdf == 70


def test_get_all(doc_mem):
    for i in range(3):
        doc_mem.store(_chunk(i))
    assert len(doc_mem.get_all()) == 3


def test_list_chunks_for_questioning_diversifies_types(doc_mem):
    doc_mem.store(
        [
            DocumentChunk(
                chunk_id="def1",
                slide_number=1,
                chunk_type="definition",
                text="Accessibility means reducing planning friction.",
                position_in_pdf=0,
            ),
            DocumentChunk(
                chunk_id="ev1",
                slide_number=2,
                chunk_type="evidence",
                text="Survey data shows safety concerns dominate choice.",
                position_in_pdf=10,
            ),
            DocumentChunk(
                chunk_id="cl1",
                slide_number=3,
                chunk_type="claim",
                text="TravelGo improves discovery for solo travelers.",
                position_in_pdf=20,
            ),
        ]
    )

    results = doc_mem.list_chunks_for_questioning(limit=3)
    chunk_types = [r.chunk_type for r in results]
    assert len(results) == 3
    assert "definition" in chunk_types
    assert "evidence" in chunk_types


def test_list_chunks_for_questioning_respects_exclusions(doc_mem):
    for i in range(3):
        doc_mem.store(_chunk(i))

    results = doc_mem.list_chunks_for_questioning(limit=3, exclude_chunk_ids=["dc0"])
    result_ids = {r.chunk_id for r in results}
    assert "dc0" not in result_ids
