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
