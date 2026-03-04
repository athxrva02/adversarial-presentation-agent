"""Tests for memory.common_ground.CommonGroundMemory."""
from datetime import datetime

import pytest
from memory.common_ground import CommonGroundMemory
from storage.schemas import CommonGroundEntry


@pytest.fixture
def cg_mem(vec_store, rel_store):
    return CommonGroundMemory(vector_store=vec_store, relational_store=rel_store)


def _entry(i: int = 0) -> CommonGroundEntry:
    return CommonGroundEntry(
        cg_id=f"cg{i}",
        pdf_chunk_ref=None,
        original_text=f"Original text {i}",
        negotiated_text=f"Revised negotiated text {i}",
        proposed_by="agent",
        session_agreed="s1",
        version=1,
        timestamp=datetime(2025, 6, 1, 12, 0, 0),
    )


def test_store_and_retrieve(cg_mem):
    cg_mem.store(_entry(0))
    results = cg_mem.retrieve("negotiated", top_k=1)
    assert len(results) == 1
    assert results[0].cg_id == "cg0"


def test_retrieve_top_k(cg_mem):
    for i in range(4):
        cg_mem.store(_entry(i))
    results = cg_mem.retrieve("text", top_k=2)
    assert len(results) == 2


def test_get_all(cg_mem):
    for i in range(3):
        cg_mem.store(_entry(i))
    assert len(cg_mem.get_all()) == 3


def test_empty_retrieval(cg_mem):
    assert cg_mem.retrieve("anything", top_k=5) == []


def test_version_update(cg_mem):
    cg_mem.store(_entry(0))
    updated = CommonGroundEntry(
        cg_id="cg0",
        pdf_chunk_ref=None,
        original_text="Original text 0",
        negotiated_text="Updated negotiated text",
        proposed_by="user",
        session_agreed="s2",
        version=2,
        timestamp=datetime(2025, 6, 2, 12, 0, 0),
    )
    cg_mem.store(updated)
    got = cg_mem.get_all()
    assert len(got) == 1
    assert got[0].version == 2
    assert got[0].negotiated_text == "Updated negotiated text"
