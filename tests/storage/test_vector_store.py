"""Tests for storage.vector_store.VectorStore using an in-memory mock client.

ChromaDB is broken on Python 3.14 (pydantic v1), so we use a lightweight
fake that implements the same collection interface.
"""
import pytest
from storage.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Lightweight in-memory ChromaDB substitute
# ---------------------------------------------------------------------------

class _FakeCollection:
    """Mimics chromadb.Collection with in-memory storage and trivial ranking."""

    def __init__(self, name: str, embedding_function=None):
        self.name = name
        self._ef = embedding_function
        self._store: dict[str, dict] = {}  # id -> {document, metadata, embedding}

    def count(self) -> int:
        return len(self._store)

    def upsert(self, ids, documents, metadatas):
        embeddings = self._ef(documents) if self._ef else [[0.0]] * len(ids)
        for i, doc_id in enumerate(ids):
            self._store[doc_id] = {
                "document": documents[i],
                "metadata": metadatas[i],
                "embedding": embeddings[i],
            }

    def query(self, query_texts, n_results, where=None):
        items = list(self._store.items())
        if where:
            items = [
                (k, v) for k, v in items
                if all(v["metadata"].get(wk) == wv for wk, wv in where.items())
            ]
        # Trivial distance: index order
        items = items[:n_results]
        return {
            "ids": [[k for k, _ in items]],
            "documents": [[v["document"] for _, v in items]],
            "metadatas": [[v["metadata"] for _, v in items]],
            "distances": [[float(i) for i, _ in enumerate(items)]],
        }

    def delete(self, ids):
        for doc_id in ids:
            self._store.pop(doc_id, None)


class _FakeClient:
    """Mimics chromadb.EphemeralClient."""

    def __init__(self):
        self._collections: dict[str, _FakeCollection] = {}

    def get_or_create_collection(self, name, embedding_function=None):
        if name not in self._collections:
            self._collections[name] = _FakeCollection(
                name, embedding_function=embedding_function,
            )
        return self._collections[name]


def _mock_embedding_fn(texts):
    return [[float(i) / 100] * 384 for i, _ in enumerate(texts)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    return VectorStore(client=_FakeClient(), embedding_fn=_mock_embedding_fn)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_upsert_and_query(store):
    store.upsert(
        "document_memory",
        ids=["c1"],
        documents=["Neural networks improve accuracy"],
        metadatas=[{"chunk_type": "claim"}],
    )
    results = store.query("document_memory", "neural networks", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "c1"


def test_query_top_k(store):
    for i in range(5):
        store.upsert(
            "document_memory",
            ids=[f"c{i}"],
            documents=[f"Document chunk number {i}"],
            metadatas=[{"chunk_type": "claim"}],
        )
    results = store.query("document_memory", "document chunk", top_k=2)
    assert len(results) == 2


def test_upsert_deduplicates(store):
    store.upsert(
        "document_memory",
        ids=["c1"],
        documents=["Version 1"],
        metadatas=[{"chunk_type": "claim"}],
    )
    store.upsert(
        "document_memory",
        ids=["c1"],
        documents=["Version 2"],
        metadatas=[{"chunk_type": "claim"}],
    )
    col = store.get_or_create_collection("document_memory")
    assert col.count() == 1
    results = store.query("document_memory", "version", top_k=1)
    assert results[0]["document"] == "Version 2"


def test_delete(store):
    store.upsert(
        "document_memory",
        ids=["c1"],
        documents=["To be deleted"],
        metadatas=[{"chunk_type": "claim"}],
    )
    store.delete("document_memory", ids=["c1"])
    results = store.query("document_memory", "deleted", top_k=1)
    assert len(results) == 0


def test_query_empty_collection(store):
    results = store.query("document_memory", "anything", top_k=5)
    assert results == []


def test_collections_are_named(store):
    col1 = store.get_or_create_collection("document_memory")
    col2 = store.get_or_create_collection("document_memory")
    assert col1.name == col2.name == "document_memory"


def test_query_with_where_filter(store):
    store.upsert(
        "document_memory",
        ids=["c1", "c2"],
        documents=["Claim text", "Evidence text"],
        metadatas=[{"chunk_type": "claim"}, {"chunk_type": "evidence"}],
    )
    results = store.query(
        "document_memory", "text", top_k=5, where={"chunk_type": "claim"}
    )
    assert len(results) == 1
    assert results[0]["metadata"]["chunk_type"] == "claim"
