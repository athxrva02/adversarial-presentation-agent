"""Shared fixtures for memory tests."""
import pytest
from tests.storage.test_vector_store import _FakeClient, _mock_embedding_fn
from storage.vector_store import VectorStore
from storage.relational_store import RelationalStore


@pytest.fixture
def vec_store():
    return VectorStore(client=_FakeClient(), embedding_fn=_mock_embedding_fn)


@pytest.fixture
def rel_store(tmp_path):
    return RelationalStore(db_path=str(tmp_path / "test.db"))
