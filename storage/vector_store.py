"""ChromaDB wrapper for vector storage and similarity search.

ChromaDB is imported lazily so the module can be loaded even when chromadb
is broken (e.g. pydantic v1 on Python 3.14).  Tests inject a mock client
via the ``client`` constructor parameter.
"""
from __future__ import annotations

import os
from typing import Any, Callable

from config import settings


class VectorStore:
    """Thin wrapper around ChromaDB for embedding-based retrieval.

    Accepts an optional *client* (any object with ``get_or_create_collection``)
    and *embedding_fn* for test injection.  When omitted, uses ChromaDB
    PersistentClient and SentenceTransformerEmbeddingFunction.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            import chromadb
            os.makedirs(settings.chroma_path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=settings.chroma_path)

        if embedding_fn is not None:
            self._ef = embedding_fn
        else:
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
            self._ef = SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model,
            )

        self._collections: dict[str, Any] = {}

    def get_or_create_collection(self, name: str) -> Any:
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                embedding_function=self._ef,
            )
        return self._collections[name]

    def upsert(
        self,
        collection_name: str,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        col = self.get_or_create_collection(collection_name)
        col.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(
        self,
        collection_name: str,
        query_text: str,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        col = self.get_or_create_collection(collection_name)
        if col.count() == 0:
            return []

        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(top_k, col.count()),
        }
        if where:
            kwargs["where"] = where

        raw = col.query(**kwargs)

        results: list[dict[str, Any]] = []
        for i in range(len(raw["ids"][0])):
            results.append({
                "id": raw["ids"][0][i],
                "document": raw["documents"][0][i],
                "metadata": raw["metadatas"][0][i],
                "distance": raw["distances"][0][i],
            })
        return results

    def delete(self, collection_name: str, ids: list[str]) -> None:
        col = self.get_or_create_collection(collection_name)
        col.delete(ids=ids)
