"""
ChromaDB wrapper for the Storage Layer.

Provides a single VectorStore class with:
- embed_and_store(chunks, collection_name)  → list[str]  (assigned embedding IDs)
- query(query_text, collection_name, top_k) → list[dict]
- delete(ids, collection_name)
- get_collection(name)                       → chromadb.Collection

Collections (one per memory type):
    "document_memory"    – DocumentChunk embeddings
    "episodic_claims"    – ClaimRecord embeddings
    "episodic_sessions"  – SessionRecord summary embeddings
    "semantic_patterns"  – SemanticPattern embeddings
    "common_ground"      – CommonGroundEntry embeddings

All collections share the same SentenceTransformer embedding function
(all-MiniLM-L6-v2 by default, overridable via config).

Design notes:
- Client is created lazily; path is read from config at construction time.
- IDs are always strings; ChromaDB requires this.
- Metadata stored alongside embeddings allows filtering without re-fetching SQLite.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# Canonical collection names used across the project
COLLECTION_DOCUMENT = "document_memory"
COLLECTION_EPISODIC_CLAIMS = "episodic_claims"
COLLECTION_EPISODIC_SESSIONS = "episodic_sessions"
COLLECTION_SEMANTIC = "semantic_patterns"
COLLECTION_COMMON_GROUND = "common_ground"

ALL_COLLECTIONS = [
    COLLECTION_DOCUMENT,
    COLLECTION_EPISODIC_CLAIMS,
    COLLECTION_EPISODIC_SESSIONS,
    COLLECTION_SEMANTIC,
    COLLECTION_COMMON_GROUND,
]


class VectorStore:
    """
    Thin wrapper around ChromaDB with a SentenceTransformer embedding function.
    """

    def __init__(self, chroma_path: Optional[str] = None, embedding_model: Optional[str] = None):
        try:
            from config import settings  # type: ignore
            self._chroma_path = chroma_path or settings.chroma_path
            self._embedding_model = embedding_model or settings.embedding_model
        except Exception:
            self._chroma_path = chroma_path or "./data/chroma"
            self._embedding_model = embedding_model or "all-MiniLM-L6-v2"

        self._client = None
        self._ef = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            import chromadb  # type: ignore
            self._client = chromadb.PersistentClient(path=self._chroma_path)
            logger.info("ChromaDB client initialised at '%s'.", self._chroma_path)
        return self._client

    def _get_ef(self):
        if self._ef is None:
            from chromadb.utils.embedding_functions import (  # type: ignore
                SentenceTransformerEmbeddingFunction,
            )
            self._ef = SentenceTransformerEmbeddingFunction(
                model_name=self._embedding_model
            )
            logger.info("Embedding function loaded: '%s'.", self._embedding_model)
        return self._ef

    # ------------------------------------------------------------------
    # Collection access
    # ------------------------------------------------------------------

    def get_collection(self, name: str):
        """Get or create a named ChromaDB collection."""
        client = self._get_client()
        ef = self._get_ef()
        return client.get_or_create_collection(name, embedding_function=ef)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def embed_and_store(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        collection_name: str,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Embed and store text documents with metadata.

        Args:
            documents:       List of text strings to embed.
            metadatas:       Parallel list of metadata dicts (same length as documents).
            collection_name: Target collection.
            ids:             Optional pre-assigned IDs. Auto-generated if None.

        Returns:
            List of string IDs used for the stored embeddings.
        """
        if not documents:
            return []

        if len(documents) != len(metadatas):
            raise ValueError("documents and metadatas must have the same length.")

        if ids is None:
            ids = [uuid4().hex for _ in documents]

        # ChromaDB metadata values must be str | int | float | bool
        safe_metadatas = [_sanitize_metadata(m) for m in metadatas]

        collection = self.get_collection(collection_name)
        collection.add(
            documents=documents,
            metadatas=safe_metadatas,
            ids=ids,
        )
        logger.debug(
            "Stored %d document(s) in collection '%s'.", len(documents), collection_name
        )
        return ids

    def query(
        self,
        query_text: str,
        collection_name: str,
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict[str, Any]]:
        """
        Query a collection by similarity.

        Args:
            query_text:      The search string; will be embedded automatically.
            collection_name: Collection to search.
            top_k:           Number of results to return.
            where:           Optional ChromaDB `where` filter dict.

        Returns:
            List of result dicts, each containing:
                {
                    "id":       str,
                    "document": str,
                    "metadata": dict,
                    "distance": float,
                }
            Ordered by ascending distance (most similar first).
        """
        collection = self.get_collection(collection_name)
        count = collection.count()
        if count == 0:
            return []

        effective_k = min(top_k, count)
        query_kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": effective_k,
        }
        if where:
            query_kwargs["where"] = where

        results = collection.query(**query_kwargs)

        # Unpack the nested ChromaDB response format
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {
                "id": id_,
                "document": doc,
                "metadata": meta or {},
                "distance": dist,
            }
            for id_, doc, meta, dist in zip(ids, docs, metas, distances)
        ]

    def delete(self, ids: list[str], collection_name: str) -> None:
        """Delete documents by ID from a collection."""
        if not ids:
            return
        collection = self.get_collection(collection_name)
        collection.delete(ids=ids)
        logger.debug("Deleted %d item(s) from '%s'.", len(ids), collection_name)

    def upsert(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
        collection_name: str,
    ) -> None:
        """
        Add or update documents (upsert semantics).
        Useful for semantic patterns and common ground which are versioned.
        """
        if not documents:
            return

        safe_metadatas = [_sanitize_metadata(m) for m in metadatas]
        collection = self.get_collection(collection_name)
        collection.upsert(
            documents=documents,
            metadatas=safe_metadatas,
            ids=ids,
        )
        logger.debug("Upserted %d item(s) in '%s'.", len(documents), collection_name)

    def count(self, collection_name: str) -> int:
        """Return number of documents in a collection."""
        collection = self.get_collection(collection_name)
        return collection.count()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_metadata(meta: dict) -> dict:
    """
    ChromaDB only accepts str, int, float, bool as metadata values.
    Convert anything else to str.
    """
    safe = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            safe[k] = v
        elif v is None:
            safe[k] = ""
        else:
            safe[k] = str(v)
    return safe


# Singleton for convenience (import and use directly)
_default_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return the default singleton VectorStore."""
    global _default_store
    if _default_store is None:
        _default_store = VectorStore()
    return _default_store
