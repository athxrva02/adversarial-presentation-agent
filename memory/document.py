"""
Document Memory — stores and retrieves PDF chunks.

This is the memory type that backs the PDF ingestion pipeline.

Public API:
    DocumentMemory.store(chunks: list[DocumentChunk]) -> None
    DocumentMemory.retrieve(query: str, top_k: int) -> list[DocumentChunk]
    DocumentMemory.clear() -> None
    DocumentMemory.count() -> int

Pipeline (store):
    chunk list → embed into ChromaDB → write metadata to SQLite

Pipeline (retrieve):
    query string → vector similarity search → reconstruct DocumentChunk objects

Design notes:
- embedding_id on DocumentChunk is the ChromaDB document ID (same as chunk_id for simplicity).
- Both stores are kept in sync: clear() wipes both.
- Metadata stored in ChromaDB allows filtering by slide_number or chunk_type.
"""

from __future__ import annotations

import logging
from typing import Optional

from storage.schemas import DocumentChunk
from storage.vector_store import VectorStore, COLLECTION_DOCUMENT, get_vector_store
from storage.relational_store import RelationalStore, get_relational_store

logger = logging.getLogger(__name__)


class DocumentMemory:
    """Read/write access to document (PDF chunk) memory."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        relational_store: Optional[RelationalStore] = None,
    ):
        self._vs = vector_store or get_vector_store()
        self._rs = relational_store or get_relational_store()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store(self, chunks: list[DocumentChunk]) -> None:
        """
        Embed and persist a list of DocumentChunk objects.

        - Uses chunk_id as the ChromaDB embedding ID (deterministic, dedup-friendly).
        - Writes metadata to SQLite for structured lookups.
        - Skips chunks already in SQLite (INSERT OR IGNORE).
        """
        if not chunks:
            return

        documents = [c.text for c in chunks]
        metadatas = [
            {
                "chunk_id": c.chunk_id,
                "slide_number": c.slide_number if c.slide_number is not None else -1,
                "chunk_type": c.chunk_type,
                "position_in_pdf": c.position_in_pdf,
            }
            for c in chunks
        ]
        ids = [c.chunk_id for c in chunks]

        # Embed into ChromaDB (upsert so re-ingesting a PDF is safe)
        self._vs.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            collection_name=COLLECTION_DOCUMENT,
        )

        # Stamp embedding_id before writing to SQLite
        stamped = [
            c.model_copy(update={"embedding_id": c.chunk_id}) for c in chunks
        ]

        # Write to relational store
        self._rs.insert_chunks(stamped)

        logger.info("Stored %d document chunks.", len(chunks))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        slide_number: Optional[int] = None,
        chunk_type: Optional[str] = None,
    ) -> list[DocumentChunk]:
        """
        Retrieve the most relevant DocumentChunk objects for a query.

        Args:
            query:        Natural-language query string.
            top_k:        Maximum number of chunks to return.
            slide_number: If set, restrict results to this slide.
            chunk_type:   If set, restrict by chunk type (claim | evidence | …).

        Returns:
            List of DocumentChunk objects ordered by relevance (closest first).
        """
        where: dict = {}
        if slide_number is not None:
            where["slide_number"] = slide_number
        if chunk_type is not None:
            where["chunk_type"] = chunk_type

        results = self._vs.query(
            query_text=query,
            collection_name=COLLECTION_DOCUMENT,
            top_k=top_k,
            where=where if where else None,
        )

        chunks: list[DocumentChunk] = []
        for r in results:
            meta = r.get("metadata", {})
            chunks.append(
                DocumentChunk(
                    chunk_id=meta.get("chunk_id", r["id"]),
                    slide_number=meta.get("slide_number") if meta.get("slide_number", -1) != -1 else None,
                    chunk_type=meta.get("chunk_type", "claim"),
                    text=r.get("document", ""),
                    position_in_pdf=int(meta.get("position_in_pdf", 0)),
                    embedding_id=r["id"],
                )
            )

        return chunks

    # Context expansion change: expose document chunks for document-driven questioning
    def list_chunks_for_questioning(
        self,
        limit: int = 6,
        exclude_chunk_ids: Optional[list[str]] = None,
    ) -> list[DocumentChunk]:
        """
        Return document chunks for question generation independent of similarity search.

        Strategy:
        - exclude chunks already used for document-driven questions
        - diversify by chunk_type first
        - then spread coverage across the PDF by position
        """
        exclude = set(exclude_chunk_ids or [])
        rows = [
            r for r in self._rs.get_all_chunks()
            if r.get("chunk_id") not in exclude
        ]
        if not rows:
            return []

        chosen: list[dict] = []
        chosen_ids: set[str] = set()

        type_priority = ("definition", "evidence", "claim", "conclusion")
        for chunk_type in type_priority:
            for row in rows:
                if str(row.get("chunk_type", "")).strip().lower() != chunk_type:
                    continue
                cid = str(row.get("chunk_id", "") or "")
                if not cid or cid in chosen_ids:
                    continue
                chosen.append(row)
                chosen_ids.add(cid)
                break
            if len(chosen) >= limit:
                break

        if len(chosen) < limit:
            total = len(rows)
            step = max(1, total // max(limit, 1))
            for idx in range(0, total, step):
                row = rows[idx]
                cid = str(row.get("chunk_id", "") or "")
                if not cid or cid in chosen_ids:
                    continue
                chosen.append(row)
                chosen_ids.add(cid)
                if len(chosen) >= limit:
                    break

        if len(chosen) < limit:
            for row in rows:
                cid = str(row.get("chunk_id", "") or "")
                if not cid or cid in chosen_ids:
                    continue
                chosen.append(row)
                chosen_ids.add(cid)
                if len(chosen) >= limit:
                    break

        return [
            DocumentChunk(
                chunk_id=row["chunk_id"],
                slide_number=row.get("slide_number"),
                chunk_type=row.get("chunk_type", "claim"),
                text=row.get("text", ""),
                position_in_pdf=int(row.get("position_in_pdf", 0)),
                embedding_id=row.get("embedding_id"),
                source_file=row.get("source_file"),
            )
            for row in chosen
        ]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all document chunks from both stores."""
        # Get all IDs from SQLite and delete from ChromaDB
        rows = self._rs.get_all_chunks()
        ids = [r["chunk_id"] for r in rows]
        if ids:
            self._vs.delete(ids=ids, collection_name=COLLECTION_DOCUMENT)
        self._rs.delete_all_chunks()
        logger.info("Cleared all document chunks.")

    def count(self) -> int:
        """Return the total number of stored chunks."""
        return self._vs.count(COLLECTION_DOCUMENT)
