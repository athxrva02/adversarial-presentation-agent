"""Document Memory — stores and retrieves PDF chunks via vector + relational stores."""
from __future__ import annotations

from storage.schemas import DocumentChunk
from storage.vector_store import VectorStore
from storage.relational_store import RelationalStore

_COLLECTION = "document_memory"


class DocumentMemory:
    def __init__(
        self, *, vector_store: VectorStore, relational_store: RelationalStore
    ) -> None:
        self._vec = vector_store
        self._rel = relational_store

    def store(self, chunk: DocumentChunk) -> None:
        self._rel.upsert_document_chunk(chunk)
        self._vec.upsert(
            _COLLECTION,
            ids=[chunk.chunk_id],
            documents=[chunk.text],
            metadatas=[{
                "chunk_id": chunk.chunk_id,
                "chunk_type": chunk.chunk_type,
                "slide_number": chunk.slide_number if chunk.slide_number is not None else -1,
                "position_in_pdf": chunk.position_in_pdf,
            }],
        )

    def retrieve(self, query: str, top_k: int) -> list[DocumentChunk]:
        results = self._vec.query(_COLLECTION, query, top_k)
        chunks: list[DocumentChunk] = []
        for r in results:
            chunk = self._rel.get_document_chunk(r["id"])
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def get_all(self) -> list[DocumentChunk]:
        return self._rel.get_all_document_chunks()
