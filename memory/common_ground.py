"""Common Ground Memory — versioned negotiated entries."""
from __future__ import annotations

from storage.schemas import CommonGroundEntry
from storage.vector_store import VectorStore
from storage.relational_store import RelationalStore

_COLLECTION = "common_ground"


class CommonGroundMemory:
    def __init__(
        self, *, vector_store: VectorStore, relational_store: RelationalStore
    ) -> None:
        self._vec = vector_store
        self._rel = relational_store

    def store(self, entry: CommonGroundEntry) -> None:
        self._rel.upsert_common_ground(entry)
        self._vec.upsert(
            _COLLECTION,
            ids=[entry.cg_id],
            documents=[entry.negotiated_text],
            metadatas=[{
                "cg_id": entry.cg_id,
                "version": entry.version,
                "proposed_by": entry.proposed_by,
            }],
        )

    def retrieve(self, query: str, top_k: int) -> list[CommonGroundEntry]:
        results = self._vec.query(_COLLECTION, query, top_k)
        entries: list[CommonGroundEntry] = []
        for r in results:
            entry = self._rel.get_common_ground(r["id"])
            if entry is not None:
                entries.append(entry)
        return entries

    def get_all(self) -> list[CommonGroundEntry]:
        return self._rel.get_all_common_ground()
