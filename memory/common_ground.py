"""Common Ground Memory — versioned negotiated entries."""
from __future__ import annotations

from datetime import datetime

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
            documents=[entry.negotiated_text],
            metadatas=[{
                "cg_id": entry.cg_id,
                "version": entry.version,
                "proposed_by": entry.proposed_by,
            }],
            ids=[entry.cg_id],
            collection_name=_COLLECTION,
        )

    def retrieve(self, query: str, top_k: int) -> list[CommonGroundEntry]:
        results = self._vec.query(query, _COLLECTION, top_k)
        entries: list[CommonGroundEntry] = []
        for r in results:
            all_cg = self._rel.get_all_common_ground()
            match = next((d for d in all_cg if d.get("cg_id") == r["id"]), None)
            if match:
                try:
                    entries.append(CommonGroundEntry(
                        cg_id=match["cg_id"],
                        pdf_chunk_ref=match.get("pdf_chunk_ref"),
                        original_text=match.get("original_text"),
                        negotiated_text=match.get("negotiated_text", ""),
                        proposed_by=match.get("proposed_by", "agent"),
                        session_agreed=match.get("session_agreed", ""),
                        version=int(match.get("version", 1)),
                        timestamp=match.get("timestamp", datetime.now()),
                    ))
                except Exception:
                    continue
        return entries

    def get_all(self) -> list[CommonGroundEntry]:
        return self._rel.get_all_common_ground()
