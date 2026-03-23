"""Common Ground Memory — versioned negotiated entries."""
from __future__ import annotations

from datetime import datetime
import logging

from storage.schemas import CommonGroundEntry
from storage.vector_store import VectorStore
from storage.relational_store import RelationalStore

_COLLECTION = "common_ground"
logger = logging.getLogger(__name__)


class CommonGroundMemory:
    def __init__(
        self, *, vector_store: VectorStore, relational_store: RelationalStore
    ) -> None:
        self._vec = vector_store
        self._rel = relational_store

    def clear(self) -> None:
        rows = self._rel.get_all_common_ground()
        ids = [r["cg_id"] for r in rows]
        if ids:
            try:
                self._vec.delete(ids=ids, collection_name=_COLLECTION)
            except Exception as exc:
                logger.warning("Failed to clear common_ground vectors; continuing with SQLite cleanup: %s", exc)
        self._rel.delete_all_common_ground()

    def store(self, entry: CommonGroundEntry) -> None:
        self._rel.upsert_common_ground(entry)
        try:
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
        except Exception as exc:
            logger.warning("Failed to upsert common_ground vector; SQLite record was still stored: %s", exc)

    def retrieve(self, query: str, top_k: int) -> list[CommonGroundEntry]:
        all_rows = self._rel.get_all_common_ground()
        all_cg = {d["cg_id"]: d for d in all_rows}
        entries: list[CommonGroundEntry] = []

        try:
            results = self._vec.query(query, _COLLECTION, top_k)
            for r in results:
                match = all_cg.get(r["id"])
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
        except Exception as exc:
            logger.warning(
                "Vector query failed for common_ground; falling back to SQLite recency order: %s",
                exc,
            )
            fallback: list[CommonGroundEntry] = []
            for match in all_rows[:top_k]:
                try:
                    fallback.append(CommonGroundEntry(
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
            return fallback

    def get_all(self) -> list[CommonGroundEntry]:
        return self._rel.get_all_common_ground()
