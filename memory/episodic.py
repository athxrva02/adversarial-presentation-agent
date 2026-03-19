"""Episodic Memory — stores and retrieves session records and claim records."""
from __future__ import annotations

from datetime import datetime

from storage.schemas import ClaimRecord, SessionRecord
from storage.vector_store import VectorStore
from storage.relational_store import RelationalStore
from memory.recency import annotate_with_session_index, get_current_session_index

_CLAIMS_COLLECTION = "episodic_claims"
_SESSIONS_COLLECTION = "episodic_sessions"


class EpisodicMemory:
    def __init__(
        self, *, vector_store: VectorStore, relational_store: RelationalStore
    ) -> None:
        self._vec = vector_store
        self._rel = relational_store
    
    def clear(self) -> None:
        claim_ids = [r["claim_id"] for r in self._rel.get_recent_claims(limit=100000)]
        session_ids = [r["session_id"] for r in self._rel.get_all_sessions(limit=100000)]

        if claim_ids:
            self._vec.delete(ids=claim_ids, collection_name=_CLAIMS_COLLECTION)
        if session_ids:
            self._vec.delete(ids=session_ids, collection_name=_SESSIONS_COLLECTION)

        self._rel.delete_all_claims()
        self._rel.delete_all_sessions()

    def store_claim(self, claim: ClaimRecord) -> None:
        self._rel.insert_claim(claim)
        session_index = get_current_session_index()
        metadata = annotate_with_session_index({
            "session_id": claim.session_id,
            "turn_number": claim.turn_number,
            "alignment": claim.alignment.value,
        }, session_index)
        self._vec.upsert(
            documents=[claim.claim_text],
            metadatas=[metadata],
            ids=[claim.claim_id],
            collection_name=_CLAIMS_COLLECTION,
        )

    def store_session(
        self, session: SessionRecord, claims: list[ClaimRecord]
    ) -> None:
        self._rel.insert_session(session)
        for claim in claims:
            self.store_claim(claim)
        summary_text = (
            f"Session {session.session_id}. "
            f"Strengths: {', '.join(session.strengths)}. "
            f"Weaknesses: {', '.join(session.weaknesses)}."
        )
        session_index = get_current_session_index()
        metadata = annotate_with_session_index({
            "session_id": session.session_id,
            "overall_score": session.overall_score if session.overall_score is not None else 0.0,
        }, session_index)
        self._vec.upsert(
            documents=[summary_text],
            metadatas=[metadata],
            ids=[session.session_id],
            collection_name=_SESSIONS_COLLECTION,
        )

    def retrieve_claims(self, query: str, top_k: int) -> list[ClaimRecord]:
        results = self._vec.query(query, _CLAIMS_COLLECTION, top_k)
        claims: list[ClaimRecord] = []
        for r in results:
            claim_dict = self._rel.get_claim(r["id"])
            if claim_dict is not None:
                try:
                    claims.append(ClaimRecord(
                        claim_id=claim_dict["claim_id"],
                        session_id=claim_dict["session_id"],
                        turn_number=int(claim_dict["turn_number"]),
                        claim_text=claim_dict["claim_text"],
                        alignment=claim_dict.get("alignment", "novel"),
                        mapped_to_slide=claim_dict.get("mapped_to_slide"),
                        prior_conflict=claim_dict.get("prior_conflict"),
                        timestamp=claim_dict.get("timestamp", datetime.now()),
                    ))
                except Exception:
                    continue
        return claims

    def retrieve_sessions(self, query: str, top_k: int) -> list[SessionRecord]:
        results = self._vec.query(query, _SESSIONS_COLLECTION, top_k)
        sessions: list[SessionRecord] = []
        for r in results:
            session_dict = self._rel.get_session(r["id"])
            if session_dict is not None:
                try:
                    sessions.append(SessionRecord(
                        session_id=session_dict["session_id"],
                        timestamp=session_dict.get("timestamp", datetime.now()),
                        duration_seconds=float(session_dict.get("duration_seconds", 0.0)),
                        overall_score=session_dict.get("overall_score"),
                        strengths=session_dict.get("strengths", []),
                        weaknesses=session_dict.get("weaknesses", []),
                        claims_count=int(session_dict.get("claims_count", 0)),
                        contradictions_detected=int(session_dict.get("contradictions_detected", 0)),
                    ))
                except Exception:
                    continue
        return sessions
