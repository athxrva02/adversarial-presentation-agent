"""Episodic Memory — stores and retrieves session records and claim records."""
from __future__ import annotations

from datetime import datetime

from storage.schemas import ClaimRecord, SessionRecord
from storage.vector_store import VectorStore
from storage.relational_store import RelationalStore

_CLAIMS_COLLECTION = "episodic_claims"
_SESSIONS_COLLECTION = "episodic_sessions"


class EpisodicMemory:
    def __init__(
        self, *, vector_store: VectorStore, relational_store: RelationalStore
    ) -> None:
        self._vec = vector_store
        self._rel = relational_store

    def store_claim(self, claim: ClaimRecord) -> None:
        self._rel.insert_claim(claim)
        self._vec.upsert(
            documents=[claim.claim_text],
            metadatas=[{
                "session_id": claim.session_id,
                "turn_number": claim.turn_number,
                "alignment": claim.alignment.value,
            }],
            ids=[claim.claim_id],
            collection_name=_CLAIMS_COLLECTION,
        )

    def store_session(
        self, session: SessionRecord, claims: list[ClaimRecord]
    ) -> None:
        self._rel.insert_session(session)
        for claim in claims:
            self.store_claim(claim)
        score_str = f"{session.overall_score:.2f}" if session.overall_score is not None else "N/A"
        summary_text = (
            f"Session {session.session_id}. "
            f"Score: {score_str}. "
            f"Claims: {session.claims_count}. "
            f"Contradictions: {session.contradictions_detected}. "
            f"Strengths: {', '.join(session.strengths)}. "
            f"Weaknesses: {', '.join(session.weaknesses)}."
        )
        self._vec.upsert(
            documents=[summary_text],
            metadatas=[{
                "session_id": session.session_id,
                "overall_score": session.overall_score if session.overall_score is not None else 0.0,
            }],
            ids=[session.session_id],
            collection_name=_SESSIONS_COLLECTION,
        )

    def retrieve_claims(self, query: str, top_k: int) -> list[ClaimRecord]:
        results = self._vec.query(query, _CLAIMS_COLLECTION, top_k)
        claims: list[ClaimRecord] = []
        for r in results:
            meta = r.get("metadata", {})
            try:
                claims.append(ClaimRecord(
                    claim_id=r["id"],
                    session_id=meta.get("session_id", ""),
                    turn_number=int(meta.get("turn_number", 0)),
                    claim_text=r.get("document", ""),
                    alignment=meta.get("alignment", "novel"),
                    mapped_to_slide=None,
                    prior_conflict=None,
                    timestamp=datetime.now(),
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
