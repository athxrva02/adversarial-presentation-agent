"""Episodic Memory — stores and retrieves session records and claim records."""
from __future__ import annotations

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
            _CLAIMS_COLLECTION,
            ids=[claim.claim_id],
            documents=[claim.claim_text],
            metadatas=[{
                "session_id": claim.session_id,
                "turn_number": claim.turn_number,
                "alignment": claim.alignment.value,
            }],
        )

    def store_session(
        self, session: SessionRecord, claims: list[ClaimRecord]
    ) -> None:
        self._rel.upsert_session(session)
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
            _SESSIONS_COLLECTION,
            ids=[session.session_id],
            documents=[summary_text],
            metadatas=[{
                "session_id": session.session_id,
                "overall_score": session.overall_score if session.overall_score is not None else 0.0,
            }],
        )

    def retrieve_claims(self, query: str, top_k: int) -> list[ClaimRecord]:
        results = self._vec.query(_CLAIMS_COLLECTION, query, top_k)
        claims: list[ClaimRecord] = []
        for r in results:
            claim = self._rel.get_claim(r["id"])
            if claim is not None:
                claims.append(claim)
        return claims

    def retrieve_sessions(self, query: str, top_k: int) -> list[SessionRecord]:
        results = self._vec.query(_SESSIONS_COLLECTION, query, top_k)
        sessions: list[SessionRecord] = []
        for r in results:
            session = self._rel.get_session(r["id"])
            if session is not None:
                sessions.append(session)
        return sessions
