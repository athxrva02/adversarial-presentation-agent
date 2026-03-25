"""MemoryModule — orchestrates all memory sub-modules.

This is the single entry point the reasoning layer uses to read and write
memory.  It delegates to :class:`DocumentMemory`, :class:`EpisodicMemory`,
:class:`SemanticMemory`, and :class:`CommonGroundMemory` internally.
"""
from __future__ import annotations

from config import settings
from memory.common_ground import CommonGroundMemory
from memory.document import DocumentMemory
from memory.episodic import EpisodicMemory
from memory.retrieval import merge_and_rank
from memory.semantic import SemanticMemory
from storage.relational_store import RelationalStore
from storage.schemas import (
    ClaimRecord,
    CommonGroundEntry,
    DocumentChunk,
    MemoryBundle,
    SemanticPattern,
    SessionRecord,
)
from storage.vector_store import VectorStore


class MemoryModule:
    """Facade that routes memory operations to the appropriate sub-module."""

    def __init__(
        self,
        *,
        vector_store: VectorStore | None = None,
        relational_store: RelationalStore | None = None,
    ) -> None:
        vec = vector_store or VectorStore()
        rel = relational_store or RelationalStore()

        self._document = DocumentMemory(vector_store=vec, relational_store=rel)
        self._episodic = EpisodicMemory(vector_store=vec, relational_store=rel)
        self._semantic = SemanticMemory(vector_store=vec, relational_store=rel)
        self._common_ground = CommonGroundMemory(vector_store=vec, relational_store=rel)
        self._rel = rel

    def clear_all(self) -> None:
        self._common_ground.clear()
        self._semantic.clear()
        self._episodic.clear()
        self._document.clear()

    # ---- Retrieval ----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        stores: list[str],
        top_k: int | None = None,
    ) -> MemoryBundle:
        k = top_k or settings.retrieval_top_k

        chunks = self._document.retrieve(query, k) if "document" in stores else []
        claims = self._episodic.retrieve_claims(query, k) if "episodic" in stores else []
        sessions = (
            self._episodic.retrieve_sessions(query, k)
            if "episodic" in stores
            else []
        )
        patterns = self._semantic.retrieve(query, k) if "semantic" in stores else []
        cg = self._common_ground.retrieve(query, k) if "common_ground" in stores else []

        # Build session ordering for recency weighting (newest = age 0)
        all_sessions = self._rel.get_all_sessions()
        session_order = {s["session_id"]: i for i, s in enumerate(all_sessions)}

        return merge_and_rank(
            document_chunks=chunks,
            episodic_claims=claims,
            episodic_sessions=sessions,
            semantic_patterns=patterns,
            common_ground=cg,
            top_k=k,
            session_order=session_order,
        )

    # Context expansion change: expose document-only candidates for document-driven questioning
    def get_document_question_candidates(
        self,
        *,
        limit: int = 6,
        exclude_chunk_ids: list[str] | None = None,
    ) -> list[DocumentChunk]:
        return self._document.list_chunks_for_questioning(
            limit=limit,
            exclude_chunk_ids=exclude_chunk_ids,
        )

    # ---- Storage ------------------------------------------------------------

    def store_document(self, chunk: DocumentChunk) -> None:
        self._document.store([chunk])

    def store_claim(self, claim: ClaimRecord) -> None:
        self._episodic.store_claim(claim)

    def store_session(
        self, session: SessionRecord, claims: list[ClaimRecord]
    ) -> None:
        self._episodic.store_session(session, claims)

    def store_common_ground(self, entry: CommonGroundEntry) -> None:
        self._common_ground.store(entry)

    def store_semantic_pattern(self, pattern: SemanticPattern) -> None:
        self._semantic.store_pattern(pattern)

    # ---- Promotion ----------------------------------------------------------

    def promote_patterns(self, session_id: str) -> list[SemanticPattern]:
        return self._semantic.promote(session_id)
