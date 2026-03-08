"""Semantic Memory — long-term patterns promoted from episodic data."""
from __future__ import annotations

from collections import defaultdict

from config import settings
from storage.schemas import SemanticPattern
from storage.vector_store import VectorStore
from storage.relational_store import RelationalStore

_COLLECTION = "semantic_patterns"


class SemanticMemory:
    def __init__(
        self, *, vector_store: VectorStore, relational_store: RelationalStore
    ) -> None:
        self._vec = vector_store
        self._rel = relational_store

    def store_pattern(self, pattern: SemanticPattern) -> None:
        self._rel.upsert_semantic_pattern(pattern)
        self._vec.upsert(
            _COLLECTION,
            ids=[pattern.pattern_id],
            documents=[pattern.text],
            metadatas=[{
                "category": pattern.category,
                "confidence": pattern.confidence,
                "status": pattern.status,
            }],
        )

    def retrieve(self, query: str, top_k: int) -> list[SemanticPattern]:
        results = self._vec.query(_COLLECTION, query, top_k)
        patterns: list[SemanticPattern] = []
        for r in results:
            pattern = self._rel.get_pattern(r["id"])
            if pattern is not None:
                patterns.append(pattern)
        return patterns

    def get_active(self) -> list[SemanticPattern]:
        return self._rel.get_semantic_patterns(status="active")

    def promote(self, session_id: str) -> list[SemanticPattern]:
        """Scan all claims across sessions. If an alignment value appears in
        >= promotion_threshold distinct sessions, create or update a pattern.
        """
        all_claims = self._rel.get_all_claims()
        if not all_claims:
            return []

        # Group claim alignment values by distinct sessions
        # alignment_value -> set of session_ids
        alignment_sessions: dict[str, set[str]] = defaultdict(set)
        # alignment_value -> list of (claim_id, claim_text)
        alignment_evidence: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for claim in all_claims:
            aval = claim.alignment.value
            alignment_sessions[aval].add(claim.session_id)
            alignment_evidence[aval].append((claim.claim_id, claim.claim_text))

        threshold = settings.promotion_threshold
        promoted: list[SemanticPattern] = []

        for alignment_val, sessions in alignment_sessions.items():
            if len(sessions) < threshold:
                continue

            pattern_id = f"sp_{alignment_val}"
            existing = self._rel.get_pattern(pattern_id)
            evidence_ids = [cid for cid, _ in alignment_evidence[alignment_val]]
            representative_text = alignment_evidence[alignment_val][0][1]

            if existing is not None:
                updated = SemanticPattern(
                    pattern_id=pattern_id,
                    category=alignment_val,
                    text=existing.text,
                    confidence=min(0.95, existing.confidence + 0.05),
                    direction=existing.direction,
                    first_seen=existing.first_seen,
                    last_updated=session_id,
                    session_count=len(sessions),
                    status="active",
                    evidence=evidence_ids,
                )
            else:
                updated = SemanticPattern(
                    pattern_id=pattern_id,
                    category=alignment_val,
                    text=representative_text,
                    confidence=0.7,
                    direction="stable",
                    first_seen=session_id,
                    last_updated=session_id,
                    session_count=len(sessions),
                    status="active",
                    evidence=evidence_ids,
                )

            self.store_pattern(updated)
            promoted.append(updated)

        return promoted
