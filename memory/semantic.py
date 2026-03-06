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
        self._rel.upsert_pattern(pattern)
        self._vec.upsert(
            documents=[pattern.text],
            metadatas=[{
                "category": pattern.category,
                "confidence": pattern.confidence,
                "status": pattern.status,
            }],
            ids=[pattern.pattern_id],
            collection_name=_COLLECTION,
        )

    def retrieve(self, query: str, top_k: int) -> list[SemanticPattern]:
        results = self._vec.query(query, _COLLECTION, top_k)
        patterns: list[SemanticPattern] = []
        for r in results:
            meta = r.get("metadata", {})
            try:
                patterns.append(SemanticPattern(
                    pattern_id=r["id"],
                    category=meta.get("category", "unknown"),
                    text=r.get("document", ""),
                    confidence=float(meta.get("confidence", 0.5)),
                    direction="stable",
                    first_seen="",
                    last_updated="",
                    session_count=1,
                    status=meta.get("status", "active"),
                    evidence=[],
                ))
            except Exception:
                continue
        return patterns

    def get_active(self) -> list[SemanticPattern]:
        rows = self._rel.get_all_patterns(status="active")
        patterns = []
        for d in rows:
            try:
                patterns.append(SemanticPattern(
                    pattern_id=d["pattern_id"],
                    category=d.get("category", "unknown"),
                    text=d.get("text", ""),
                    confidence=float(d.get("confidence", 0.5)),
                    direction=d.get("direction", "stable"),
                    first_seen=d.get("first_seen", ""),
                    last_updated=d.get("last_updated", ""),
                    session_count=int(d.get("session_count", 1)),
                    status=d.get("status", "active"),
                    evidence=d.get("evidence", []),
                ))
            except Exception:
                continue
        return patterns

    def promote(self, session_id: str) -> list[SemanticPattern]:
        all_claim_dicts = self._rel.get_recent_claims(limit=1000)
        if not all_claim_dicts:
            return []

        alignment_sessions: dict[str, set[str]] = defaultdict(set)
        alignment_evidence: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for claim in all_claim_dicts:
            aval = claim.get("alignment", "novel")
            sid = claim.get("session_id", "")
            cid = claim.get("claim_id", "")
            ctext = claim.get("claim_text", "")
            alignment_sessions[aval].add(sid)
            alignment_evidence[aval].append((cid, ctext))

        threshold = settings.promotion_threshold
        promoted: list[SemanticPattern] = []

        for alignment_val, sessions in alignment_sessions.items():
            if len(sessions) < threshold:
                continue

            pattern_id = f"sp_{alignment_val}"
            existing_rows = self._rel.get_all_patterns(status=None)
            existing = next((d for d in existing_rows if d["pattern_id"] == pattern_id), None)

            evidence_ids = [cid for cid, _ in alignment_evidence[alignment_val]]
            representative_text = alignment_evidence[alignment_val][0][1]

            if existing is not None:
                updated = SemanticPattern(
                    pattern_id=pattern_id,
                    category=alignment_val,
                    text=existing["text"],
                    confidence=min(0.95, float(existing["confidence"]) + 0.05),
                    direction=existing.get("direction", "stable"),
                    first_seen=existing.get("first_seen", session_id),
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
