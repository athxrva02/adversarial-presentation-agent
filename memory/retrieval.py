"""Retrieval utilities — merge, dedup, and recency-weighted ranking."""
from __future__ import annotations

from typing import Any

from config import settings
from storage.schemas import (
    ClaimRecord,
    CommonGroundEntry,
    DocumentChunk,
    MemoryBundle,
    SemanticPattern,
    SessionRecord,
)


def _recency_score(item: Any, session_order: dict[str, int]) -> float:
    """Return a recency weight for *item* based on its session age.

    ``session_order`` maps ``session_id`` → age (0 = most recent).
    Score = ``recency_decay_factor ** age``.
    """
    session_id = getattr(item, "session_id", None)
    if session_id is None or session_id not in session_order:
        return 1.0
    age = session_order[session_id]
    return settings.recency_decay_factor ** age


def _dedup_by_id(items: list, id_attr: str) -> list:
    """Remove duplicates by a unique id attribute, keeping first occurrence."""
    seen: set[str] = set()
    result: list = []
    for item in items:
        item_id = getattr(item, id_attr, None)
        if item_id is not None and item_id not in seen:
            seen.add(item_id)
            result.append(item)
    return result


def merge_and_rank(
    document_chunks: list[DocumentChunk],
    episodic_claims: list[ClaimRecord],
    episodic_sessions: list[SessionRecord],
    semantic_patterns: list[SemanticPattern],
    common_ground: list[CommonGroundEntry],
    top_k: int,
    session_order: dict[str, int] | None = None,
) -> MemoryBundle:
    """Merge per-store results into a single :class:`MemoryBundle`.

    Deduplicates by primary key, applies recency weighting to claims,
    and truncates each list to *top_k*.
    """
    order = session_order or {}

    chunks = _dedup_by_id(document_chunks, "chunk_id")[:top_k]

    claims = _dedup_by_id(episodic_claims, "claim_id")
    sessions = _dedup_by_id(episodic_sessions, "session_id")
    patterns = _dedup_by_id(semantic_patterns, "pattern_id")
    cg = _dedup_by_id(common_ground, "cg_id")

    if order:
        claims = sorted(claims, key=lambda c: _recency_score(c, order), reverse=True)
        sessions = sorted(sessions, key=lambda s: _recency_score(s, order), reverse=True)
        patterns = sorted(
            patterns,
            key=lambda p: settings.recency_decay_factor ** order.get(p.last_updated, len(order)),
            reverse=True,
        )
        cg = sorted(
            cg,
            key=lambda e: settings.recency_decay_factor ** order.get(e.session_agreed, len(order)),
            reverse=True,
        )

    claims = claims[:top_k]
    sessions = sessions[:top_k]
    patterns = patterns[:top_k]
    cg = cg[:top_k]

    return MemoryBundle(
        document_context=chunks,
        episodic_claims=claims,
        episodic_sessions=sessions,
        semantic_patterns=patterns,
        common_ground=cg,
    )
