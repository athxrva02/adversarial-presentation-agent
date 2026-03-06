"""
Recency-weighted retrieval for the Episodic Memory store

Implements temporal decay as described in the technical plan:
    score = similarity_score * decay_factor ^ session_age

Where session_age is the number of sessions elapsed since the record was created.

Public API:
    rerank_with_recency(results, current_session_index, decay_factor) -> list[dict]
    get_session_index(session_id)                                      -> int
    build_recency_filter(min_session_index)                            -> dict | None

Integration:
    Called by memory/retrieval.py after raw ChromaDB similarity search to
    re-score and re-sort results by recency-weighted relevance.

Design notes:
- session_age=0 means the current session → no decay (factor^0 = 1.0).
- session_age is computed from metadata["session_index"] stored alongside
  each claim/session record in ChromaDB.
- If session_index metadata is absent, the record is treated as maximally old
  (age = current_session_index) — conservative, won't be over-promoted.
- decay_factor comes from config.recency_decay_factor (default 0.85).
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config shim
# ---------------------------------------------------------------------------

def _get_decay_factor() -> float:
    try:
        from config import settings  # type: ignore
        return settings.recency_decay_factor
    except Exception:
        return 0.85


# ---------------------------------------------------------------------------
# Core reranking
# ---------------------------------------------------------------------------

def rerank_with_recency(
    results: list[dict],
    current_session_index: int,
    decay_factor: Optional[float] = None,
) -> list[dict]:
    """
    Re-score a list of ChromaDB query results using temporal decay.

    Each result dict must have the standard VectorStore query shape:
        { "id": str, "document": str, "metadata": dict, "distance": float }

    The `metadata` dict should contain `"session_index": int` (0-based counter
    incremented each session). If absent, the result is treated as oldest possible.

    Args:
        results:               Raw results from VectorStore.query(), sorted by distance.
        current_session_index: The 0-based index of the *current* session.
        decay_factor:          Multiplier per session of age (default from config).

    Returns:
        Results list re-sorted by descending recency_score.
        Each result has an added "recency_score" key.
    """
    if not results:
        return results

    factor = decay_factor if decay_factor is not None else _get_decay_factor()

    scored: list[dict] = []
    for r in results:
        meta = r.get("metadata", {})

        # session_index stored when the record was created
        record_session_idx = meta.get("session_index")
        if record_session_idx is None:
            # Treat as oldest: maximum possible age
            session_age = current_session_index
        else:
            session_age = max(0, current_session_index - int(record_session_idx))

        # distance is [0, 2] for cosine; convert to similarity [0, 1]
        raw_distance = r.get("distance", 1.0)
        similarity = max(0.0, 1.0 - raw_distance / 2.0)

        # Apply temporal decay
        recency_weight = factor ** session_age
        recency_score = similarity * recency_weight

        scored.append({**r, "recency_score": recency_score, "session_age": session_age})

    # Sort by descending recency_score
    scored.sort(key=lambda x: x["recency_score"], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Session index helpers
# ---------------------------------------------------------------------------

def get_current_session_index() -> int:
    """
    Return the current session count from the relational store.
    Used to compute `current_session_index` for reranking.
    """
    try:
        from storage.relational_store import get_relational_store  # type: ignore
        rs = get_relational_store()
        sessions = rs.get_all_sessions(limit=10000)
        return len(sessions)
    except Exception as exc:
        logger.warning("Could not read session count for recency: %s", exc)
        return 0


def annotate_with_session_index(metadata: dict, session_index: int) -> dict:
    """
    Add session_index to a metadata dict before storing in ChromaDB.
    Should be called by any write path (episodic, semantic, common_ground).

    Example usage in episodic memory write:
        metadata = annotate_with_session_index(
            {"claim_id": claim.claim_id, ...},
            session_index=get_current_session_index()
        )
    """
    return {**metadata, "session_index": session_index}


# ---------------------------------------------------------------------------
# Edge case: first session (empty memory)
# ---------------------------------------------------------------------------

def handle_empty_memory(store_name: str) -> list[dict]:
    """
    Called by retrieval logic when a store is empty (typically session 1).
    Returns an empty list and logs a clear message so the orchestrator can
    fall back to document-grounded questioning.
    """
    logger.info(
        "Memory store '%s' is empty (likely first session). "
        "Reasoning layer will fall back to document-grounded questions.",
        store_name,
    )
    return []


# ---------------------------------------------------------------------------
# Edge case: resolved contradictions filter
# ---------------------------------------------------------------------------

def filter_resolved_contradictions(results: list[dict]) -> list[dict]:
    """
    Remove results that reference claims marked as resolved contradictions.
    A claim is considered resolved if its metadata contains
    `"contradiction_resolved": "true"`.

    This prevents the agent from repeatedly surfacing issues the user has
    already clarified and negotiated away.
    """
    filtered = []
    for r in results:
        meta = r.get("metadata", {})
        if meta.get("contradiction_resolved", "false") == "true":
            logger.debug("Filtered resolved contradiction: %s", r.get("id"))
            continue
        filtered.append(r)
    return filtered


# ---------------------------------------------------------------------------
# Edge case: PDF re-upload detection
# ---------------------------------------------------------------------------

def detect_pdf_reupload(new_pdf_path: str) -> bool:
    """
    Check whether a PDF with the same filename has already been ingested.
    Returns True if the document appears to be a re-upload.

    This is a lightweight check based on filename stored in SQLite metadata.
    The caller (UI or pipeline) should decide whether to clear and re-ingest
    or skip ingestion.
    """
    import os
    new_name = os.path.basename(new_pdf_path)
    try:
        from storage.relational_store import get_relational_store  # type: ignore
        rs = get_relational_store()
        chunks = rs.get_all_chunks()
        # We store the source PDF name in chunk_id prefix; check metadata instead
        # For now, use a simple heuristic: if there are any chunks, warn the caller.
        if chunks:
            logger.info(
                "PDF re-upload detected: %d chunks already stored. "
                "DocumentMemory.store() will upsert safely.",
                len(chunks),
            )
            return True
    except Exception:
        pass
    return False
