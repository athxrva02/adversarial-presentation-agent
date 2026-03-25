"""Retrieve node — fetches a MemoryBundle from the MemoryModule.

The MemoryModule instance is passed through the graph state under the
``_memory_module`` key. If absent or ``None``, the node is a no-op and
sets ``memory_bundle`` to ``None``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from reasoning.state import SessionState

logger = logging.getLogger(__name__)

_ALL_STORES = ["document", "episodic", "semantic", "common_ground"]
_DOCUMENT_ONLY_STORES = ["document"]


# Context expansion change: broaden retrieval query beyond only the latest answer
def _trim(text: str, max_chars: int = 700) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _get_previous_agent_question(turns: list[dict[str, Any]]) -> str:
    for turn in reversed(turns):
        if str(turn.get("role", "")).strip().lower() == "agent":
            return str(turn.get("content", "")).strip()
    return ""


def _get_session_anchor(turns: list[dict[str, Any]]) -> str:
    # Use the first user turn as the presentation/topic anchor.
    for turn in turns:
        if str(turn.get("role", "")).strip().lower() == "user":
            text = str(turn.get("content", "")).strip()
            if text:
                return _trim(text, max_chars=700)
    return ""


def _get_recent_claim_anchor(claims: list[Any], limit: int = 2) -> str:
    snippets: list[str] = []
    for claim in claims[-limit:]:
        text = str(getattr(claim, "claim_text", "") or "").strip()
        if text:
            snippets.append(_trim(text, max_chars=220))
    return " | ".join(snippets)


def _build_retrieval_query(state: SessionState) -> str:
    latest_input = str(state.get("user_input", "") or "").strip()
    turns = list(state.get("turns", []) or [])
    claims = list(state.get("claims", []) or [])

    previous_question = _get_previous_agent_question(turns)
    session_anchor = _get_session_anchor(turns)
    recent_claims = _get_recent_claim_anchor(claims, limit=2)

    parts: list[str] = []
    if latest_input:
        parts.append(f"LATEST_ANSWER: {latest_input}")
    if previous_question:
        parts.append(f"PREVIOUS_QUESTION: {previous_question}")
    if session_anchor and session_anchor != latest_input:
        parts.append(f"PRESENTATION_CONTEXT: {session_anchor}")
    if recent_claims:
        parts.append(f"RECENT_CLAIMS: {recent_claims}")

    query = "\n".join(parts).strip()
    return query or latest_input


def run(state: SessionState) -> Dict[str, Any]:
    module = state.get("_memory_module")
    if module is None:
        logger.debug("retrieve: no memory module, returning None")
        return {"memory_bundle": None}

    query = _build_retrieval_query(state)

    mode = state.get("memory_mode", "hybrid")
    stores = _DOCUMENT_ONLY_STORES if mode == "document_only" else _ALL_STORES

    bundle = module.retrieve(
        query=query,
        stores=stores,
    )
    n_claims = len(bundle.episodic_claims) if bundle else 0
    n_docs = len(bundle.document_context) if bundle else 0
    logger.info(
        "retrieve: mode=%s, query=%r, episodic_claims=%d, document_chunks=%d",
        mode, query[:120], n_claims, n_docs,
    )
    return {"memory_bundle": bundle}
