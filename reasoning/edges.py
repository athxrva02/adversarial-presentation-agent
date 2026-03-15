"""
Conditional edge routing for LangGraph.

These functions inspect SessionState and return a string key that the graph
uses to decide the next node.

They are intentionally conservative and avoid dependence on any one module.
"""

from __future__ import annotations

import logging

from reasoning.state import SessionState
from storage.schemas import ConflictStatus, ResponseClass

logger = logging.getLogger(__name__)


def route_after_phase(state: SessionState) -> str:
    """
    Decide whether to continue the practice loop or end the session.

    Expected return values:
      - "continue"
      - "end_session"
    """
    if state.get("session_active", True):
        return "continue"
    return "end_session"


def route_after_classification(state: SessionState) -> str:
    """
    Choose a questioning strategy after contradiction check + classification.

    Expected return values (keys you map in graph.add_conditional_edges):
      - "escalate_contradiction"
      - "probe_weak"
      - "request_evidence"
      - "redirect"

    Logic:
    1) If contradiction module marked a true contradiction => escalate.
    2) If classification missing => redirect (ask a clarifying question).
    3) If user evaded => redirect.
    4) If weak => probe_weak.
    5) If strong => request_evidence (still adversarial: ask for support/assumptions).
    6) If classification says contradiction => escalate.
    Default => request_evidence.
    """
    conflict = state.get("conflict_result")
    classification = state.get("classification")
    rc = getattr(classification, "response_class", None) if classification else None
    conflict_status = getattr(conflict, "status", None) if conflict else None

    if conflict is not None and conflict_status == ConflictStatus.TRUE_CONTRADICTION:
        logger.info("Routing: escalate_contradiction (conflict_result=%s)", conflict_status)
        return "escalate_contradiction"

    if classification is None:
        logger.info("Routing: redirect (no classification)")
        return "redirect"

    if rc == ResponseClass.EVASION:
        logger.info("Routing: redirect (evasion)")
        return "redirect"

    if rc == ResponseClass.WEAK:
        logger.info("Routing: probe_weak")
        return "probe_weak"

    if rc == ResponseClass.CONTRADICTION:
        logger.info(
            "Routing: escalate_contradiction (classification=%s, conflict_status=%s)",
            rc, conflict_status,
        )
        return "escalate_contradiction"

    if rc == ResponseClass.STRONG:
        logger.info("Routing: request_evidence")
        return "request_evidence"

    logger.info("Routing: request_evidence (default, rc=%s)", rc)
    return "request_evidence"