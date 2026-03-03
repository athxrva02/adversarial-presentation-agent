"""
Conditional edge routing for LangGraph.

These functions inspect SessionState and return a string key that the graph
uses to decide the next node.

They are intentionally conservative and avoid dependence on any one module.
"""

from __future__ import annotations

from reasoning.state import SessionState
from storage.schemas import ConflictStatus, ResponseClass


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
    if conflict is not None and getattr(conflict, "status", None) == ConflictStatus.TRUE_CONTRADICTION:
        return "escalate_contradiction"

    classification = state.get("classification")
    if classification is None:
        return "redirect"

    rc = getattr(classification, "response_class", None)

    if rc == ResponseClass.EVASION:
        return "redirect"

    if rc == ResponseClass.WEAK:
        return "probe_weak"

    if rc == ResponseClass.CONTRADICTION:
        return "escalate_contradiction"

    if rc == ResponseClass.STRONG:
        # Even a "strong" answer can be pushed: ask for evidence/assumptions/boundary cases
        return "request_evidence"

    return "request_evidence"