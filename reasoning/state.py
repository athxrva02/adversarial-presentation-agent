"""
Shared LangGraph state for a single practice session.

This is intentionally a TypedDict so:
- LangGraph nodes can read/write partial updates
- the team can evolve fields without breaking runtime
"""

from __future__ import annotations

from typing import TypedDict, Optional, Annotated, Any
from operator import add

from storage.schemas import (
    MemoryBundle,
    Classification,
    ConflictResult,
    SessionRecord,
    ClaimRecord,
)


class SessionState(TypedDict, total=False):
    # Current turn input
    user_input: str
    turn_number: int

    # Optional session identifier (set by orchestrator / runner)
    session_id: str

    # LLM outputs
    classification: Optional[Classification]
    agent_response: Optional[str]

    # Session lifecycle phase:
    # "practice" | "assessment" | "negotiation" | "update"
    phase: str

    # Memory I/O
    memory_bundle: Optional[MemoryBundle]

    # Contradiction module I/O
    conflict_result: Optional[ConflictResult]

    # Accumulated session data
    # A list of turn dicts, e.g. {"role": "user"|"assistant", "content": "...", ...}
    turns: Annotated[list[dict[str, Any]], add]

    # Claim records appended per user turn (or per extracted claim if you later refine)
    claims: Annotated[list[ClaimRecord], add]

    # Session-end artifacts
    session_summary: Optional[SessionRecord]

    # Optional scoring breakdown
    score_breakdown: Optional[dict[str, Any]]

    # Negotiation artifacts
    negotiation_items: Optional[list[dict[str, Any]]]
    negotiation_decisions: Optional[list[dict[str, Any]]]

    # Control flags
    session_active: bool