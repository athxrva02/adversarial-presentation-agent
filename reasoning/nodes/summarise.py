"""
LangGraph node: summarise a session at session end.

Responsibilities:
- Build summarisation prompt from turns/claims (+ optional memory_bundle)
- Call LLM structured JSON
- Convert to SessionRecord by adding metadata in Python:
  - session_id
  - timestamp
  - duration_seconds (optional from state, else 0.0)
  - claims_count
- Write to state: session_summary
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from reasoning.state import SessionState
from reasoning.llm import call_llm_structured, opts_summarise_or_score
from reasoning.prompts.summarisation import (
    build_summarisation_prompt,
    SUMMARISATION_SCHEMA_HINT,
)
from storage.schemas import SessionRecord


def run(state: SessionState) -> Dict[str, Any]:
    turns = state.get("turns", [])
    claims = state.get("claims", [])
    memory_bundle = state.get("memory_bundle")

    prompt = build_summarisation_prompt(
        turns=turns,
        claims=claims,
        memory_bundle=memory_bundle,
    )

    raw = call_llm_structured(
        system_prompt=prompt["system"],
        user_prompt=prompt["user"],
        schema_hint=SUMMARISATION_SCHEMA_HINT,
        options=opts_summarise_or_score(),
    )

    # Defensive parsing with defaults
    strengths = list(raw.get("strengths") or []) if isinstance(raw, dict) else []
    weaknesses = list(raw.get("weaknesses") or []) if isinstance(raw, dict) else []

    cd_raw = raw.get("contradictions_detected") if isinstance(raw, dict) else 0
    if cd_raw is None:
        contradictions_detected = 0
    elif isinstance(cd_raw, bool):
        # Avoid True/False becoming 1/0 accidentally
        contradictions_detected = 0
    else:
        try:
            contradictions_detected = int(cd_raw)
        except (TypeError, ValueError):
            contradictions_detected = 0

    session_id = state.get("session_id", "unknown_session")
    duration_seconds = float(state.get("duration_seconds", 0.0))  # optional field

    session_summary = SessionRecord(
        session_id=session_id,
        timestamp=datetime.now(),
        duration_seconds=duration_seconds,
        overall_score=None,  # score node will fill this later
        strengths=strengths,
        weaknesses=weaknesses,
        claims_count=len(claims),
        contradictions_detected=contradictions_detected,
    )

    # You may optionally store the extra LLM fields in score_breakdown or similar,
    # but keep SessionRecord schema clean and stable.
    score_breakdown = {
        "key_claims": (raw.get("key_claims") or []) if isinstance(raw, dict) else [],
        "open_issues": (raw.get("open_issues") or []) if isinstance(raw, dict) else [],
        "overall_notes": (raw.get("overall_notes") or "") if isinstance(raw, dict) else "",
    }

    return {
        "session_summary": session_summary,
        "score_breakdown": score_breakdown,
    }