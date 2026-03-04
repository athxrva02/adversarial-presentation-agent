"""
LangGraph node: score a session (end of session).

Responsibilities:
- Build scoring prompt from session_summary (+ small evidence from turns)
- Call LLM structured JSON
- Update SessionRecord.overall_score
- Store rubric + notes into state["score_breakdown"] (merged with existing breakdown if present)

Important:
- Do NOT modify strengths/weaknesses here (summarise node owns that)
- Keep scoring stable (use low temperature via opts_summarise_or_score)
"""

from __future__ import annotations

from typing import Any, Dict

from reasoning.state import SessionState
from reasoning.llm import call_llm_structured, opts_summarise_or_score
from reasoning.prompts.scoring import build_scoring_prompt, SCORING_SCHEMA_HINT
from storage.schemas import SessionRecord


def _clamp_score(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 100.0:
        return 100.0
    return v


def run(state: SessionState) -> Dict[str, Any]:
    session_summary = state.get("session_summary")
    turns = state.get("turns", [])

    if session_summary is None:
        raise ValueError("score node requires state['session_summary'] to be set (run summarise first).")

    prompt = build_scoring_prompt(
        session_summary=session_summary,
        turns=turns,
    )

    raw = call_llm_structured(
        system_prompt=prompt["system"],
        user_prompt=prompt["user"],
        schema_hint=SCORING_SCHEMA_HINT,
        options=opts_summarise_or_score(),
    )

    # Parse outputs defensively
    overall_score = _clamp_score(raw.get("overall_score", 0.0)) if isinstance(raw, dict) else 0.0
    rubric = raw.get("rubric", {}) if isinstance(raw, dict) else {}
    notes = raw.get("notes", {}) if isinstance(raw, dict) else {}

    # Update SessionRecord.overall_score (Pydantic model is immutable? default BaseModel is mutable)
    if isinstance(session_summary, SessionRecord):
        session_summary.overall_score = overall_score
        updated_summary = session_summary
    else:
        # in case someone passed a dict (shouldn't happen)
        updated_summary = SessionRecord(**{**session_summary, "overall_score": overall_score})

    # Merge into score_breakdown
    prior_breakdown = state.get("score_breakdown") or {}
    merged_breakdown = {
        **prior_breakdown,
        "overall_score": overall_score,
        "rubric": rubric,
        "notes": notes,
    }

    return {
        "session_summary": updated_summary,
        "score_breakdown": merged_breakdown,
    }