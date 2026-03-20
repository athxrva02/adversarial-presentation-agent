"""
LangGraph node: score a session (end of session).

Responsibilities:
- Build scoring prompt from session_summary (+ small evidence from turns)
- Call LLM structured JSON
- Compute overall_score deterministically from rubric weights (NOT from LLM)
- Update SessionRecord.overall_score
- Store rubric + notes into state["score_breakdown"] (merged with existing breakdown if present)

Important:
- Do NOT modify strengths/weaknesses here (summarise node owns that)
- Keep scoring stable (use low temperature via opts_summarise_or_score)
- overall_score is computed in Python, not by the LLM, for reproducibility
"""

from __future__ import annotations

from typing import Any, Dict

from reasoning.state import SessionState
from reasoning.llm import call_llm_structured, opts_summarise_or_score
from reasoning.prompts.scoring import build_scoring_prompt, SCORING_SCHEMA_HINT, RUBRIC_WEIGHTS
from storage.schemas import SessionRecord


def _clamp_score(x: Any, lo: float = 1.0, hi: float = 5.0) -> int:
    """Clamp a rubric dimension score to valid 1-5 range."""
    try:
        v = int(round(float(x)))
    except Exception:
        return 1
    return max(int(lo), min(int(hi), v))


def _extract_dimension_score(dim_value: Any) -> int:
    """Extract integer score from a rubric dimension (handles both CoT dict and plain number)."""
    if isinstance(dim_value, dict):
        return _clamp_score(dim_value.get("score", 1))
    return _clamp_score(dim_value)


def _extract_dimension_reasoning(dim_value: Any) -> str:
    """Extract reasoning string from a rubric dimension."""
    if isinstance(dim_value, dict):
        return str(dim_value.get("reasoning", ""))
    return ""


def compute_overall_score(rubric: dict[str, Any]) -> float:
    """
    Compute overall_score (0-100) deterministically from rubric dimension scores (1-5).

    Uses explicit weights from RUBRIC_WEIGHTS. Maps the weighted 1-5 average
    to a 0-100 scale: score_100 = (weighted_avg - 1) * 25.
    """
    weighted_sum = 0.0
    total_weight = 0.0
    for dim, weight in RUBRIC_WEIGHTS.items():
        score = _extract_dimension_score(rubric.get(dim, 1))
        weighted_sum += score * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    weighted_avg = weighted_sum / total_weight  # 1.0 to 5.0
    # Map 1-5 to 0-100: (avg - 1) * 25
    score_100 = (weighted_avg - 1.0) * 25.0
    return round(min(100.0, max(0.0, score_100)), 1)


def run(state: SessionState) -> Dict[str, Any]:
    session_summary = state.get("session_summary")
    turns = state.get("turns", [])

    if session_summary is None:
        raise ValueError("score node requires state['session_summary'] to be set (run summarise first).")

    prompt = build_scoring_prompt(
        session_summary=session_summary,
        turns=turns,
    )

    try:
        raw = call_llm_structured(
            system_prompt=prompt["system"],
            user_prompt=prompt["user"],
            schema_hint=SCORING_SCHEMA_HINT,
            options=opts_summarise_or_score(),
        )
    except Exception:
        raw = {}  # fall through to defaults — session results still display

    # Parse outputs defensively
    rubric = raw.get("rubric", {}) if isinstance(raw, dict) else {}
    notes = raw.get("notes", {}) if isinstance(raw, dict) else {}

    # Compute overall score deterministically from rubric weights
    overall_score = compute_overall_score(rubric)

    # Build a flattened rubric view with scores and reasoning separated
    rubric_scores = {}
    rubric_reasoning = {}
    for dim in RUBRIC_WEIGHTS:
        rubric_scores[dim] = _extract_dimension_score(rubric.get(dim, 1))
        rubric_reasoning[dim] = _extract_dimension_reasoning(rubric.get(dim, ""))

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
        "rubric_scores": rubric_scores,
        "rubric_reasoning": rubric_reasoning,
        "rubric_weights": dict(RUBRIC_WEIGHTS),
        "notes": notes,
    }

    return {
        "session_summary": updated_summary,
        "score_breakdown": merged_breakdown,
    }
