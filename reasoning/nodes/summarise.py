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
    voice_summary = state.get("voice_summary")
    
    prompt = build_summarisation_prompt(
        turns=turns,
        claims=claims,
        memory_bundle=memory_bundle,
        voice_summary=voice_summary,
    )

    try:
        raw = call_llm_structured(
            system_prompt=prompt["system"],
            user_prompt=prompt["user"],
            schema_hint=SUMMARISATION_SCHEMA_HINT,
            options=opts_summarise_or_score(),
        )
    except Exception:
        raw = {}  # fall through to defaults below — better than crashing

    # Defensive parsing with defaults
    strengths = list(raw.get("strengths") or []) if isinstance(raw, dict) else []
    weaknesses = list(raw.get("weaknesses") or []) if isinstance(raw, dict) else []

    # Count contradictions from actual claim records (prior_conflict set),
    # not from LLM output which is unreliable.
    contradictions_detected = sum(
        1 for c in claims
        if str(getattr(c, "prior_conflict", "") or "").strip()
    )

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

    # Optionally store the extra LLM fields in score_breakdown or similar,
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

def _render_voice_summary(voice_summary: Optional[dict[str, Any]]) -> str:
    if not voice_summary:
        return "VOICE_SUMMARY: (none)\n"

    return (
        "VOICE_SUMMARY:\n"
        f"- delivery_voice_score: {voice_summary.get('delivery_voice_score')}\n"
        f"- speaking_rate_wpm: {voice_summary.get('speaking_rate_wpm'):.1f}\n"
        f"- articulation_rate_wpm: {voice_summary.get('articulation_rate_wpm'):.1f}\n"
        f"- pause_count: {voice_summary.get('pause_count')}\n"
        f"- long_pause_count: {voice_summary.get('long_pause_count')}\n"
        f"- mean_pause_s: {voice_summary.get('mean_pause_s'):.2f}\n"
        f"- silence_ratio: {voice_summary.get('silence_ratio'):.2f}\n"
        f"- volume_mean_dbfs: {voice_summary.get('volume_mean_dbfs'):.1f}\n"
        f"- volume_std_db: {voice_summary.get('volume_std_db'):.1f}\n"
        f"- pitch_range_semitones: {voice_summary.get('pitch_range_semitones'):.1f}\n"
        f"- delivery_feedback: {voice_summary.get('delivery_feedback', [])}\n"
    )
