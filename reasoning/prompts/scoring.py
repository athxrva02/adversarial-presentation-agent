"""
Prompt builder for composite performance scoring (end of session).

Design goals:
- Produce a stable numeric overall_score (0..100) for longitudinal tracking
- Provide a rubric breakdown for analysis and debugging
- Stay grounded in session evidence (turns/claims and summary)
- Avoid inventing facts

Output is strict JSON.

The score node will:
- call this prompt
- parse JSON
- write SessionRecord.overall_score
- store breakdown in state["score_breakdown"]
"""

from __future__ import annotations

from typing import Any, Optional

from reasoning.prompts._base import json_only_system, _truncate

SCORING_SCHEMA_HINT = """
{
  "overall_score": number,
  "rubric": {
    "clarity_structure": number,
    "evidence_specificity": number,
    "definition_precision": number,
    "logical_coherence": number,
    "handling_adversarial_questions": number
  },
  "notes": {
    "top_strengths": [string],
    "top_weaknesses": [string],
    "most_important_next_step": string
  }
}
""".strip()


def _render_summary(session_summary: Optional[Any], *, max_chars: int = 1200) -> str:
    """
    Render SessionRecord (or dict) compactly.
    """
    if session_summary is None:
        return "SESSION_SUMMARY: (none)\n"

    # duck-typed for Pydantic model or dict
    strengths = getattr(session_summary, "strengths", None) or (session_summary.get("strengths") if isinstance(session_summary, dict) else None) or []
    weaknesses = getattr(session_summary, "weaknesses", None) or (session_summary.get("weaknesses") if isinstance(session_summary, dict) else None) or []
    contradictions = getattr(session_summary, "contradictions_detected", None) or (session_summary.get("contradictions_detected") if isinstance(session_summary, dict) else 0) or 0

    block = (
        "SESSION_SUMMARY:\n"
        f"- strengths: {strengths}\n"
        f"- weaknesses: {weaknesses}\n"
        f"- contradictions_detected: {contradictions}\n"
    )
    return _truncate(block, max_chars=max_chars) + "\n"


def _render_turns(turns: Optional[list[dict[str, Any]]], *, max_items: int = 12, max_chars: int = 260) -> str:
    """
    Provide lightweight evidence for scoring without sending huge transcripts.
    """
    if not turns:
        return "TURN_EVIDENCE: (none)\n"

    subset = turns[-max_items:]
    lines = ["TURN_EVIDENCE (most recent first):"]
    for t in reversed(subset):
        role = str(t.get("role", "unknown"))
        content = _truncate(str(t.get("content", "")), max_chars=max_chars)
        lines.append(f"- {role}: {content}")
    return "\n".join(lines) + "\n"


def build_scoring_prompt(
    *,
    session_summary: Optional[Any],
    turns: Optional[list[dict[str, Any]]] = None,
) -> dict[str, str]:
    """
    Build system+user prompt for scoring.

    Inputs:
      session_summary: SessionRecord produced by summarise node (preferred)
      turns: short evidence window of the session turns

    Output:
      {"system": ..., "user": ...}
    """
    system = json_only_system(SCORING_SCHEMA_HINT)

    summary_block = _render_summary(session_summary)
    turns_block = _render_turns(turns)

    user = (
        "Task: Assign a composite performance score for this adversarial presentation practice session.\n\n"
        "You must return ONLY valid JSON matching the schema.\n\n"
        "Scoring rubric (each 0..100):\n"
        "- clarity_structure: argument is clear, well-structured, answers questions directly\n"
        "- evidence_specificity: provides concrete evidence, metrics, baselines, methods when claiming results\n"
        "- definition_precision: defines key terms precisely, avoids ambiguity\n"
        "- logical_coherence: claims follow logically, avoids contradictions and unsupported leaps\n"
        "- handling_adversarial_questions: handles probing follow-ups without evasion; admits uncertainty appropriately\n\n"
        "overall_score guidance:\n"
        "- overall_score is NOT a simple average; weight evidence_specificity and logical_coherence slightly higher.\n"
        "- Keep scores stable and conservative; do not inflate.\n\n"
        "Constraints:\n"
        "- Base scores ONLY on the provided session summary and turn evidence.\n"
        "- If evidence is insufficient for a dimension, score it lower rather than guessing.\n"
        "- Provide short, actionable notes.\n\n"
        f"{summary_block}\n"
        f"{turns_block}\n"
        "Return ONLY the JSON object."
    )

    return {"system": system, "user": user}