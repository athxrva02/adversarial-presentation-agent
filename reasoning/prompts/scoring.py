"""
Prompt builder for composite performance scoring (end of session).

Design goals:
- Produce a stable numeric overall_score (0..100) for longitudinal tracking
  (computed deterministically in Python from rubric scores)
- Provide a rubric breakdown with chain-of-thought reasoning per dimension
- Use a 1-5 scale per dimension with clear anchor descriptions for LLM reliability
- Stay grounded in session evidence (turns/claims and summary)
- Include few-shot calibration examples for consistent scoring
- Avoid inventing facts

Output is strict JSON.

The score node will:
- call this prompt
- parse JSON (rubric + notes)
- compute overall_score deterministically from rubric weights
- write SessionRecord.overall_score
- store breakdown in state["score_breakdown"]
"""

from __future__ import annotations

from typing import Any, Optional

from reasoning.prompts._base import json_only_system, _truncate

SCORING_SCHEMA_HINT = """
{
  "rubric": {
    "clarity_structure": { "reasoning": string, "score": integer 1-5 },
    "evidence_specificity": { "reasoning": string, "score": integer 1-5 },
    "definition_precision": { "reasoning": string, "score": integer 1-5 },
    "logical_coherence": { "reasoning": string, "score": integer 1-5 },
    "handling_adversarial_questions": { "reasoning": string, "score": integer 1-5 },
    "depth_of_understanding": { "reasoning": string, "score": integer 1-5 },
    "concession_and_qualification": { "reasoning": string, "score": integer 1-5 },
    "recovery_from_challenge": { "reasoning": string, "score": integer 1-5 },
    "vocal_delivery": { "reasoning": string, "score": integer 1-5 }
  },
  "notes": {
    "top_strengths": [string],
    "top_weaknesses": [string],
    "most_important_next_step": string
  }
}
""".strip()

# Weights for deterministic overall_score computation (must sum to 1.0)
RUBRIC_WEIGHTS: dict[str, float] = {
    "clarity_structure": 0.10,
    "evidence_specificity": 0.18,
    "definition_precision": 0.10,
    "logical_coherence": 0.18,
    "handling_adversarial_questions": 0.16,
    "depth_of_understanding": 0.12,
    "concession_and_qualification": 0.08,
    "recovery_from_challenge": 0.08,
}
# Only applied when VOICE_SUMMARY is available.
VOCAL_DELIVERY_WEIGHT: float = 0.08

RUBRIC_DIMENSIONS = (
    "Scoring rubric — rate each dimension on a 1-5 integer scale.\n"
    "For each dimension, first write a short 'reasoning' sentence explaining your score, then give the integer score.\n\n"

    "1. clarity_structure — Argument is clear, well-structured, answers questions directly.\n"
    "   1 = Rambling, disorganised, does not answer the question asked\n"
    "   2 = Partially answers but structure is unclear or unfocused\n"
    "   3 = Adequate structure, mostly on-topic, some tangents\n"
    "   4 = Well-structured, clear, directly addresses the question\n"
    "   5 = Exceptionally clear, logically sequenced, concise and complete\n\n"

    "2. evidence_specificity — Provides concrete evidence, metrics, baselines, methods when claiming results.\n"
    "   1 = No evidence, only vague assertions\n"
    "   2 = Some evidence but missing key details (no numbers, no baselines)\n"
    "   3 = Provides some metrics but incomplete (e.g. metric without baseline)\n"
    "   4 = Concrete numbers with baselines and methods stated\n"
    "   5 = Thorough quantitative evidence with baselines, confidence intervals or ablations\n\n"

    "3. definition_precision — Defines key terms precisely, avoids ambiguity.\n"
    "   1 = Key terms undefined or used inconsistently\n"
    "   2 = Some terms defined but important ones remain vague\n"
    "   3 = Most key terms defined adequately\n"
    "   4 = All key terms precisely defined with little ambiguity\n"
    "   5 = Definitions are precise, consistent, and proactively clarified\n\n"

    "4. logical_coherence — Claims follow logically, avoids contradictions and unsupported leaps.\n"
    "   1 = Major contradictions or completely unsupported claims\n"
    "   2 = Some logical gaps or minor contradictions\n"
    "   3 = Mostly coherent with occasional unsupported leaps\n"
    "   4 = Logically sound, claims follow from evidence\n"
    "   5 = Airtight reasoning, no gaps, all claims well-supported\n\n"

    "5. handling_adversarial_questions — Handles probing follow-ups without evasion; admits uncertainty appropriately.\n"
    "   1 = Evades questions, becomes defensive or dismissive\n"
    "   2 = Attempts answers but deflects on key challenges\n"
    "   3 = Addresses most challenges, occasionally evasive\n"
    "   4 = Engages constructively with challenges, admits gaps when appropriate\n"
    "   5 = Handles all challenges head-on, turns weaknesses into learning opportunities\n\n"

    "6. depth_of_understanding — Demonstrates understanding beyond surface-level claims.\n"
    "   1 = Only repeats memorised phrases, no deeper understanding\n"
    "   2 = Some understanding but cannot explain why or how\n"
    "   3 = Understands core concepts but struggles with edge cases\n"
    "   4 = Strong understanding, can explain mechanisms and trade-offs\n"
    "   5 = Deep mastery, connects to broader context and explains nuances\n\n"

    "7. concession_and_qualification — Appropriately qualifies claims and concedes valid counterpoints.\n"
    "   1 = Refuses to concede anything, overstates all claims\n"
    "   2 = Rarely qualifies, treats all claims as absolute\n"
    "   3 = Sometimes qualifies but misses important caveats\n"
    "   4 = Appropriately qualifies claims, concedes valid points\n"
    "   5 = Expertly balances confidence with intellectual honesty\n\n"

    "8. recovery_from_challenge — After being caught on a weak point, recovers constructively.\n"
    "   1 = Collapses or becomes incoherent after challenge\n"
    "   2 = Struggles to recover, repeats same weak answer\n"
    "   3 = Partially recovers but does not fully address the issue\n"
    "   4 = Recovers well, provides additional evidence or reframes\n"
    "   5 = Turns challenges into strengths, provides compelling recovery\n"

    "9. vocal_delivery — Spoken delivery quality based ONLY on VOICE_SUMMARY when available.\n"
    "   Consider pace, long pauses, silence ratio, pitch variation, volume variation, and clipping.\n"
    "   If VOICE_SUMMARY is missing, this dimension is not applicable for the session.\n"
    "   1 = Delivery seriously hurts comprehension: many long pauses, highly unstable pace, very flat or problematic voice\n"
    "   2 = Noticeable delivery problems: frequent pauses, weak variation, distracting pacing issues\n"
    "   3 = Adequate delivery: understandable, some hesitation or flatness, but not severely disruptive\n"
    "   4 = Strong delivery: clear pace, controlled pauses, good vocal emphasis and variation\n"
    "   5 = Excellent delivery: confident, well-paced, expressive, controlled, and highly effective\n\n"


)

FEW_SHOT_EXAMPLES = (
    "CALIBRATION EXAMPLES (use these to anchor your scoring):\n\n"

    "Example A — Weak performance (scores mostly 1-2):\n"
    "  Agent: What is your main contribution?\n"
    "  User: We did some improvements to the model.\n"
    "  Agent: What specifically improved and by how much?\n"
    "  User: It's just better overall, we ran some tests.\n"
    "  -> clarity_structure: 2, evidence_specificity: 1, logical_coherence: 2, handling_adversarial_questions: 1\n\n"

    "Example B — Adequate performance (scores mostly 3):\n"
    "  Agent: What is your main contribution?\n"
    "  User: We improved accuracy by 15%.\n"
    "  Agent: Compared to what baseline?\n"
    "  User: Against a standard baseline, I think logistic regression. The metric was accuracy.\n"
    "  -> clarity_structure: 3, evidence_specificity: 3, logical_coherence: 3, handling_adversarial_questions: 3\n\n"

    "Example C — Strong performance (scores mostly 4-5):\n"
    "  Agent: What is your main contribution?\n"
    "  User: We improve F1-score by 12% over a logistic regression baseline on the IMDB dataset.\n"
    "  Agent: Why F1 and not accuracy? Your dataset may be imbalanced.\n"
    "  User: Good point — the dataset has a 60/40 split, so F1 better reflects minority class performance. Accuracy also improved by 8%.\n"
    "  -> clarity_structure: 5, evidence_specificity: 4, logical_coherence: 5, handling_adversarial_questions: 5\n\n"
)


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
    Shows turns in chronological order to avoid recency bias.
    """
    if not turns:
        return "TURN_EVIDENCE: (none)\n"

    subset = turns[-max_items:]
    lines = ["TURN_EVIDENCE (chronological order):"]
    for t in subset:
        role = str(t.get("role", "unknown"))
        content = _truncate(str(t.get("content", "")), max_chars=max_chars)
        lines.append(f"- {role}: {content}")
    return "\n".join(lines) + "\n"


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
        f"- silence_ratio: {voice_summary.get('silence_ratio'):.2f}\n"
        f"- pitch_range_semitones: {voice_summary.get('pitch_range_semitones'):.1f}\n"
        f"- volume_std_db: {voice_summary.get('volume_std_db'):.1f}\n"
        f"- delivery_feedback: {voice_summary.get('delivery_feedback', [])}\n"
    )


def build_scoring_prompt(
    *,
    session_summary: Optional[Any],
    turns: Optional[list[dict[str, Any]]] = None,
    voice_summary: Optional[dict[str, Any]] = None,
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
    voice_block = _render_voice_summary(voice_summary)

    user = (
        "Task: Score this adversarial presentation practice session.\n\n"
        "You must return ONLY valid JSON matching the schema.\n"
        "For each rubric dimension, write a short 'reasoning' sentence FIRST, then assign the integer score (1-5).\n\n"
        f"{RUBRIC_DIMENSIONS}\n"
        f"{FEW_SHOT_EXAMPLES}\n"
        "- vocal_delivery: rate on the SAME 1-5 scale as the other rubrics, using only VOICE_SUMMARY if provided.\n"
        "- If VOICE_SUMMARY is missing, treat vocal_delivery as not applicable for this session\n"
        "- Do NOT infer vocal delivery problems from transcript text alone.\n"
        "- In text-only sessions, score the content rubrics normally and do not provide a vocal_delivery score.\n"
        "overall_score guidance:\n"
        "Constraints:\n"
        "- Base scores ONLY on the provided session summary, turn evidence, and VOICE_SUMMARY if present.\n"
        "- If evidence is insufficient for a dimension, score it lower rather than guessing.\n"
        "- Provide short, actionable notes.\n"
        "- Weigh ALL turns equally — do not favour recent turns over earlier ones.\n\n"
        f"{summary_block}\n"
        f"{turns_block}\n"
        f"{voice_block}\n"
        "Return ONLY the JSON object."
    )

    return {"system": system, "user": user}
