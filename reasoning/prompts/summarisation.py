"""
Prompt builder for session summarisation (end of session).

Output is structured JSON so it can be stored in Episodic Memory and used for scoring.

Important:
- Do NOT ask the LLM to produce timestamps or session IDs.
- Do NOT dump full transcripts into the prompt; provide compact turns/claims.

The summarise node will convert this output into your SessionRecord by adding:
- session_id
- timestamp
- duration_seconds
- claims_count
"""

from __future__ import annotations

from typing import Any, Optional

from reasoning.prompts._base import json_only_system, render_memory_bundle, _truncate  # _truncate is internal but OK here


SUMMARISATION_SCHEMA_HINT = """
{
  "strengths": [string],
  "weaknesses": [string],
  "key_claims": [string],
  "open_issues": [string],
  "contradictions_detected": integer,
  "overall_notes": string
}
""".strip()

SUMMARISATION_ONE_SHOT = """
Example:
TURNS:
- user: We improved F1-score by 10% on the benchmark.
- assistant: Compared to what baseline?
- user: Compared to logistic regression, but I do not remember the exact baseline value.

CLAIMS:
- [sess_x-1] alignment=unsupported: We improved F1-score by 10% on the benchmark.

JSON:
{
  "strengths": ["States the main performance claim clearly"],
  "weaknesses": ["Does not provide the baseline value when challenged"],
  "key_claims": ["The system improved F1-score by 10% over a baseline"],
  "open_issues": ["What exact baseline score was used?"],
  "contradictions_detected": 0,
  "overall_notes": "The user presents the main claim clearly but lacks supporting detail when probed. Future practice should focus on baselines and quantitative grounding."
}
""".strip()

def _render_turns(turns: Optional[list[dict[str, Any]]], *, max_items: int = 16, max_chars: int = 280) -> str:
    """
    Render recent turns compactly for summarisation.
    Each turn dict is expected to have keys like {"role": "...", "content": "..."}.
    """
    if not turns:
        return "TURNS: (none)\n"

    # Prefer the most recent turns
    subset = turns[-max_items:]
    lines = ["TURNS (most recent first):"]
    for t in reversed(subset):
        role = str(t.get("role", "unknown"))
        content = _truncate(str(t.get("content", "")), max_chars=max_chars)
        lines.append(f"- {role}: {content}")
    return "\n".join(lines) + "\n"


def _render_claims(claims: Optional[list[Any]], *, max_items: int = 12, max_chars: int = 240) -> str:
    """
    Render extracted claims (ClaimRecord) compactly.
    """
    if not claims:
        return "CLAIMS: (none)\n"

    subset = claims[-max_items:]
    lines = ["CLAIMS (most recent first):"]
    for c in reversed(subset):
        # duck-typed: works with Pydantic models or dicts
        claim_id = getattr(c, "claim_id", None) or (c.get("claim_id") if isinstance(c, dict) else None)
        alignment = getattr(c, "alignment", None) or (c.get("alignment") if isinstance(c, dict) else None)
        text = getattr(c, "claim_text", None) or (c.get("claim_text") if isinstance(c, dict) else "")
        lines.append(f"- [{claim_id}] alignment={alignment}: {_truncate(str(text), max_chars=max_chars)}")
    return "\n".join(lines) + "\n"


def build_summarisation_prompt(
    *,
    turns: Optional[list[dict[str, Any]]] = None,
    claims: Optional[list[Any]] = None,
    memory_bundle: Optional[Any] = None,
) -> dict[str, str]:
    """
    Build system+user prompt for session summarisation.

    Inputs:
      turns: list of turn dicts accumulated in SessionState
      claims: list of ClaimRecord accumulated in SessionState
      memory_bundle: optional MemoryBundle (useful for continuity / recurring weaknesses)

    Output:
      {"system": ..., "user": ...}
    """
    system = json_only_system(SUMMARISATION_SCHEMA_HINT)

    context = render_memory_bundle(memory_bundle)

    turns_block = _render_turns(turns)
    claims_block = _render_claims(claims)

    user = (
        "Task: Summarise this practice session for later coaching.\n\n"
        "You must return ONLY valid JSON matching the schema.\n"
        "Keep lists short and actionable.\n\n"
        "Definitions:\n"
        "- strengths: what the user did well (argument clarity, evidence, structure, definitions, handling questions)\n"
        "- weaknesses: what needs improvement (vagueness, missing evidence, unclear terms, logical gaps)\n"
        "- key_claims: the main substantive claims the user asserted (as short sentences)\n"
        "- open_issues: unanswered questions / missing evidence / unclear definitions that should be revisited\n"
        "- contradictions_detected: count of apparent contradictions mentioned/flagged during the session (0 if none)\n"
        "- overall_notes: 2-4 sentences overall evaluation for internal use\n\n"
        "Constraints:\n"
        "- Do NOT invent new facts beyond the turns/claims provided.\n"
        "- Prefer concrete phrasing (e.g., 'Define X precisely', 'Provide evidence for Y') over generic advice.\n\n"
        f"{SUMMARISATION_ONE_SHOT}\n\n"
        f"{context}\n"
        f"{turns_block}\n"
        f"{claims_block}\n"
        "Return ONLY the JSON object."
    )

    return {"system": system, "user": user}