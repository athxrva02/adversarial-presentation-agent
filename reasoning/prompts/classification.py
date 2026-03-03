"""
Prompt builder for classifying a user's turn.

Purpose:
- Classify the user's response quality (strong/weak/contradiction/evasion)
- Provide an alignment label vs document/common ground (supported/contradicted/unsupported/novel)
- Return a short rationale (for debugging / optional UI)

This prompt is designed to be stable on small local instruct models.
"""

from __future__ import annotations

from typing import Any, Optional

from reasoning.prompts._base import json_only_system, render_memory_bundle, safe_user_input_block

# Keep schema hints compact to improve JSON compliance.
# Matches Classification Schema in storage/schemas.py according to the docs/technical-plan.md.
CLASSIFICATION_SCHEMA_HINT = """
{
  "response_class": "strong|weak|contradiction|evasion",
  "alignment": "supported|contradicted|unsupported|novel|negotiated",
  "confidence": number, 
  "reasoning": string
}
""".strip()


def build_classification_prompt(
    *,
    utterance: str,
    memory_bundle: Optional[Any] = None,
) -> dict[str, str]:
    """
    Build system+user prompt for classifying the current utterance.

    Inputs:
      utterance: current user text
      memory_bundle: MemoryBundle returned by Memory Module (may be None early)

    Output:
      {"system": ..., "user": ...}
    """
    system = json_only_system(CLASSIFICATION_SCHEMA_HINT)

    # The classification should be grounded. We provide compact memory bundle context.
    # Important: do not include huge transcripts. Keep it short.
    context = render_memory_bundle(memory_bundle)

    user = (
        "Task: classify the user's latest response.\n"
        "Guidelines:\n"
        "- response_class:\n"
        "  - strong: clear claim + rationale/evidence, answers the question directly\n"
        "  - weak: vague, missing evidence, unclear definitions, partial answer\n"
        "  - contradiction: conflicts with provided document context or prior common ground/claims\n"
        "  - evasion: avoids answering, changes topic, refuses without justification\n"
        "- alignment:\n"
        "  - supported: consistent with provided document context\n"
        "  - contradicted: directly conflicts with document context or common ground\n"
        "  - unsupported: makes factual/technical assertion without support in provided context\n"
        "  - novel: new claim not present in context (not necessarily wrong)\n"
        "  - negotiated: explicitly aligns with a negotiated common-ground entry\n"
        "- confidence: 0.0 to 1.0\n"
        "- reasoning: 1-3 sentences, refer to IDs when relevant\n\n"
        f"{context}\n"
        f"{safe_user_input_block(utterance)}\n"
        "Return ONLY the JSON object."
    )

    return {"system": system, "user": user}