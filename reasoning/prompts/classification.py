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
        "Return ONLY valid JSON matching the schema.\n\n"
        "Guidelines:\n"
        "- response_class:\n"
        "  - strong: clear claim + rationale/evidence, answers directly\n"
        "  - weak: vague, missing evidence, unclear definitions, partial answer\n"
        "  - contradiction: conflicts with provided document context or common ground\n"
        "  - evasion: avoids answering or dodges the question\n"
        "- alignment:\n"
        "  - supported: ONLY if DOCUMENT_CONTEXT contains an explicit matching statement\n"
        "  - contradicted: ONLY if DOCUMENT_CONTEXT/COMMON_GROUND explicitly conflicts\n"
        "  - negotiated: ONLY if COMMON_GROUND explicitly matches\n"
        "  - novel: new definition, new proposal, or new qualitative claim not in context\n"
        "  - unsupported: empirical/quantitative claim, citation, or external fact asserted without support in context\n\n"
        "- confidence: the user's confidence in their answer\n"
        "Hard constraints:\n"
        "- response_class and alignment must NEVER be null.\n"
        "- If DOCUMENT_CONTEXT is (none) and COMMON_GROUND is (none), alignment MUST be either 'novel' or 'unsupported' (not 'supported', 'contradicted', or 'negotiated').\n"
        "- Do not invent chunk IDs, sessions, or prior records.\n"
        "- confidence: 0.0..1.0\n"
        "- reasoning: 1-2 sentences; only cite IDs if they exist in the provided context.\n\n"
        f"{context}\n"
        f"{safe_user_input_block(utterance)}\n"
        "Return ONLY the JSON object."
    )
    return {"system": system, "user": user}