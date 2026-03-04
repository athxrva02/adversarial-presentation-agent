"""
Prompt builder for generating an adversarial follow-up question.

Goal:
- Produce ONE high-value follow-up question that:
  - probes assumptions
  - asks for evidence
  - requests definition/clarification
  - checks implications or boundary cases
- Stay grounded in provided document context + retrieved memory
- Keep it short and direct for live practice

Output:
- Plain text: a single question (no preamble)
"""

from __future__ import annotations

from typing import Any, Optional

from reasoning.prompts._base import (
    text_system,
    render_memory_bundle,
    safe_user_input_block,
)

def build_question_generation_prompt(
    *,
    utterance: str,
    memory_bundle: Optional[Any] = None,
    classification: Optional[Any] = None,
) -> dict[str, str]:
    """
    Build system+user prompt for adversarial question generation.

    Inputs:
      utterance: current user response
      memory_bundle: MemoryBundle (document_context, prior claims, patterns, common ground)
      classification: optional Classification object (response_class, alignment, reasoning)

    Output:
      {"system": ..., "user": ...}
    """
    system = (
        text_system()
        + "\nYou are conducting adversarial Q&A practice.\n"
          "Output ONLY a single question. No bullet points, no explanations.\n"
          "The question must be specific and test understanding.\n"
    )

    context = render_memory_bundle(memory_bundle)

    # Provide compact classification signal if available (helps focus).
    cls_block = ""
    if classification is not None:
        # duck-typed
        response_class = getattr(classification, "response_class", None)
        alignment = getattr(classification, "alignment", None)
        conf = getattr(classification, "confidence", None)
        reasoning = getattr(classification, "reasoning", None)

        cls_block = (
            "CLASSIFICATION_SIGNAL:\n"
            f"- response_class: {response_class}\n"
            f"- alignment: {alignment}\n"
            f"- confidence: {conf}\n"
            f"- notes: {reasoning}\n\n"
        )

    user = (
        "Task: Ask exactly ONE adversarial follow-up question to the user's latest response.\n\n"
        "Strategy selection (pick ONE best):\n"
        "1) If the user's response is EVASION (dodging the previous question):\n"
        "   - Redirect back to the missing information with a forced-choice or concrete request.\n"
        "2) If the user gave a vague response:\n"
        "   - Ask for an operational definition AND how it is measured or a concrete example.\n"
        "3) If the user gave a quantitative improvement claim (%, ×, faster, better):\n"
        "   - Ask for: baseline, metric, dataset, and evaluation protocol (split / CV), and controls for leakage.\n"
        "4) If the user provided a definition already:\n"
        "   - Ask for: which groups/conditions, how computed, and trade-offs or edge cases.\n"
        "5) If the user provided evidence/method already:\n"
        "   - Ask for: assumptions, failure modes, boundary cases, and what would falsify the claim.\n"
        "6) If memory shows a recurring weakness: target that weakness with a specific probe.\n"
        "7) If the response references document content: challenge consistency, edge cases, or implications.\n\n"
        "Constraints:\n"
        "- Ask exactly ONE question.\n"
        "- Keep it <= 2 sentences.\n"
        "- Avoid generic 'can you elaborate' phrasing.\n"
        "- Do not mention internal modules, memory, embeddings, or scoring.\n\n"
        f"{context}\n"
        f"{cls_block}"
        f"{safe_user_input_block(utterance)}\n"
        "Now output the single best follow-up question."
    )

    return {"system": system, "user": user}