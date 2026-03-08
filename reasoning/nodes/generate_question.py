"""
LangGraph node: generate an adversarial follow-up question.

Responsibilities:
- Build question-generation prompt using the current user_input + memory_bundle (+ optional classification)
- Call the LLM as plain text (NOT JSON)
- Return updated state fragment with agent_response
"""

from __future__ import annotations

from typing import Any, Dict

from reasoning.state import SessionState
from reasoning.llm import call_llm_text, opts_practice_question
from reasoning.prompts.question_generation import build_question_generation_prompt


def _clean_question(text: str) -> str:
    """
    Make the output robust against models adding preambles or extra lines.
    We keep only the first non-empty line and strip common prefixes.
    """
    if not text:
        return ""

    # Split into lines, keep first non-empty
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text.strip()

    first = lines[0]

    # Remove common prefixes the model might add
    prefixes = ("Question:", "Q:", "- ", "• ")
    for p in prefixes:
        if first.startswith(p):
            first = first[len(p):].strip()

    # Ensure it ends with a question mark if it's clearly a question without one
    if first and not first.endswith("?") and ("?" in first) is False:
        # Don't force '?' if the model produced something that isn't a question.
        # But usually we want it to be a question.
        # We'll add '?' if it begins with typical interrogatives.
        lowers = first.lower()
        if lowers.startswith(("why", "how", "what", "which", "when", "where", "who", "can", "could", "would", "do", "does", "did", "is", "are", "was", "were", "should")):
            first = first + "?"

    return first


def run(state: SessionState) -> Dict[str, Any]:
    """
    Generate one follow-up question.

    Expected state keys:
      - user_input: str
      - memory_bundle: Optional[MemoryBundle]
      - classification: Optional[Classification]
    """
    utterance = state.get("user_input", "")
    memory_bundle = state.get("memory_bundle")
    classification = state.get("classification")
    conflict_result = state.get("conflict_result")

    prompt = build_question_generation_prompt(
        utterance=utterance,
        memory_bundle=memory_bundle,
        classification=classification,
        conflict_result=conflict_result,
    )

    raw = call_llm_text(
        system_prompt=prompt["system"],
        user_prompt=prompt["user"],
        options=opts_practice_question(),
    )

    question = _clean_question(raw)

    return {"agent_response": question}