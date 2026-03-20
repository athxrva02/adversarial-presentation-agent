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
#Improving question generation change1: giving the agent context of the previous questions to make more sense of the answer
def _get_previous_agent_question(turns: list[dict[str, Any]]) -> str:
    if not turns:
        return ""

    # Current state already contains the latest user turn.
    # Walk backwards and find the most recent agent turn before it.
    for turn in reversed(turns[:-1]):
        if str(turn.get("role", "")).strip().lower() == "agent":
            return str(turn.get("content", "")).strip()
    return ""
#end of change1
#change3 Pass focused context, not the whole memory bundle
from reasoning.prompts._base import (
    render_document_context,
    render_claims,
    render_semantic_patterns,
    render_common_ground,
)
def _build_focused_context(memory_bundle: Any) -> str:
    if memory_bundle is None:
        return "FOCUSED_CONTEXT: (none)\n"

    parts = ["FOCUSED_CONTEXT:"]
    parts.append(render_document_context(getattr(memory_bundle, "document_context", [])[:2]))
    parts.append(render_claims(getattr(memory_bundle, "episodic_claims", [])[:3], max_items=3))
    parts.append(render_semantic_patterns(getattr(memory_bundle, "semantic_patterns", [])[:2], max_items=2))
    parts.append(render_common_ground(getattr(memory_bundle, "common_ground", [])[:2], max_items=2))
    return "\n".join(parts)
#end of change3
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
    #change1
    turns = list(state.get("turns", []) or [])
    previous_question = _get_previous_agent_question(turns)
    #end of change1
    focused_context = _build_focused_context(memory_bundle) #change3

    prompt = build_question_generation_prompt(
        utterance=utterance,
        memory_bundle=memory_bundle,
        classification=classification,
        conflict_result=conflict_result,
        previous_question=previous_question, #change1
        focused_context=focused_context, #change3
    )

    raw = call_llm_text(
        system_prompt=prompt["system"],
        user_prompt=prompt["user"],
        options=opts_practice_question(),
    )

    question = _clean_question(raw)

    return {"agent_response": question}