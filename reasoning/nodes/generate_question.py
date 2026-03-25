"""
LangGraph node: generate an adversarial follow-up question.

Responsibilities:
- Build question-generation prompt using the current user_input + memory_bundle (+ optional classification)
- Call the LLM as plain text (NOT JSON)
- Return updated state fragment with agent_response
- Rotate through QUESTION_STRATEGIES to ensure diverse questioning across turns
"""

from __future__ import annotations

from typing import Any, Dict

from reasoning.state import SessionState
from reasoning.llm import call_llm_text, opts_practice_question
from reasoning.prompts.question_generation import build_question_generation_prompt

# Eight distinct attack angles — rotated across turns to prevent repetition.
QUESTION_STRATEGIES = [
    {
        "name": "evidence_specificity",
        "instruction": "Ask for a specific number, metric, dataset, or evaluation result that supports a claim the user just made.",
        "example": "What was the exact accuracy improvement, and on which test set?",
    },
    {
        "name": "baseline_comparison",
        "instruction": "Ask what baseline or alternative method the user is comparing against to justify their improvement claim.",
        "example": "Compared to which baseline method did you measure that improvement?",
    },
    {
        "name": "definition_probe",
        "instruction": "Pick one key technical term the user used and ask for its precise definition or how it is operationalised/measured.",
        "example": "How exactly do you define 'robustness' in your evaluation?",
    },
    {
        "name": "assumption_challenge",
        "instruction": "Identify one assumption embedded in the user's claim and ask what happens when that assumption breaks.",
        "example": "You assume the data distribution at test time matches training — what if it doesn't?",
    },
    {
        "name": "causal_reasoning",
        "instruction": "Challenge whether the causal link the user claims (A causes B) is actually causal or just correlational, and ask for evidence.",
        "example": "You claim X causes Y — what evidence rules out that Y is merely correlated with X?",
    },
    {
        "name": "boundary_case",
        "instruction": "Ask about a specific scenario, input type, or condition where the user's approach would fail or degrade significantly.",
        "example": "Under what conditions or input types does your method break down or perform significantly worse?",
    },
    {
        "name": "methodology_probe",
        "instruction": "Ask about a specific implementation or evaluation detail: how something was measured, validated, or controlled for confounds.",
        "example": "How did you ensure there was no data leakage between your training and test splits?",
    },
    {
        "name": "implication",
        "instruction": "Ask about the practical consequence, real-world impact, or broader implication of a specific claim the user just made.",
        "example": "If your model achieves that accuracy, what does it mean for deployment in a real clinical setting?",
    },
]

_STRATEGY_NAMES = [s["name"] for s in QUESTION_STRATEGIES]
# How many recent turns to consider when avoiding repeats
_RECENCY_WINDOW = 4


def _pick_strategy(used: list[str], conflict_active: bool) -> dict[str, str]:
    """
    Select the strategy least recently used in the last _RECENCY_WINDOW turns.
    If a contradiction is active, always start with assumption_challenge to vary
    the angle (contradiction reconciliation is already handled by the conflict block).
    """
    recent = set(used[-_RECENCY_WINDOW:]) if used else set()
    # Prefer strategies not used at all recently
    for strategy in QUESTION_STRATEGIES:
        if strategy["name"] not in recent:
            return strategy
    # All were used recently — fall back to the one used longest ago
    for name in used:
        for strategy in QUESTION_STRATEGIES:
            if strategy["name"] == name:
                return strategy
    return QUESTION_STRATEGIES[0]


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
    turns = list(state.get("turns", []) or [])
    previous_question = _get_previous_agent_question(turns)
    focused_context = _build_focused_context(memory_bundle)

    # Pick a diverse strategy, avoiding recent repeats
    used_strategies: list[str] = list(state.get("used_question_strategies") or [])
    conflict_active = (
        conflict_result is not None
        and getattr(conflict_result, "status", None) == "true_contradiction"
    )
    strategy = _pick_strategy(used_strategies, conflict_active=conflict_active)
    updated_strategies = (used_strategies + [strategy["name"]])[-8:]  # keep last 8

    prompt = build_question_generation_prompt(
        utterance=utterance,
        memory_bundle=memory_bundle,
        classification=classification,
        conflict_result=conflict_result,
        previous_question=previous_question,
        focused_context=focused_context,
        forced_strategy=strategy,
    )

    raw = call_llm_text(
        system_prompt=prompt["system"],
        user_prompt=prompt["user"],
        options=opts_practice_question(),
    )

    question = _clean_question(raw)

    return {"agent_response": question, "used_question_strategies": updated_strategies}