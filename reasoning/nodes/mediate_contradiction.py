# from __future__ import annotations
# from typing import Any, Dict
# from reasoning.state import SessionState
# from storage.schemas import ConflictStatus

# def run(state: SessionState) -> Dict[str, Any]:
#     conflict = state.get("conflict_result")
#     if conflict is None or getattr(conflict, "status", None) != ConflictStatus.TRUE_CONTRADICTION:
#         return {"agent_response": ""}

#     prior = str(getattr(conflict, "prior_claim", "") or "").strip()
#     current = str(getattr(conflict, "current_claim", "") or state.get("user_input", "")).strip()

#     if prior:
#         q = (
#             f"You previously said: '{prior}'. Now you said: '{current}'. "
#             "Which statement is correct, and under what condition could both be true?"
#         )
#     else:
#         q = (
#             f"You now said: '{current}'. "
#             "What concrete evidence supports this, and how does it reconcile with your earlier statements?"
#         )
#     return {"agent_response": q}

# Fix:Design Issue 1:Hardcoded stub in the hot path
# I replaced the hardcoded mediation templates with an LLM-driven contradiction question so this hot path is adaptive like the rest of the reasoning nodes. 
# The node now uses prior_claim, current_claim, and explanation to build a focused prompt and generate one reconciliation question, then lightly sanitizes 
# the output for UI safety. If the LLM call fails or returns empty text, it falls back to the previous deterministic template, so behavior stays robust while 
# removing the “canned response” design issue.
from __future__ import annotations

from typing import Any, Dict

from reasoning.llm import call_llm_text, opts_practice_question
from reasoning.prompts._base import safe_user_input_block, text_system
from reasoning.state import SessionState
from storage.schemas import ConflictStatus


def _build_mediation_prompt(
    *,
    prior_claim: str,
    current_claim: str,
    explanation: str,
) -> dict[str, str]:
    system = (
        text_system()
        + "\nYou are mediating a contradiction in a live adversarial coaching session.\n"
        + "Output ONLY one concise reconciliation question.\n"
    )

    user = (
        "Task: Ask exactly ONE reconciliation question that helps the user resolve or disambiguate the conflict.\n\n"
        "CONTRADICTION_CONTEXT:\n"
        f"- prior_claim: {prior_claim or 'null'}\n"
        f"- current_claim: {current_claim or 'null'}\n"
        f"- explanation: {explanation or 'none'}\n\n"
        "Constraints:\n"
        "- Ask exactly one question.\n"
        "- Keep it <= 2 sentences.\n"
        "- Force reconciliation/disambiguation (not generic elaboration).\n"
        "- Output only the question text.\n\n"
        f"{safe_user_input_block(current_claim)}\n"
        "Now output the best reconciliation question."
    )
    return {"system": system, "user": user}


def _clean_question(text: str) -> str:
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text.strip()

    q = lines[0]
    for prefix in ("Question:", "Q:", "- "):
        if q.startswith(prefix):
            q = q[len(prefix):].strip()

    return q


def _fallback_question(prior: str, current: str) -> str:
    if prior:
        return (
            f"You previously said: '{prior}'. Now you said: '{current}'. "
            "Which statement is correct, and under what condition could both be true?"
        )
    return (
        f"You now said: '{current}'. "
        "What concrete evidence supports this, and how does it reconcile with your earlier statements?"
    )


def run(state: SessionState) -> Dict[str, Any]:
    conflict = state.get("conflict_result")
    if conflict is None or getattr(conflict, "status", None) != ConflictStatus.TRUE_CONTRADICTION:
        return {"agent_response": ""}

    prior = str(getattr(conflict, "prior_claim", "") or "").strip()
    current = str(getattr(conflict, "current_claim", "") or state.get("user_input", "")).strip()
    explanation = str(getattr(conflict, "explanation", "") or "").strip()

    prompt = _build_mediation_prompt(
        prior_claim=prior,
        current_claim=current,
        explanation=explanation,
    )

    try:
        raw = call_llm_text(
            system_prompt=prompt["system"],
            user_prompt=prompt["user"],
            options=opts_practice_question(),
        )
        question = _clean_question(raw)
    except Exception:
        question = ""

    if not question:
        question = _fallback_question(prior, current)

    return {"agent_response": question}
