from __future__ import annotations
from typing import Any, Dict
from reasoning.state import SessionState
from storage.schemas import ConflictStatus

def run(state: SessionState) -> Dict[str, Any]:
    conflict = state.get("conflict_result")
    if conflict is None or getattr(conflict, "status", None) != ConflictStatus.TRUE_CONTRADICTION:
        return {"agent_response": ""}

    prior = str(getattr(conflict, "prior_claim", "") or "").strip()
    current = str(getattr(conflict, "current_claim", "") or state.get("user_input", "")).strip()

    if prior:
        q = (
            f"You previously said: '{prior}'. Now you said: '{current}'. "
            "Which statement is correct, and under what condition could both be true?"
        )
    else:
        q = (
            f"You now said: '{current}'. "
            "What concrete evidence supports this, and how does it reconcile with your earlier statements?"
        )
    return {"agent_response": q}
