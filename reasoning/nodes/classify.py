"""
LangGraph node: classify the user's current response.

Responsibilities:
- Build classification prompt
- Call LLM (structured)
- Parse into Classification model
- Append a ClaimRecord for this turn
- Return updated state fragment
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Any

from reasoning.state import SessionState
from reasoning.llm import call_llm_structured, opts_judge_or_classify
from reasoning.prompts.classification import build_classification_prompt
from storage.schemas import Classification, ClaimRecord


def run(state: SessionState) -> Dict[str, Any]:
    """
    Classify the user's current input.

    Expected state keys:
      - user_input: str
      - turn_number: int
      - memory_bundle: Optional[MemoryBundle]
      - session_id: Optional[str] (may be added by orchestrator)
    """

    utterance = state.get("user_input", "")
    memory_bundle = state.get("memory_bundle")

    # Build prompt
    prompt = build_classification_prompt(
        utterance=utterance,
        memory_bundle=memory_bundle,
    )

    # Call LLM with structured parsing + retry
    raw = call_llm_structured(
        system_prompt=prompt["system"],
        user_prompt=prompt["user"],
        schema_hint="""
        {
          "response_class": "strong|weak|contradiction|evasion",
          "alignment": "supported|contradicted|unsupported|novel|negotiated",
          "confidence": number,
          "reasoning": string
        }
        """,
        options=opts_judge_or_classify(),
    )

    # Validate via Pydantic model
    classification = Classification(**raw)

    # Create claim record for this turn
    session_id = state.get("session_id", "unknown_session")

    claim_record = ClaimRecord(
        claim_id=f"{session_id}-{state['turn_number']}",
        session_id=session_id,
        turn_number=state["turn_number"],
        claim_text=utterance,
        alignment=classification.alignment,
        mapped_to_slide=None,  # can be filled later by retrieval logic if needed
        prior_conflict=None,
        timestamp=datetime.now(),
    )

    return {
        "classification": classification,
        "claims": [claim_record],  # LangGraph will append via Annotated[list, add]
    }