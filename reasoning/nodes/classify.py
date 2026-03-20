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
from reasoning.llm import call_llm_structured, call_llm_text, opts_judge_or_classify, LLMOptions
from reasoning.prompts.classification import build_classification_prompt
from storage.schemas import Classification, ClaimRecord


# Helpers

import logging

_logger = logging.getLogger(__name__)

_VALID_RESPONSE_CLASS = {"strong", "weak", "contradiction", "evasion"}
_VALID_ALIGNMENT = {"supported", "contradicted", "unsupported", "novel", "negotiated"}

def _clamp01(x: object, default: float = 0.5) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _normalize_classification(raw: dict) -> dict:
    """
    Make LLM output safe for Pydantic enums.

    Handles common model failure modes:
    - response_class accidentally set to an alignment value (e.g. "novel")
    - alignment accidentally set to a response_class value (e.g. "weak")
    - missing fields / nulls
    """
    out = dict(raw) if isinstance(raw, dict) else {}

    rc = out.get("response_class")
    al = out.get("alignment")

    # Normalize to lowercase strings if present
    if isinstance(rc, str):
        rc = rc.strip().lower()
    if isinstance(al, str):
        al = al.strip().lower()

    # If swapped/misplaced values, fix them.
    # Case 1: response_class is actually an alignment label.
    if isinstance(rc, str) and rc in _VALID_ALIGNMENT and (not isinstance(al, str) or al not in _VALID_ALIGNMENT):
        al = rc
        rc = None

    # Case 2: alignment is actually a response_class label.
    if isinstance(al, str) and al in _VALID_RESPONSE_CLASS and (not isinstance(rc, str) or rc not in _VALID_RESPONSE_CLASS):
        rc = al
        al = None

    # Defaulting
    if not isinstance(rc, str) or rc not in _VALID_RESPONSE_CLASS:
        rc = "weak"
    if not isinstance(al, str) or al not in _VALID_ALIGNMENT:
        al = "novel"

    out["response_class"] = rc
    out["alignment"] = al
    out["confidence"] = _clamp01(out.get("confidence"), default=0.5)

    reasoning = out.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        out["reasoning"] = "No reasoning provided."
    else:
        out["reasoning"] = reasoning.strip()

    return out

def _clean_claim_text(utterance: str) -> str:
    """
    Fix grammar and spelling in user utterance before storing as a claim.

    Uses a fast, low-token LLM call. On failure, returns the original text.
    """
    text = utterance.strip()
    if not text or len(text) < 5:
        return text
    try:
        result = call_llm_text(
            system_prompt=(
                "You are a grammar corrector. Fix grammar, spelling, and punctuation errors "
                "in the user's text. Preserve the original meaning exactly. "
                "Do NOT add information, change the meaning, or elaborate. "
                "Output ONLY the corrected text, nothing else."
            ),
            user_prompt=text,
            options=LLMOptions(temperature=0.0, num_predict=150, num_ctx=2048),
        )
        cleaned = result.strip()
        # Sanity check: if the LLM returned something wildly different or empty, keep original
        if not cleaned or len(cleaned) > len(text) * 3:
            return text
        return cleaned
    except Exception:
        _logger.debug("Grammar cleanup failed, using original text", exc_info=True)
        return text


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

    raw = raw if isinstance(raw, dict) else {}
    raw = _normalize_classification(raw)

    # Coerce missing/invalid enum fields to safe defaults
    if raw.get("response_class") is None:
        raw["response_class"] = "weak"
    if raw.get("alignment") is None:
        raw["alignment"] = "novel"
    if raw.get("confidence") is None:
        raw["confidence"] = 0.5
    if raw.get("reasoning") is None:
        raw["reasoning"] = "No reasoning provided."
    
    # Validate via Pydantic model
    classification = Classification(**raw)

    # Create claim record for this turn
    session_id = state.get("session_id", "unknown_session")
    clean_text = _clean_claim_text(utterance)

    claim_record = ClaimRecord(
        claim_id=f"{session_id}-{state['turn_number']}",
        session_id=session_id,
        turn_number=state["turn_number"],
        claim_text=clean_text,
        alignment=classification.alignment,
        mapped_to_slide=None,  # can be filled later by retrieval logic if needed
        prior_conflict=None,
        timestamp=datetime.now(),
    )

    return {
        "classification": classification,
        "claims": [claim_record],  # LangGraph will append via Annotated[list, add]
    }