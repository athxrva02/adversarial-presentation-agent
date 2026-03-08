from __future__ import annotations

from typing import Any, Dict

from reasoning.llm import call_llm_structured, opts_judge_or_classify
from reasoning.prompts.contradiction_judge import (
    CONTRADICTION_SCHEMA_HINT,
    build_contradiction_judge_prompt,
)
from reasoning.state import SessionState
from storage.schemas import ConflictAction, ConflictResult, ConflictStatus


def _default_conflict(current_claim: str, explanation: str) -> ConflictResult:
    return ConflictResult(
        status=ConflictStatus.NO_CONFLICT,
        action=ConflictAction.IGNORE,
        current_claim=current_claim,
        prior_claim=None,
        explanation=explanation,
    )


def _norm_status(v: Any) -> str:
    s = str(v or "").strip().lower().replace(" ", "_")
    if s in {"true_contradiction", "needs_clarification", "no_conflict"}:
        return s
    return "no_conflict"


def _norm_action(v: Any, status: str) -> str:
    s = str(v or "").strip().lower()
    if s in {"clarify", "update", "ignore"}:
        return s
    if status in {"true_contradiction", "needs_clarification"}:
        return "clarify"
    return "ignore"


def run(state: SessionState) -> Dict[str, Any]:
    current_claim = str(state.get("user_input", "")).strip()
    memory_bundle = state.get("memory_bundle")
    candidate_claims = list(getattr(memory_bundle, "episodic_claims", []) or [])
    common_ground = list(getattr(memory_bundle, "common_ground", []) or [])
    classification = state.get("classification")

    if not current_claim:
        return {
            "conflict_result": _default_conflict("", "Empty user input."),
            "conflict_prior_claim_id": None,
        }

    if not candidate_claims:
        return {
            "conflict_result": _default_conflict(
                current_claim,
                "No prior claims available for contradiction check.",
            ),
            "conflict_prior_claim_id": None,
        }

    prompt = build_contradiction_judge_prompt(
        current_claim=current_claim,
        candidate_claims=candidate_claims,
        classification=classification,
        common_ground=common_ground,
    )

    try:
        raw = call_llm_structured(
            system_prompt=prompt["system"],
            user_prompt=prompt["user"],
            schema_hint=CONTRADICTION_SCHEMA_HINT,
            options=opts_judge_or_classify(),
        )
    except Exception:
        return {
            "conflict_result": _default_conflict(
                current_claim,
                "Contradiction check failed; defaulting to no_conflict.",
            ),
            "conflict_prior_claim_id": None,
        }

    raw = raw if isinstance(raw, dict) else {}

    status = _norm_status(raw.get("status"))
    action = _norm_action(raw.get("action"), status)

    prior_claim_id = raw.get("prior_claim_id")
    prior_claim = raw.get("prior_claim")

    if isinstance(prior_claim_id, str):
        match = next(
            (c for c in candidate_claims if getattr(c, "claim_id", None) == prior_claim_id),
            None,
        )
        if match is not None:
            prior_claim = getattr(match, "claim_text", prior_claim)

    result = ConflictResult(
        status=ConflictStatus(status),
        action=ConflictAction(action),
        current_claim=current_claim,
        prior_claim=str(prior_claim).strip() if prior_claim else None,
        explanation=str(raw.get("explanation") or "No explanation provided.").strip(),
    )

    return {
        "conflict_result": result,
        "conflict_prior_claim_id": prior_claim_id if isinstance(prior_claim_id, str) else None,
    }
