from __future__ import annotations

from typing import Any, Dict

from reasoning.llm import call_llm_structured, opts_judge_or_classify
from reasoning.prompts.contradiction_judge import (
    CONTRADICTION_SCHEMA_HINT,
    build_contradiction_judge_prompt,
)
from reasoning.state import SessionState
from storage.schemas import ConflictAction, ConflictResult, ConflictStatus
import logging
logger = logging.getLogger(__name__)


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
    candidate_claims = sorted(
        candidate_claims,
        key=lambda c: getattr(c, "turn_number", 0),
        reverse=True,
    )[:8]
    common_ground = list(getattr(memory_bundle, "common_ground", []) or [])
    classification = state.get("classification")

    if not current_claim:
        logger.debug("detect_contradiction: empty user input, skipping")
        return {
            "conflict_result": _default_conflict("", "Empty user input."),
            "conflict_prior_claim_id": None,
        }

    if not candidate_claims:
        logger.debug("detect_contradiction: no prior claims available (turn may be first)")
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
        max_candidates=len(candidate_claims),
    )

    try:
        raw = call_llm_structured(
            system_prompt=prompt["system"],
            user_prompt=prompt["user"],
            schema_hint=CONTRADICTION_SCHEMA_HINT,
            options=opts_judge_or_classify(),
        )
    except Exception:
        logger.warning("Contradiction check failed; defaulting to no_conflict.", exc_info=True)
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
        # First try the in-memory candidate list (fast, already fetched)
        match = next(
            (c for c in candidate_claims if getattr(c, "claim_id", None) == prior_claim_id),
            None,
        )
        if match is not None:
            prior_claim = getattr(match, "claim_text", prior_claim)
        else:
            # The LLM returned a prior_claim_id that wasn't in the similarity
            # search results (common in session 2+ when the conflicting claim is
            # from a previous session and didn't rank in the top-k candidates).
            # Fall back to a direct SQLite lookup by exact ID so that
            # ConflictResult.prior_claim is always populated when the ID is valid.
            mm = state.get("_memory_module")
            if mm is not None:
                try:
                    row = mm._rel.get_claim(prior_claim_id)
                    if row is not None:
                        prior_claim = row.get("claim_text") or prior_claim
                except Exception:
                    pass

    result = ConflictResult(
        status=ConflictStatus(status),
        action=ConflictAction(action),
        current_claim=current_claim,
        prior_claim=str(prior_claim).strip() if prior_claim else None,
        explanation=str(raw.get("explanation") or "No explanation provided.").strip(),
    )

    logger.info(
        "detect_contradiction result: status=%s, action=%s, prior_claim_id=%s, explanation=%s",
        result.status.value, result.action.value, prior_claim_id, result.explanation,
    )

    # Only propagate conflict_prior_claim_id for TRUE contradictions.
    emit_prior_id = (
        prior_claim_id
        if isinstance(prior_claim_id, str) and status == "true_contradiction"
        else None
    )

    return {
        "conflict_result": result,
        "conflict_prior_claim_id": emit_prior_id,
    }
