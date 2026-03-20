from __future__ import annotations

from typing import Any, Optional

from reasoning.prompts._base import (
    json_only_system,
    render_claims,
    render_common_ground,
    safe_user_input_block,
)

CONTRADICTION_SCHEMA_HINT = """
{
  "status": "true_contradiction|needs_clarification|no_conflict",
  "action": "clarify|update|ignore",
  "current_claim": string,
  "prior_claim": string|null,
  "prior_claim_id": string|null,
  "explanation": string
}
""".strip()


def build_contradiction_judge_prompt(
    *,
    current_claim: str,
    candidate_claims: list[Any],
    classification: Optional[Any] = None,
    common_ground: Optional[list[Any]] = None,
    max_candidates: int | None = None,
) -> dict[str, str]:
    system = json_only_system(CONTRADICTION_SCHEMA_HINT)

    claims_block = render_claims(candidate_claims,max_items=max_candidates)
    cg_block = render_common_ground(common_ground)

    cls_block = ""
    if classification is not None:
        cls_block = (
            "CLASSIFICATION_SIGNAL:\n"
            f"- response_class: {getattr(classification, 'response_class', None)}\n"
            f"- alignment: {getattr(classification, 'alignment', None)}\n\n"
        )

    user = (
        "Task: Determine whether CURRENT_CLAIM directly contradicts a prior claim.\n\n"
        "Decision policy:\n"
        "- true_contradiction: the current claim is LOGICALLY INCOMPATIBLE with a prior claim.\n"
        "  This means one claim explicitly negates or is mutually exclusive with the other.\n"
        "  Example: prior='SAGA is not used' vs current='SAGA is used'.\n"
        "  NOT a contradiction: elaboration, specification, or adding detail to a prior claim.\n"
        "  NOT a contradiction: two claims about the same topic that could both be true.\n"
        "  NOT a contradiction: using different terminology for compatible concepts.\n"
        "- needs_clarification: the claims appear contradictory but context could reconcile them.\n"
        "  Only use this if you can identify a specific sentence-level logical tension.\n"
        "- no_conflict: the claims are compatible, additive, complementary, or about different aspects.\n"
        "  When in doubt, choose no_conflict.\n\n"
        "Action policy:\n"
        "- clarify: use ONLY with true_contradiction or needs_clarification.\n"
        "- update: use ONLY with true_contradiction when the conflict is unambiguous.\n"
        "- ignore: use with no_conflict.\n\n"
        "Constraints:\n"
        "- Use only PRIOR_CLAIMS and COMMON_GROUND provided here.\n"
        "- If you select prior_claim_id, it must exist in PRIOR_CLAIMS.\n"
        "- Keep explanation <= 2 sentences.\n"
        "- If no single prior claim is a clear logical inverse of the current claim, output no_conflict.\n\n"
        f"{claims_block}\n"
        f"{cg_block}\n"
        f"{cls_block}"
        f"{safe_user_input_block(current_claim)}\n"
        "Return ONLY the JSON object."
    )

    return {"system": system, "user": user}
