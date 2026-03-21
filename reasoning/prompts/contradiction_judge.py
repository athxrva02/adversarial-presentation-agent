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

CONTRADICTION_FEW_SHOTS = """
Examples:

Example 1:
PRIOR_CLAIMS:
- [p1] session=s1 turn=1 alignment=novel
  We use SAGA in the optimizer.

USER_INPUT:
We do not use SAGA in this system.

JSON:
{
  "status": "true_contradiction",
  "action": "update",
  "current_claim": "We do not use SAGA in this system.",
  "prior_claim": "We use SAGA in the optimizer.",
  "prior_claim_id": "p1",
  "explanation": "The current claim directly negates the prior claim."
}

Example 2:
PRIOR_CLAIMS:
- [p1] session=s1 turn=1 alignment=novel
  Our model is fast.

USER_INPUT:
Our model runs in under 50 ms on CPU.

JSON:
{
  "status": "no_conflict",
  "action": "ignore",
  "current_claim": "Our model runs in under 50 ms on CPU.",
  "prior_claim": null,
  "prior_claim_id": null,
  "explanation": "The current claim adds detail and does not contradict the prior claim."
}

Example 3:
PRIOR_CLAIMS:
- [p1] session=s1 turn=1 alignment=novel
  The method works well for small datasets.

USER_INPUT:
The method may fail on very small datasets.

JSON:
{
  "status": "needs_clarification",
  "action": "clarify",
  "current_claim": "The method may fail on very small datasets.",
  "prior_claim": "The method works well for small datasets.",
  "prior_claim_id": "p1",
  "explanation": "The claims appear in tension but may be reconcilable depending on what counts as small."
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
        f"{CONTRADICTION_FEW_SHOTS}\n\n"
        f"{claims_block}\n"
        f"{cg_block}\n"
        f"{cls_block}"
        f"{safe_user_input_block(current_claim)}\n"
        "Return ONLY the JSON object."
    )

    return {"system": system, "user": user}
