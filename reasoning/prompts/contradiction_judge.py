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
    max_candidates: int | None = None, # Fix: Design Issue 3: Unranked, undocumented claim truncation 
) -> dict[str, str]:
    system = json_only_system(CONTRADICTION_SCHEMA_HINT)

    claims_block = render_claims(candidate_claims,max_items=max_candidates) # Fix: Design Issue 3: Unranked, undocumented claim truncation 
    cg_block = render_common_ground(common_ground)

    cls_block = ""
    if classification is not None:
        cls_block = (
            "CLASSIFICATION_SIGNAL:\n"
            f"- response_class: {getattr(classification, 'response_class', None)}\n"
            f"- alignment: {getattr(classification, 'alignment', None)}\n\n"
        )

    user = (
        "Task: Determine whether CURRENT_CLAIM contradicts a prior claim.\n\n"
        "Decision policy:\n"
        "- true_contradiction: explicit logical conflict with a prior claim.\n"
        "- needs_clarification: possible conflict, but ambiguity prevents judgment.\n"
        "- no_conflict: compatible, unrelated, or insufficient evidence of conflict.\n\n"
        "Action policy:\n"
        "- clarify: ask user to reconcile or disambiguate.\n"
        "- update: conflict is clear enough that common-ground update is recommended.\n"
        "- ignore: no contradiction action needed.\n\n"
        "Constraints:\n"
        "- Use only PRIOR_CLAIMS and COMMON_GROUND provided here.\n"
        "- If you select prior_claim_id, it must exist in PRIOR_CLAIMS.\n"
        "- Keep explanation <= 2 sentences.\n\n"
        f"{claims_block}\n"
        f"{cg_block}\n"
        f"{cls_block}"
        f"{safe_user_input_block(current_claim)}\n"
        "Return ONLY the JSON object."
    )

    return {"system": system, "user": user}
