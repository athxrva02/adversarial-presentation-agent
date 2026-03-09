from __future__ import annotations

import hashlib
from typing import Any, Dict
from uuid import uuid4

from reasoning.state import SessionState
from storage.schemas import ConflictStatus


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _stable_cg_id(text: str) -> str:
    digest = hashlib.sha1(_norm(text).encode("utf-8")).hexdigest()[:12]
    return f"cg_{digest}"


def _item(
    kind: str,
    proposed_text: str,
    source: str,
    rationale: str,
    default_decision: str,
    **extra: Any,
) -> dict[str, Any]:
    base = {
        "item_id": f"neg_{uuid4().hex[:10]}",
        "kind": kind,  # common_ground | semantic_strength | semantic_weakness
        "source": source,
        "proposed_text": proposed_text,
        "rationale": rationale,
        "default_decision": default_decision,  # accept | reject | clarify
    }
    base.update(extra)
    return base


def run(state: SessionState) -> Dict[str, Any]:
    summary = state.get("session_summary")
    if summary is None:
        return {"phase": "negotiation", "negotiation_items": []}

    memory_bundle = state.get("memory_bundle")
    existing_common_ground = list(getattr(memory_bundle, "common_ground", []) or [])
    existing_by_id: dict[str, Any] = {}
    for entry in existing_common_ground:
        cid = getattr(entry, "cg_id", None)
        if isinstance(cid, str) and cid:
            existing_by_id[cid] = entry

    claims = list(state.get("claims", []) or [])
    claim_by_id: dict[str, Any] = {}
    for c in claims:
        cid = getattr(c, "claim_id", None)
        if isinstance(cid, str) and cid:
            claim_by_id[cid] = c

    # Hard gate: negotiation is only for contradiction resolution.
    contradiction_flag = False
    conflict = state.get("conflict_result")
    if conflict is not None and getattr(conflict, "status", None) == ConflictStatus.TRUE_CONTRADICTION:
        contradiction_flag = True
    if any(str(getattr(c, "prior_conflict", "") or "").strip() for c in claims):
        contradiction_flag = True
    try:
        contradiction_flag = contradiction_flag or int(getattr(summary, "contradictions_detected", 0) or 0) > 0
    except Exception:
        pass

    if not contradiction_flag:
        return {"phase": "negotiation", "negotiation_items": []}

    items: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    if conflict is not None and getattr(conflict, "status", None) == ConflictStatus.TRUE_CONTRADICTION:
        conflict_text = str(getattr(conflict, "explanation", "") or "").strip()
        current_claim = str(getattr(conflict, "current_claim", "") or "").strip()
        prior_claim = str(getattr(conflict, "prior_claim", "") or "").strip()
        if conflict_text and (current_claim or prior_claim):
            key = (current_claim, prior_claim)
            seen_pairs.add(key)
            proposed = f"Resolve contradiction: {conflict_text}"
            cg_id = _stable_cg_id(proposed)
            prior_entry = existing_by_id.get(cg_id)
            items.append(
                _item(
                    kind="common_ground",
                    proposed_text=proposed,
                    source="conflict_result",
                    rationale="Persist contradiction resolution.",
                    default_decision="clarify",
                    cg_id=cg_id,
                    version=int(getattr(prior_entry, "version", 0) or 0),
                    proposed_by="agent",
                    pdf_chunk_ref=getattr(prior_entry, "pdf_chunk_ref", None),
                    original_text=getattr(prior_entry, "negotiated_text", None),
                    conflict_explanation=conflict_text,
                    current_claim=current_claim,
                    past_claim=prior_claim,
                )
            )

    for c in claims:
        prior_id = str(getattr(c, "prior_conflict", "") or "").strip()
        if not prior_id:
            continue
        current_claim = str(getattr(c, "claim_text", "") or "").strip()
        prior_claim_obj = claim_by_id.get(prior_id)
        prior_claim = str(getattr(prior_claim_obj, "claim_text", "") or "").strip()
        key = (current_claim, prior_claim)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        explanation = "Current claim conflicts with a past claim."
        proposed = f"Resolve contradiction between claims: {prior_id} and {getattr(c, 'claim_id', 'current')}"
        cg_id = _stable_cg_id(proposed)
        prior_entry = existing_by_id.get(cg_id)
        items.append(
            _item(
                kind="common_ground",
                proposed_text=proposed,
                source="conflict_history",
                rationale="Persist contradiction resolution from session history.",
                default_decision="clarify",
                cg_id=cg_id,
                version=int(getattr(prior_entry, "version", 0) or 0),
                proposed_by="agent",
                pdf_chunk_ref=getattr(prior_entry, "pdf_chunk_ref", None),
                original_text=getattr(prior_entry, "negotiated_text", None),
                conflict_explanation=explanation,
                current_claim=current_claim,
                past_claim=prior_claim,
            )
        )

    if not items:
        # Contradictions were counted but details are unavailable. Ask for one explicit resolution note.
        proposed = "Resolve contradictions identified in this session."
        cg_id = _stable_cg_id(proposed)
        prior_entry = existing_by_id.get(cg_id)
        items.append(
            _item(
                kind="common_ground",
                proposed_text=proposed,
                source="conflict_summary",
                rationale="Persist a contradiction-resolution note when details are missing.",
                default_decision="clarify",
                cg_id=cg_id,
                version=int(getattr(prior_entry, "version", 0) or 0),
                proposed_by="agent",
                pdf_chunk_ref=getattr(prior_entry, "pdf_chunk_ref", None),
                original_text=getattr(prior_entry, "negotiated_text", None),
                conflict_explanation="Contradiction(s) detected in session summary.",
                current_claim="",
                past_claim="",
            )
        )

    return {"phase": "negotiation", "negotiation_items": items}
