from __future__ import annotations

import hashlib
from typing import Any, Dict
from uuid import uuid4

from reasoning.state import SessionState
from storage.schemas import ConflictStatus


def _uniq_nonempty(values: list[Any], max_items: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        s = str(v or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_items:
            break
    return out


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

    breakdown = state.get("score_breakdown") or {}
    notes = breakdown.get("notes", {}) if isinstance(breakdown, dict) else {}
    open_issues = breakdown.get("open_issues", []) if isinstance(breakdown, dict) else []

    strengths = _uniq_nonempty(list(getattr(summary, "strengths", []) or []), 3)
    weaknesses = _uniq_nonempty(list(getattr(summary, "weaknesses", []) or []), 5)
    next_step = str(notes.get("most_important_next_step", "")).strip() if isinstance(notes, dict) else ""
    open_issues = _uniq_nonempty(list(open_issues or []), 5)
    memory_bundle = state.get("memory_bundle")
    existing_common_ground = list(getattr(memory_bundle, "common_ground", []) or [])
    existing_by_id: dict[str, Any] = {}
    for entry in existing_common_ground:
        cid = getattr(entry, "cg_id", None)
        if isinstance(cid, str) and cid:
            existing_by_id[cid] = entry

    items: list[dict[str, Any]] = []

    for s in strengths:
        items.append(
            _item(
                kind="semantic_strength",
                proposed_text=s,
                source="session_summary.strengths",
                rationale="Promote recurring strength signal.",
                default_decision="accept",
            )
        )

    for w in weaknesses:
        items.append(
            _item(
                kind="semantic_weakness",
                proposed_text=w,
                source="session_summary.weaknesses",
                rationale="Track recurring weakness across sessions.",
                default_decision="accept",
            )
        )

    if next_step:
        cg_id = _stable_cg_id(next_step)
        prior_entry = existing_by_id.get(cg_id)
        items.append(
            _item(
                kind="common_ground",
                proposed_text=next_step,
                source="score_breakdown.notes.most_important_next_step",
                rationale="Carry agreed coaching priority into next session.",
                default_decision="clarify",
                cg_id=cg_id,
                version=int(getattr(prior_entry, "version", 0) or 0),
                proposed_by="agent",
                pdf_chunk_ref=getattr(prior_entry, "pdf_chunk_ref", None),
                original_text=getattr(prior_entry, "negotiated_text", None),
            )
        )

    for issue in open_issues:
        cg_id = _stable_cg_id(issue)
        prior_entry = existing_by_id.get(cg_id)
        items.append(
            _item(
                kind="common_ground",
                proposed_text=issue,
                source="score_breakdown.open_issues",
                rationale="Persist unresolved issue as negotiable common ground.",
                default_decision="clarify",
                cg_id=cg_id,
                version=int(getattr(prior_entry, "version", 0) or 0),
                proposed_by="agent",
                pdf_chunk_ref=getattr(prior_entry, "pdf_chunk_ref", None),
                original_text=getattr(prior_entry, "negotiated_text", None),
            )
        )

    conflict = state.get("conflict_result")
    if conflict is not None and getattr(conflict, "status", None) == ConflictStatus.TRUE_CONTRADICTION:
        conflict_text = str(getattr(conflict, "explanation", "") or "").strip()
        if conflict_text:
            proposed = f"Resolve contradiction: {conflict_text}"
            current_claim = str(getattr(conflict, "current_claim", "") or "").strip()
            prior_claim = str(getattr(conflict, "prior_claim", "") or "").strip()
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

    return {
        "phase": "negotiation",
        "negotiation_items": items,
    }
