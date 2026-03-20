"""
export.py — Export session results to CSV after each session ends.

Creates: results/<YYYY-MM-DD_HH-MM-SS>/
    summary.csv  — session metadata, scores, strengths, weaknesses
    turns.csv    — Q&A turn-by-turn log
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any


def export_session(
    state: dict[str, Any],
    pdf_path: str,
    results_root: str = "results",
) -> str:
    """
    Write session data to CSV files under results/<datetime>/.

    Args:
        state:        The runner.state dict after end_session() completes.
        pdf_path:     Path to the PDF used for this session.
        results_root: Parent directory for all results (created if missing).

    Returns:
        The path to the run-specific results directory.
    """
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(results_root, run_ts)
    os.makedirs(run_dir, exist_ok=True)

    _write_summary(state, pdf_path, run_dir)
    _write_turns(state, run_dir, session_id=state.get("session_id", ""))

    return run_dir


# ── Internal helpers ───────────────────────────────────────────────────────────

def _write_summary(state: dict[str, Any], pdf_path: str, run_dir: str) -> None:
    rec = state.get("session_summary")
    breakdown = state.get("score_breakdown") or {}
    negotiation_items = state.get("negotiation_items") or []
    negotiation_decisions = state.get("negotiation_decisions") or []

    rubric: dict[str, Any] = breakdown.get("rubric_scores") or {}
    notes: dict[str, Any] = breakdown.get("notes") or {}

    # Count negotiation outcomes
    accepted = sum(
        1 for d in negotiation_decisions if d.get("decision") in {"accept", "update"}
    )
    rejected = sum(
        1 for d in negotiation_decisions if d.get("decision") == "reject"
    )

    row: dict[str, Any] = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": rec.session_id if rec else state.get("session_id", ""),
        "memory_type": state.get("memory_mode", ""),
        "pdf_file": os.path.basename(pdf_path),
        "session_timestamp": rec.timestamp.strftime("%Y-%m-%d %H:%M:%S") if rec and rec.timestamp else "",
        "duration_seconds": rec.duration_seconds if rec else "",
        "overall_score": rec.overall_score if rec else "",
        "claims_count": rec.claims_count if rec else "",
        "contradictions_detected": rec.contradictions_detected if rec else "",
        "contradictions_accepted": accepted,
        "contradictions_rejected": rejected,
        "strengths": " | ".join(rec.strengths) if rec and rec.strengths else "",
        "weaknesses": " | ".join(rec.weaknesses) if rec and rec.weaknesses else "",
        "top_priority": notes.get("most_important_next_step", "") if isinstance(notes, dict) else "",
    }

    # Flatten rubric dimensions
    for dim, val in rubric.items():
        safe_key = f"rubric_{dim.lower().replace(' ', '_')}"
        row[safe_key] = val

    _write_csv(os.path.join(run_dir, "summary.csv"), [row])


def _write_turns(state: dict[str, Any], run_dir: str, session_id: str = "") -> None:
    turns: list[dict[str, Any]] = state.get("turns") or []
    claims: list[Any] = state.get("claims") or []

    # Build a lookup from turn_number → claim record
    claim_by_turn: dict[int, Any] = {}
    for claim in claims:
        if hasattr(claim, "turn_number"):
            claim_by_turn[claim.turn_number] = claim

    rows = []
    for i, t in enumerate(turns):
        turn_num = t.get("turn_number") or (i + 1)
        claim = claim_by_turn.get(turn_num)

        row = {
            "session_id": session_id,
            "turn_number": turn_num,
            "role": t.get("role", ""),
            "content": t.get("content", ""),
            "response_class": t.get("response_class", ""),
            "alignment": t.get("alignment", ""),
            "confidence": t.get("confidence", ""),
            "claim_text": claim.claim_text if claim else "",
            "claim_alignment": claim.alignment.value if claim and hasattr(claim.alignment, "value") else (claim.alignment if claim else ""),
            "mapped_to_slide": claim.mapped_to_slide if claim else "",
            "prior_conflict": claim.prior_conflict if claim else "",
        }
        rows.append(row)

    if not rows:
        # Still write empty file with headers so the folder is self-documenting
        rows = [{}]

    _write_csv(os.path.join(run_dir, "turns.csv"), rows)


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    # Collect all fieldnames preserving insertion order across all rows
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
