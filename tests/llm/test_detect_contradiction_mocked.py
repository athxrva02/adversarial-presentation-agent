from datetime import datetime
from unittest.mock import patch

from reasoning.nodes.detect_contradiction import run
from storage.schemas import (
    ClaimAlignment,
    ClaimRecord,
    ConflictAction,
    ConflictStatus,
    MemoryBundle,
)


def _claim(claim_id: str, text: str, turn: int = 1) -> ClaimRecord:
    return ClaimRecord(
        claim_id=claim_id,
        session_id="s1",
        turn_number=turn,
        claim_text=text,
        alignment=ClaimAlignment.NOVEL,
        mapped_to_slide=None,
        prior_conflict=None,
        timestamp=datetime(2026, 1, 1),
    )


def test_detect_contradiction_empty_input_defaults_to_no_conflict():
    out = run({"user_input": "", "memory_bundle": None, "classification": None})
    res = out["conflict_result"]
    assert res.status == ConflictStatus.NO_CONFLICT
    assert res.action == ConflictAction.IGNORE
    assert out["conflict_prior_claim_id"] is None


def test_detect_contradiction_no_prior_claims_defaults_to_no_conflict():
    bundle = MemoryBundle(
        document_context=[],
        episodic_claims=[],
        episodic_sessions=[],
        semantic_patterns=[],
        common_ground=[],
    )
    out = run({"user_input": "Our method is faster.", "memory_bundle": bundle, "classification": None})
    res = out["conflict_result"]
    assert res.status == ConflictStatus.NO_CONFLICT
    assert res.action == ConflictAction.IGNORE
    assert out["conflict_prior_claim_id"] is None


def test_detect_contradiction_maps_prior_claim_id_and_text():
    c1 = _claim("c1", "Accuracy improved by 10%.", 1)
    c2 = _claim("c2", "Accuracy decreased by 5%.", 2)
    bundle = MemoryBundle(
        document_context=[],
        episodic_claims=[c1, c2],
        episodic_sessions=[],
        semantic_patterns=[],
        common_ground=[],
    )

    fake_llm = {
        "status": "true_contradiction",
        "action": "update",
        "current_claim": "Accuracy improved by 20%.",
        "prior_claim": "placeholder",
        "prior_claim_id": "c2",
        "explanation": "Current claim conflicts with earlier statement.",
    }

    with patch("reasoning.nodes.detect_contradiction.call_llm_structured", return_value=fake_llm):
        out = run({"user_input": "Accuracy improved by 20%.", "memory_bundle": bundle, "classification": None})

    res = out["conflict_result"]
    assert res.status == ConflictStatus.TRUE_CONTRADICTION
    assert res.action == ConflictAction.UPDATE
    assert res.prior_claim == c2.claim_text
    assert out["conflict_prior_claim_id"] == "c2"


def test_detect_contradiction_llm_failure_falls_back_to_no_conflict():
    c1 = _claim("c1", "Claim one.")
    bundle = MemoryBundle(
        document_context=[],
        episodic_claims=[c1],
        episodic_sessions=[],
        semantic_patterns=[],
        common_ground=[],
    )

    with patch("reasoning.nodes.detect_contradiction.call_llm_structured", side_effect=RuntimeError("boom")):
        out = run({"user_input": "Claim two.", "memory_bundle": bundle, "classification": None})

    res = out["conflict_result"]
    assert res.status == ConflictStatus.NO_CONFLICT
    assert res.action == ConflictAction.IGNORE
    assert out["conflict_prior_claim_id"] is None
