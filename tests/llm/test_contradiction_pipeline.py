"""
End-to-end tests for the contradiction detection pipeline.

Verifies the full chain:
  claim storage → vector retrieval → dedup/ranking → detect_contradiction → routing → mediation

These tests are mocked (no Ollama needed) and exercise the real storage layer
with temp directories to catch the exact class of bugs where dict-vs-object
mismatches silently drop claims.

Run:
    pytest tests/llm/test_contradiction_pipeline.py -v
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from unittest.mock import patch

import pytest

from storage.schemas import (
    ClaimAlignment,
    ClaimRecord,
    Classification,
    ConflictAction,
    ConflictStatus,
    MemoryBundle,
    ResponseClass,
    SessionRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stores(tmp_path):
    """Return fresh (VectorStore, RelationalStore) backed by temp directories."""
    pytest.importorskip("chromadb", reason="chromadb required")
    from storage.vector_store import VectorStore
    from storage.relational_store import RelationalStore

    db_path = str(tmp_path / "db" / "agent.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    vs = VectorStore(
        chroma_path=str(tmp_path / "chroma"),
        embedding_model="all-MiniLM-L6-v2",
    )
    rs = RelationalStore(db_path=db_path)
    return vs, rs


def _make_runner(tmp_path, session_id="test-contra"):
    """Create a SessionRunner with real memory stores."""
    pytest.importorskip("chromadb", reason="chromadb required")
    from memory.module import MemoryModule
    from reasoning.graph import SessionRunner

    vs, rs = _make_stores(tmp_path)
    mm = MemoryModule(vector_store=vs, relational_store=rs)
    return SessionRunner(session_id=session_id, memory_module=mm), mm, rs


FAKE_CLASSIFICATION_WEAK = {
    "response_class": "weak",
    "alignment": "unsupported",
    "confidence": 0.6,
    "reasoning": "Vague claim.",
}

FAKE_CLASSIFICATION_CONTRADICTION = {
    "response_class": "contradiction",
    "alignment": "contradicted",
    "confidence": 0.8,
    "reasoning": "Directly contradicts previous claim.",
}

FAKE_QUESTION = "Can you provide more details on that?"

FAKE_SUMMARY = {
    "strengths": ["Stated the topic."],
    "weaknesses": ["Contradicted self."],
    "key_claims": ["Accuracy claim."],
    "open_issues": [],
    "contradictions_detected": 1,
    "overall_notes": "User contradicted earlier statements.",
}

FAKE_SCORE = {
    "overall_score": 40,
    "rubric": {
        "clarity_structure": 50,
        "evidence_specificity": 30,
        "definition_precision": 40,
        "logical_coherence": 30,
        "handling_adversarial_questions": 50,
    },
    "notes": {
        "top_strengths": [],
        "top_weaknesses": ["Self-contradictory claims."],
        "most_important_next_step": "Be consistent.",
    },
}


# ---------------------------------------------------------------------------
# Unit tests: retrieval layer preserves claims through dedup
# ---------------------------------------------------------------------------

class TestRetrievalDedup:
    """Verify _dedup_by_id works with both dicts and Pydantic models."""

    def test_dedup_with_dicts(self):
        from memory.retrieval import _dedup_by_id

        items = [
            {"claim_id": "c1", "text": "A"},
            {"claim_id": "c2", "text": "B"},
            {"claim_id": "c1", "text": "dupe"},
        ]
        result = _dedup_by_id(items, "claim_id")
        assert len(result) == 2
        assert result[0]["claim_id"] == "c1"
        assert result[1]["claim_id"] == "c2"

    def test_dedup_with_pydantic_models(self):
        from memory.retrieval import _dedup_by_id

        c1 = ClaimRecord(
            claim_id="c1", session_id="s1", turn_number=1,
            claim_text="A", alignment=ClaimAlignment.NOVEL,
            mapped_to_slide=None, timestamp=datetime.now(),
        )
        c2 = ClaimRecord(
            claim_id="c2", session_id="s1", turn_number=2,
            claim_text="B", alignment=ClaimAlignment.NOVEL,
            mapped_to_slide=None, timestamp=datetime.now(),
        )
        result = _dedup_by_id([c1, c2, c1], "claim_id")
        assert len(result) == 2

    def test_recency_score_with_dicts(self):
        from memory.retrieval import _recency_score

        item = {"session_id": "s1", "claim_id": "c1"}
        order = {"s1": 0}
        score = _recency_score(item, order)
        assert score == 1.0  # age 0 → decay^0 = 1

    def test_recency_score_with_pydantic_models(self):
        from memory.retrieval import _recency_score

        c = ClaimRecord(
            claim_id="c1", session_id="s1", turn_number=1,
            claim_text="A", alignment=ClaimAlignment.NOVEL,
            mapped_to_slide=None, timestamp=datetime.now(),
        )
        order = {"s1": 2}
        score = _recency_score(c, order)
        assert 0 < score < 1  # age 2 → decayed


# ---------------------------------------------------------------------------
# Integration: episodic store → retrieve returns ClaimRecord objects
# ---------------------------------------------------------------------------

class TestEpisodicClaimRoundTrip:
    """Claims stored in episodic memory must be retrievable as ClaimRecord objects."""

    def test_store_and_retrieve_returns_claim_records(self, tmp_path):
        pytest.importorskip("chromadb", reason="chromadb required")
        from memory.episodic import EpisodicMemory

        vs, rs = _make_stores(tmp_path)

        # FK constraint: need a session record
        rs.insert_session(SessionRecord(
            session_id="s1", timestamp=datetime.now(),
            duration_seconds=0, overall_score=None,
            strengths=[], weaknesses=[], claims_count=0,
            contradictions_detected=0,
        ))

        em = EpisodicMemory(vector_store=vs, relational_store=rs)

        claim = ClaimRecord(
            claim_id="s1-1", session_id="s1", turn_number=1,
            claim_text="Replication improves availability",
            alignment=ClaimAlignment.NOVEL,
            mapped_to_slide=None, timestamp=datetime.now(),
        )
        em.store_claim(claim)

        results = em.retrieve_claims("availability of replicated data", top_k=5)
        assert len(results) >= 1, "Should retrieve the stored claim"
        assert isinstance(results[0], ClaimRecord), (
            f"Expected ClaimRecord, got {type(results[0])}"
        )
        assert results[0].claim_id == "s1-1"
        assert results[0].claim_text == "Replication improves availability"

    def test_retrieved_claims_survive_merge_and_rank(self, tmp_path):
        """Claims from episodic retrieval must not be dropped by merge_and_rank."""
        pytest.importorskip("chromadb", reason="chromadb required")
        from memory.episodic import EpisodicMemory
        from memory.retrieval import merge_and_rank

        vs, rs = _make_stores(tmp_path)
        rs.insert_session(SessionRecord(
            session_id="s1", timestamp=datetime.now(),
            duration_seconds=0, overall_score=None,
            strengths=[], weaknesses=[], claims_count=0,
            contradictions_detected=0,
        ))

        em = EpisodicMemory(vector_store=vs, relational_store=rs)
        em.store_claim(ClaimRecord(
            claim_id="s1-1", session_id="s1", turn_number=1,
            claim_text="Our model achieves 95% accuracy",
            alignment=ClaimAlignment.NOVEL,
            mapped_to_slide=None, timestamp=datetime.now(),
        ))

        claims = em.retrieve_claims("accuracy of the model", top_k=5)
        assert len(claims) >= 1

        bundle = merge_and_rank(
            document_chunks=[], episodic_claims=claims,
            episodic_sessions=[], semantic_patterns=[],
            common_ground=[], top_k=5,
            session_order={"s1": 0},
        )
        assert len(bundle.episodic_claims) >= 1, (
            "merge_and_rank must not drop claims"
        )
        assert bundle.episodic_claims[0].claim_id == "s1-1"


# ---------------------------------------------------------------------------
# Integration: detect_contradiction node receives prior claims
# ---------------------------------------------------------------------------

class TestDetectContradictionWithRealMemory:
    """
    The detect_contradiction node must receive prior claims from memory
    and call the LLM with them.
    """

    def test_prior_claims_reach_contradiction_detector(self, tmp_path):
        """After storing a claim, the next turn's retrieve+detect pipeline sees it."""
        pytest.importorskip("chromadb", reason="chromadb required")
        from memory.module import MemoryModule

        vs, rs = _make_stores(tmp_path)
        mm = MemoryModule(vector_store=vs, relational_store=rs)

        # Simulate session record for FK
        rs.insert_session(SessionRecord(
            session_id="s1", timestamp=datetime.now(),
            duration_seconds=0, overall_score=None,
            strengths=[], weaknesses=[], claims_count=0,
            contradictions_detected=0,
        ))

        # Store a claim (simulating what happens after turn 1)
        mm.store_claim(ClaimRecord(
            claim_id="s1-1", session_id="s1", turn_number=1,
            claim_text="Disconnecting nodes affects partition tolerance",
            alignment=ClaimAlignment.NOVEL,
            mapped_to_slide=None, timestamp=datetime.now(),
        ))

        # Retrieve for turn 2 (contradictory statement)
        bundle = mm.retrieve(
            query="Disconnecting nodes does not affect partition tolerance",
            stores=["document", "episodic", "semantic", "common_ground"],
        )
        assert len(bundle.episodic_claims) >= 1, (
            "MemoryModule.retrieve must return stored claims"
        )

        # Now run detect_contradiction with this bundle
        from reasoning.nodes.detect_contradiction import run as detect_run

        fake_llm_response = {
            "status": "true_contradiction",
            "action": "clarify",
            "current_claim": "Disconnecting nodes does not affect partition tolerance",
            "prior_claim": "Disconnecting nodes affects partition tolerance",
            "prior_claim_id": "s1-1",
            "explanation": "Direct negation of prior claim.",
        }

        state = {
            "user_input": "Disconnecting nodes does not affect partition tolerance",
            "memory_bundle": bundle,
            "classification": None,
        }

        with patch(
            "reasoning.nodes.detect_contradiction.call_llm_structured",
            return_value=fake_llm_response,
        ):
            out = detect_run(state)

        result = out["conflict_result"]
        assert result.status == ConflictStatus.TRUE_CONTRADICTION
        assert result.action == ConflictAction.CLARIFY
        assert out["conflict_prior_claim_id"] == "s1-1"
        assert result.prior_claim == "Disconnecting nodes affects partition tolerance"


# ---------------------------------------------------------------------------
# Integration: full SessionRunner multi-turn contradiction flow
# ---------------------------------------------------------------------------

class TestSessionRunnerContradictionFlow:
    """
    Full end-to-end: two practice turns through SessionRunner where the second
    turn contradicts the first, using real memory stores.
    """

    def test_two_turn_contradiction_detected(self, tmp_path):
        """
        Turn 1: 'Our model achieves 95% accuracy'
        Turn 2: 'Our model only reaches 78% accuracy'
        -> detect_contradiction should receive turn 1's claim and flag it.

        We mock the LLM calls inside each node (not the node functions
        themselves) so the real retrieve + detect_contradiction logic runs.
        """
        runner, mm, rs = _make_runner(tmp_path, session_id="contra-test")

        # --- Turn 1: establish a claim ---
        with patch(
            "reasoning.nodes.classify.call_llm_structured",
            return_value=FAKE_CLASSIFICATION_WEAK,
        ), patch(
            "reasoning.nodes.detect_contradiction.call_llm_structured",
            return_value={
                "status": "no_conflict",
                "action": "ignore",
                "current_claim": "Our model achieves 95% accuracy",
                "prior_claim": None,
                "prior_claim_id": None,
                "explanation": "No prior claims.",
            },
        ), patch(
            "reasoning.nodes.generate_question.call_llm_text",
            return_value=FAKE_QUESTION,
        ):
            q1 = runner.handle_user_input("Our model achieves 95% accuracy")

        assert q1, "Turn 1 should return a question"
        assert len(runner.state["claims"]) == 1
        assert runner.state["claims"][0].claim_text == "Our model achieves 95% accuracy"

        # Verify claim was stored in memory
        stored = rs.get_claims_for_session("contra-test")
        assert len(stored) >= 1, "Turn 1 claim must be persisted in SQLite"

        # --- Turn 2: contradict turn 1 ---
        # Mock the LLM to return TRUE_CONTRADICTION (detect_contradiction node
        # itself runs for real, so it will exercise the retrieve → dedup → bundle path)
        with patch(
            "reasoning.nodes.classify.call_llm_structured",
            return_value=FAKE_CLASSIFICATION_CONTRADICTION,
        ), patch(
            "reasoning.nodes.detect_contradiction.call_llm_structured",
            return_value={
                "status": "true_contradiction",
                "action": "clarify",
                "current_claim": "Our model only reaches 78% accuracy",
                "prior_claim": "Our model achieves 95% accuracy",
                "prior_claim_id": "contra-test-1",
                "explanation": "95% vs 78% is a direct contradiction.",
            },
        ), patch(
            "reasoning.nodes.mediate_contradiction.call_llm_text",
            return_value="You said 95% before but now 78% — which is correct?",
        ):
            q2 = runner.handle_user_input("Our model only reaches 78% accuracy")

        # The detect_contradiction node ran for real with the memory bundle.
        # If prior claims were dropped (the original bug), it would have
        # short-circuited to NO_CONFLICT and never called call_llm_structured,
        # so the router would have gone to generate_question instead of
        # mediate_contradiction — giving us FAKE_QUESTION, not the mediation text.
        conflict = runner.state.get("conflict_result")
        assert conflict is not None, "conflict_result must be set"
        assert conflict.status == ConflictStatus.TRUE_CONTRADICTION, (
            f"Expected TRUE_CONTRADICTION, got {conflict.status} — "
            "prior claims may not have reached detect_contradiction"
        )
        assert runner.state.get("conflict_prior_claim_id") == "contra-test-1"

        # Verify the agent produced the mediation response (not blank, not FAKE_QUESTION)
        assert q2, "Turn 2 must produce a non-empty mediation question"
        assert q2 != FAKE_QUESTION, (
            "Response should be from mediate_contradiction, not generate_question — "
            "routing may have failed"
        )

    def test_contradiction_patches_prior_conflict_on_claim(self, tmp_path):
        """
        When a contradiction is detected, the current turn's ClaimRecord
        should have prior_conflict set to the conflicting claim's ID.
        """
        runner, mm, rs = _make_runner(tmp_path, session_id="patch-test")

        # Turn 1
        with patch(
            "reasoning.nodes.classify.call_llm_structured",
            return_value=FAKE_CLASSIFICATION_WEAK,
        ), patch(
            "reasoning.nodes.detect_contradiction.call_llm_structured",
            return_value={
                "status": "no_conflict",
                "action": "ignore",
                "current_claim": "Replication improves availability",
                "prior_claim": None,
                "prior_claim_id": None,
                "explanation": "First turn.",
            },
        ), patch(
            "reasoning.nodes.generate_question.call_llm_text",
            return_value=FAKE_QUESTION,
        ):
            runner.handle_user_input("Replication improves availability")

        # Turn 2: contradiction detected
        with patch(
            "reasoning.nodes.classify.call_llm_structured",
            return_value=FAKE_CLASSIFICATION_CONTRADICTION,
        ), patch(
            "reasoning.nodes.detect_contradiction.call_llm_structured",
            return_value={
                "status": "true_contradiction",
                "action": "clarify",
                "current_claim": "Replication does not improve availability",
                "prior_claim": "Replication improves availability",
                "prior_claim_id": "patch-test-1",
                "explanation": "Direct negation.",
            },
        ), patch(
            "reasoning.nodes.mediate_contradiction.call_llm_text",
            return_value="You previously said replication improves availability — which is it?",
        ):
            q2 = runner.handle_user_input("Replication does not improve availability")

        assert q2, "Should produce a mediation question"

        # Verify the contradiction was detected (not short-circuited)
        conflict = runner.state.get("conflict_result")
        assert conflict is not None
        assert conflict.status == ConflictStatus.TRUE_CONTRADICTION

        # Check that the turn 2 claim has prior_conflict patched
        turn2_claims = [c for c in runner.state["claims"] if c.turn_number == 2]
        assert len(turn2_claims) == 1
        assert turn2_claims[0].prior_conflict == "patch-test-1", (
            f"Expected prior_conflict='patch-test-1', got {turn2_claims[0].prior_conflict!r}"
        )


# ---------------------------------------------------------------------------
# Routing: mediate_contradiction handles classification-only contradictions
# ---------------------------------------------------------------------------

class TestMediationNodeHandlesClassificationRoute:
    """
    When classification says 'contradiction' but conflict_result is not
    TRUE_CONTRADICTION, the mediation node must still produce output.
    """

    def test_mediation_from_classification_not_blank(self):
        from reasoning.nodes.mediate_contradiction import run as mediate_run
        from storage.schemas import ConflictResult

        # conflict_result says NO_CONFLICT, but classification says CONTRADICTION
        state = {
            "user_input": "Replication cannot improve availability",
            "conflict_result": ConflictResult(
                status=ConflictStatus.NO_CONFLICT,
                action=ConflictAction.IGNORE,
                current_claim="Replication cannot improve availability",
                prior_claim=None,
                explanation="No prior claims.",
            ),
            "classification": Classification(
                response_class=ResponseClass.CONTRADICTION,
                alignment=ClaimAlignment.CONTRADICTED,
                confidence=0.8,
                reasoning="User contradicts their presentation.",
            ),
        }

        with patch(
            "reasoning.nodes.mediate_contradiction.call_llm_text",
            return_value="How does this align with your earlier claim about availability?",
        ):
            out = mediate_run(state)

        assert out["agent_response"], (
            "Mediation must produce output when classification says contradiction, "
            f"got: {out['agent_response']!r}"
        )

    def test_mediation_from_true_contradiction(self):
        from reasoning.nodes.mediate_contradiction import run as mediate_run
        from storage.schemas import ConflictResult

        state = {
            "user_input": "X is false",
            "conflict_result": ConflictResult(
                status=ConflictStatus.TRUE_CONTRADICTION,
                action=ConflictAction.CLARIFY,
                current_claim="X is false",
                prior_claim="X is true",
                explanation="Direct negation.",
            ),
            "classification": None,
        }

        with patch(
            "reasoning.nodes.mediate_contradiction.call_llm_text",
            return_value="You said X is true before — which is correct?",
        ):
            out = mediate_run(state)

        assert out["agent_response"], "Must produce output for TRUE_CONTRADICTION"

    def test_mediation_fallback_when_llm_fails(self):
        from reasoning.nodes.mediate_contradiction import run as mediate_run
        from storage.schemas import ConflictResult

        state = {
            "user_input": "Y is false",
            "conflict_result": ConflictResult(
                status=ConflictStatus.TRUE_CONTRADICTION,
                action=ConflictAction.CLARIFY,
                current_claim="Y is false",
                prior_claim="Y is true",
                explanation="Direct negation.",
            ),
            "classification": None,
        }

        with patch(
            "reasoning.nodes.mediate_contradiction.call_llm_text",
            side_effect=RuntimeError("LLM down"),
        ):
            out = mediate_run(state)

        resp = out["agent_response"]
        assert resp, "Fallback must produce a non-empty question"
        assert "Y is true" in resp, "Fallback should reference the prior claim"
        assert "Y is false" in resp, "Fallback should reference the current claim"

    def test_no_contradiction_signal_returns_empty(self):
        from reasoning.nodes.mediate_contradiction import run as mediate_run
        from storage.schemas import ConflictResult

        state = {
            "user_input": "Some normal response",
            "conflict_result": ConflictResult(
                status=ConflictStatus.NO_CONFLICT,
                action=ConflictAction.IGNORE,
                current_claim="Some normal response",
                prior_claim=None,
                explanation="No conflict.",
            ),
            "classification": Classification(
                response_class=ResponseClass.WEAK,
                alignment=ClaimAlignment.UNSUPPORTED,
                confidence=0.5,
                reasoning="Vague.",
            ),
        }

        out = mediate_run(state)
        assert out["agent_response"] == "", (
            "No contradiction signal → should return empty"
        )
