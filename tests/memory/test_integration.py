"""
Integration test: input → question → memory write → retrieval.

A single runnable proof that all four layers
talk to each other correctly.

What is tested end-to-end:
    PDF parser  →  DocumentMemory.store  →  DocumentMemory.retrieve
    SessionRunner.handle_user_input  (mocked LLM calls)
    SessionRunner.end_session        (mocked LLM calls)
    RelationalStore: claim + session persistence
    VectorStore: chunk embedding + similarity query

Design decisions:
- LLM calls (classify, generate_question, summarise, score) are mocked so the
  test runs without Ollama and completes in < 5 s.
- ChromaDB and SQLite use temporary directories, cleaned up automatically.
- PyMuPDF is used to build a real in-memory PDF; the test is skipped if it is
  not installed (pytest.importorskip).

Run:
    pytest tests/memory/test_integration.py -v
    pytest tests/memory/test_integration.py -v -k "not slow"
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdf(tmp_path, pages: list[str]) -> str:
    """Create a real multi-page PDF using PyMuPDF. Skip test if not installed."""
    fitz = pytest.importorskip("fitz", reason="PyMuPDF required for PDF integration tests")
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    pdf_path = str(tmp_path / "presentation.pdf")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


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


# ---------------------------------------------------------------------------
# Mocked LLM responses
# ---------------------------------------------------------------------------

FAKE_CLASSIFICATION = {
    "response_class": "weak",
    "alignment": "unsupported",
    "confidence": 0.6,
    "reasoning": "Claim lacks supporting evidence from the provided document.",
}

FAKE_QUESTION = "What specific metric did you use to measure the improvement?"

FAKE_SUMMARY = {
    "strengths": ["Identified the research question clearly."],
    "weaknesses": ["Missing quantitative evidence for main claim."],
    "key_claims": ["The proposed method reduces error rates."],
    "open_issues": ["No baseline comparison provided."],
    "contradictions_detected": 0,
    "overall_notes": "A reasonable first attempt; needs more concrete evidence.",
}

FAKE_SCORE = {
    "overall_score": 62,
    "rubric": {
        "clarity_structure": 65,
        "evidence_specificity": 55,
        "definition_precision": 60,
        "logical_coherence": 70,
        "handling_adversarial_questions": 60,
    },
    "notes": {
        "top_strengths": ["Clear research question."],
        "top_weaknesses": ["No quantitative baseline."],
        "most_important_next_step": "State baseline and evaluation metric upfront.",
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPDFToSessionIntegration:
    """
    End-to-end: PDF → chunks → store → SessionRunner → memory write → retrieval.
    """

    def test_pdf_ingest_then_practice_turn(self, tmp_path):
        """
        Full pipeline:
        1. Parse a PDF into DocumentChunks.
        2. Store chunks in DocumentMemory (ChromaDB + SQLite).
        3. Run one practice turn through SessionRunner (LLM mocked).
        4. Assert the agent produced a question and a ClaimRecord was stored.
        """
        pytest.importorskip("fitz", reason="PyMuPDF required")
        pytest.importorskip("chromadb", reason="chromadb required")

        from interaction.pdf_parser import ingest_pdf
        from memory.document import DocumentMemory
        from reasoning.graph import SessionRunner

        vs, rs = _make_stores(tmp_path)
        doc_mem = DocumentMemory(vector_store=vs, relational_store=rs)

        # Step 1 — parse PDF
        pdf_path = _make_pdf(tmp_path, [
            "We argue that transformer models outperform RNNs on sequence tasks.",
            "According to benchmarks, BLEU scores improved by 4.2 points.",
            "Therefore, we conclude that attention mechanisms are the key driver.",
        ])
        chunks = ingest_pdf(pdf_path)
        assert len(chunks) > 0, "Parser must produce chunks"

        # Step 2 — store
        doc_mem.store(chunks)
        assert doc_mem.count() > 0

        # Step 3 — practice turn (LLM mocked)
        runner = SessionRunner(session_id="integ-test-01")

        with patch("reasoning.nodes.classify.call_llm_structured", return_value=FAKE_CLASSIFICATION), \
             patch("reasoning.nodes.generate_question.call_llm_text", return_value=FAKE_QUESTION):
            question = runner.handle_user_input(
                "Our model improves translation quality compared to the baseline."
            )

        # Step 4 — assertions
        assert question, "SessionRunner must return a non-empty question"
        assert "?" in question, f"Response should be a question, got: {question!r}"
        assert runner.state["turn_number"] == 1
        assert len(runner.state["claims"]) == 1
        claim = runner.state["claims"][0]
        assert claim.session_id == "integ-test-01"
        assert claim.turn_number == 1

    def test_multi_turn_then_end_session(self, tmp_path):
        """
        Three practice turns → end_session → SessionRecord persisted with score.
        """
        pytest.importorskip("fitz", reason="PyMuPDF required")
        pytest.importorskip("chromadb", reason="chromadb required")

        from interaction.pdf_parser import ingest_pdf
        from memory.document import DocumentMemory
        from storage.relational_store import RelationalStore
        from reasoning.graph import SessionRunner

        vs, rs = _make_stores(tmp_path)
        doc_mem = DocumentMemory(vector_store=vs, relational_store=rs)

        pdf_path = _make_pdf(tmp_path, [
            "We claim that our pruning method reduces model size by 40%.",
            "Evidence from three datasets supports the compression ratio claim.",
            "Thus, we conclude that structured pruning is preferable to random dropout.",
        ])
        doc_mem.store(ingest_pdf(pdf_path))

        runner = SessionRunner(session_id="integ-test-02")

        with patch("reasoning.nodes.classify.call_llm_structured", return_value=FAKE_CLASSIFICATION), \
             patch("reasoning.nodes.generate_question.call_llm_text", return_value=FAKE_QUESTION):
            for i, turn_text in enumerate([
                "The model is smaller after pruning.",
                "We measured on ImageNet and CIFAR-10.",
                "Accuracy drops less than 1% in both cases.",
            ], start=1):
                q = runner.handle_user_input(turn_text)
                assert q, f"Turn {i}: expected a question"

        assert runner.state["turn_number"] == 3
        assert len(runner.state["claims"]) == 3

        # End session
        with patch("reasoning.nodes.summarise.call_llm_structured", return_value=FAKE_SUMMARY), \
             patch("reasoning.nodes.score.call_llm_structured", return_value=FAKE_SCORE):
            session_record = runner.end_session()

        assert session_record is not None
        assert session_record.session_id == "integ-test-02"
        assert session_record.overall_score == 62.0
        assert session_record.claims_count == 3
        assert "Missing quantitative evidence" in session_record.weaknesses[0]

        # Persist to relational store and verify round-trip
        rs.insert_session(session_record)
        row = rs.get_session("integ-test-02")
        assert row is not None
        assert row["overall_score"] == 62.0
        assert row["claims_count"] == 3

    def test_document_retrieval_returns_relevant_chunks(self, tmp_path):
        """
        After ingesting a PDF, a query semantically related to page content
        should return the correct slide's chunk as the top result.
        """
        pytest.importorskip("fitz", reason="PyMuPDF required")
        pytest.importorskip("chromadb", reason="chromadb required")

        from interaction.pdf_parser import ingest_pdf
        from memory.document import DocumentMemory

        vs, rs = _make_stores(tmp_path)
        doc_mem = DocumentMemory(vector_store=vs, relational_store=rs)

        pdf_path = _make_pdf(tmp_path, [
            "Deep learning models have transformed natural language processing tasks.",
            "The French Revolution began in 1789 with the storming of the Bastille.",
            "Photosynthesis converts sunlight into chemical energy in plant cells.",
        ])
        doc_mem.store(ingest_pdf(pdf_path))

        results = doc_mem.retrieve("neural networks and language models", top_k=1)
        assert results, "Should return at least one result"
        top_text = results[0].text.lower()
        assert "deep learning" in top_text or "natural language" in top_text, (
            f"Expected NLP-related chunk as top result, got: {results[0].text!r}"
        )

    def test_first_session_empty_memory_does_not_crash(self, tmp_path):
        """
        Edge case from technical plan: first session with no document uploaded.
        SessionRunner must still complete a turn gracefully with empty memory_bundle.
        """
        from reasoning.graph import SessionRunner

        runner = SessionRunner(session_id="integ-test-empty")

        with patch("reasoning.nodes.classify.call_llm_structured", return_value=FAKE_CLASSIFICATION), \
             patch("reasoning.nodes.generate_question.call_llm_text", return_value=FAKE_QUESTION):
            question = runner.handle_user_input("My research is about neural compression.")

        assert question, "Should return a question even with no PDF uploaded"
        assert runner.state["memory_bundle"] is None  # no memory injected — that's fine

    def test_pdf_reupload_clears_old_chunks(self, tmp_path):
        """
        Edge case from technical plan: re-uploading a PDF replaces the previous
        document context (clear → re-ingest).
        """
        fitz = pytest.importorskip("fitz", reason="PyMuPDF required")
        pytest.importorskip("chromadb", reason="chromadb required")

        from interaction.pdf_parser import ingest_pdf
        from memory.document import DocumentMemory

        vs, rs = _make_stores(tmp_path)
        doc_mem = DocumentMemory(vector_store=vs, relational_store=rs)

        def _write_pdf(text: str, path: str) -> None:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), text, fontsize=11)
            doc.save(path)
            doc.close()

        p1 = str(tmp_path / "v1.pdf")
        p2 = str(tmp_path / "v2.pdf")
        _write_pdf("Version one: we argue that X is true.", p1)
        _write_pdf("Version two: we now claim that Y is the key insight.", p2)

        # Upload v1
        doc_mem.store(ingest_pdf(p1))
        assert doc_mem.count() > 0, "v1 chunks must be stored"

        # Re-upload: clear old, store new
        doc_mem.clear()
        assert doc_mem.count() == 0, "clear() must remove all chunks"

        doc_mem.store(ingest_pdf(p2))
        assert doc_mem.count() > 0, "v2 chunks must be stored after re-ingest"

        # v2 content is retrievable
        results = doc_mem.retrieve("key insight", top_k=1)
        assert results, "Query must return results after re-ingest"
        assert "key insight" in results[0].text.lower() or "Y" in results[0].text, (
            f"Expected v2 content, got: {results[0].text!r}"
        )

    def test_claim_records_written_across_turns(self, tmp_path):
        """
        Each turn through SessionRunner appends exactly one ClaimRecord to state.
        After N turns, state['claims'] has exactly N records with sequential turn numbers.
        """
        from reasoning.graph import SessionRunner

        runner = SessionRunner(session_id="integ-test-claims")
        turns = [
            "We propose a new attention mechanism.",
            "It reduces memory usage by 30% compared to standard attention.",
            "We verified this on three NLP benchmarks.",
        ]

        with patch("reasoning.nodes.classify.call_llm_structured", return_value=FAKE_CLASSIFICATION), \
             patch("reasoning.nodes.generate_question.call_llm_text", return_value=FAKE_QUESTION):
            for text in turns:
                runner.handle_user_input(text)

        claims = runner.state["claims"]
        assert len(claims) == 3, f"Expected 3 claims, got {len(claims)}"
        for i, claim in enumerate(claims, start=1):
            assert claim.turn_number == i, f"Claim {i} has wrong turn_number {claim.turn_number}"
            assert claim.session_id == "integ-test-claims"
            assert claim.claim_text == turns[i - 1]
