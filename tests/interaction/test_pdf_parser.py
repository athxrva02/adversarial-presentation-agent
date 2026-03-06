"""
Tests for interaction/pdf_parser.py

These tests are unit-level and do NOT require:
- Ollama running
- ChromaDB running
- Any network access

They use a synthetic in-memory PDF created with PyMuPDF so the test suite
is fully self-contained.

Run with:
    cd adversarial-presentation-agent
    pytest tests/test_pdf_parser.py -v
"""

from __future__ import annotations

import os
import tempfile
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdf(pages: list[str]) -> str:
    """
    Create a real (minimal) PDF from a list of page text strings.
    Returns the path to a temporary file (caller must delete it).
    """
    fitz = pytest.importorskip("fitz", reason="PyMuPDF required")

    doc = fitz.open()
    for page_text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), page_text, fontsize=11)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc.save(tmp.name)
    doc.close()
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIngestPDF:

    def test_returns_list_of_document_chunks(self):
        pytest.importorskip("fitz", reason="PyMuPDF required")
        from interaction.pdf_parser import ingest_pdf

        pdf_path = _make_pdf(["We argue that climate change requires urgent action."])
        try:
            chunks = ingest_pdf(pdf_path)
            assert isinstance(chunks, list)
            assert len(chunks) > 0
        finally:
            os.unlink(pdf_path)

    def test_chunk_fields_are_populated(self):
        pytest.importorskip("fitz", reason="PyMuPDF required")
        from interaction.pdf_parser import ingest_pdf
        from storage.schemas import DocumentChunk

        pdf_path = _make_pdf(["We claim this is important evidence according to recent studies."])
        try:
            chunks = ingest_pdf(pdf_path)
            for chunk in chunks:
                assert isinstance(chunk, DocumentChunk)
                assert chunk.chunk_id
                assert chunk.text.strip()
                assert chunk.chunk_type in {"claim", "definition", "evidence", "conclusion"}
                assert chunk.position_in_pdf >= 1
                assert chunk.slide_number == 1
        finally:
            os.unlink(pdf_path)

    def test_multipage_pdf_assigns_correct_slide_numbers(self):
        pytest.importorskip("fitz", reason="PyMuPDF required")
        from interaction.pdf_parser import ingest_pdf

        pdf_path = _make_pdf([
            "Page one claims something important about climate.",
            "Page two defines the key terms used in this study.",
            "Page three concludes therefore the hypothesis is confirmed.",
        ])
        try:
            chunks = ingest_pdf(pdf_path)
            slide_numbers = {c.slide_number for c in chunks}
            # Should have at least 2 different slide numbers (3 pages, some may be merged)
            assert len(slide_numbers) >= 2
        finally:
            os.unlink(pdf_path)

    def test_chunk_ids_are_unique(self):
        pytest.importorskip("fitz", reason="PyMuPDF required")
        from interaction.pdf_parser import ingest_pdf

        long_text = " ".join([
            "This is a fairly long sentence that should generate multiple chunks.",
            "We argue that testing is important.",
            "According to studies, coverage matters.",
            "Therefore, we conclude that comprehensive tests are necessary.",
        ] * 10)

        pdf_path = _make_pdf([long_text])
        try:
            chunks = ingest_pdf(pdf_path)
            ids = [c.chunk_id for c in chunks]
            assert len(ids) == len(set(ids)), "chunk_ids must be unique"
        finally:
            os.unlink(pdf_path)

    def test_position_in_pdf_is_monotonically_increasing(self):
        pytest.importorskip("fitz", reason="PyMuPDF required")
        from interaction.pdf_parser import ingest_pdf

        pdf_path = _make_pdf([
            "We argue that argument one is valid.",
            "We claim argument two is also valid.",
            "Therefore, argument three follows.",
        ])
        try:
            chunks = ingest_pdf(pdf_path)
            positions = [c.position_in_pdf for c in chunks]
            assert positions == sorted(positions), "positions must be monotonically increasing"
        finally:
            os.unlink(pdf_path)

    def test_chunk_ids_are_deterministic(self):
        """Same PDF content → same chunk IDs on repeated ingestion."""
        pytest.importorskip("fitz", reason="PyMuPDF required")
        from interaction.pdf_parser import ingest_pdf

        text = "We argue that determinism is essential for reproducible pipelines."
        pdf_path = _make_pdf([text])
        try:
            chunks_a = ingest_pdf(pdf_path)
            chunks_b = ingest_pdf(pdf_path)
            ids_a = [c.chunk_id for c in chunks_a]
            ids_b = [c.chunk_id for c in chunks_b]
            assert ids_a == ids_b
        finally:
            os.unlink(pdf_path)

    def test_file_not_found_raises(self):
        from interaction.pdf_parser import ingest_pdf
        with pytest.raises(FileNotFoundError):
            ingest_pdf("/nonexistent/path/file.pdf")

    def test_non_pdf_extension_raises(self):
        from interaction.pdf_parser import ingest_pdf
        with pytest.raises(ValueError, match="Expected a .pdf file"):
            ingest_pdf("/tmp/file.docx")

    def test_blank_page_produces_no_chunks(self):
        """A PDF with only blank pages should return empty list."""
        pytest.importorskip("fitz", reason="PyMuPDF required")
        import fitz

        doc = fitz.open()
        doc.new_page()  # blank page
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc.save(tmp.name)
        doc.close()
        tmp.close()

        from interaction.pdf_parser import ingest_pdf
        try:
            chunks = ingest_pdf(tmp.name)
            assert chunks == []
        finally:
            os.unlink(tmp.name)


class TestChunkClassification:
    """Unit-test the heuristic chunk classifier in isolation."""

    def test_evidence_detected(self):
        from interaction.pdf_parser import _classify_chunk
        assert _classify_chunk("According to studies, the effect is significant.") == "evidence"

    def test_conclusion_detected(self):
        from interaction.pdf_parser import _classify_chunk
        assert _classify_chunk("Therefore, we conclude the treatment is effective.") == "conclusion"

    def test_definition_detected(self):
        from interaction.pdf_parser import _classify_chunk
        assert _classify_chunk("Machine learning is defined as a subset of AI.") == "definition"

    def test_claim_is_default(self):
        from interaction.pdf_parser import _classify_chunk
        assert _classify_chunk("The sky is blue on a sunny day.") == "claim"


class TestWindowSentences:
    """Unit-test the sentence windowing logic."""

    def test_short_text_single_chunk(self):
        from interaction.pdf_parser import _window_sentences
        sentences = ["This is short.", "So is this."]
        chunks = _window_sentences(sentences, max_tokens=100, overlap_tokens=10)
        assert len(chunks) == 1

    def test_long_text_splits_into_multiple_chunks(self):
        from interaction.pdf_parser import _window_sentences
        # ~10 tokens each, max 20 tokens → should split
        sentences = [f"Sentence number {i} is a moderately long piece of text." for i in range(20)]
        chunks = _window_sentences(sentences, max_tokens=20, overlap_tokens=5)
        assert len(chunks) > 1

    def test_overlap_produces_shared_content(self):
        """The tail of chunk N should appear at the start of chunk N+1."""
        from interaction.pdf_parser import _window_sentences
        sentences = [f"Unique sentence {i}." for i in range(30)]
        chunks = _window_sentences(sentences, max_tokens=30, overlap_tokens=10)
        if len(chunks) >= 2:
            # At least one sentence should be shared between consecutive chunks
            words_end = set(chunks[0].split())
            words_start = set(chunks[1].split())
            assert words_end & words_start, "Expected overlapping tokens between chunks"

    def test_empty_input(self):
        from interaction.pdf_parser import _window_sentences
        assert _window_sentences([], max_tokens=100, overlap_tokens=10) == []

    def test_single_oversized_sentence_emitted_as_is(self):
        from interaction.pdf_parser import _window_sentences
        very_long = " ".join(["word"] * 300)  # ~300 tokens >> max_tokens
        chunks = _window_sentences([very_long], max_tokens=50, overlap_tokens=10)
        assert len(chunks) == 1
        assert chunks[0] == very_long
