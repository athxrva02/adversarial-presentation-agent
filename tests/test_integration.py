"""
Member C Integration Tests — Week 3
Full pipeline: audio → STT → text | PDF → chunks → embed → store → query

Tests cover:
1. STT pipeline (mocked Whisper)
2. PDF ingest → chunk → store → query (real ChromaDB + SQLite in tmp dirs)
3. Recency weighting on episodic retrieval
4. Edge cases: empty memory, PDF re-upload, resolved contradictions

Run all:
    cd adversarial-presentation-agent
    pytest tests/test_member_c_integration.py -v

Run only fast (no real DB):
    pytest tests/test_member_c_integration.py -v -m "not slow"
"""

from __future__ import annotations

import os
import struct
import tempfile
import wave
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers: create synthetic audio and PDF files
# ---------------------------------------------------------------------------

def _make_wav(path: str, duration_s: float = 1.0, freq_hz: float = 440.0, sample_rate: int = 16000) -> str:
    """Write a real WAV file with a sine wave."""
    import math
    n_samples = int(sample_rate * duration_s)
    amplitude = 16000
    samples = [
        int(amplitude * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        for i in range(n_samples)
    ]
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        raw = struct.pack(f"{n_samples}h", *samples)
        wf.writeframes(raw)
    return path


def _make_pdf(path: str, pages: list[str]) -> str:
    """Write a real minimal PDF using PyMuPDF."""
    try:
        import fitz  # type: ignore
    except ImportError:
        pytest.skip("PyMuPDF (fitz) is not installed — run: pip install PyMuPDF")
    doc = fitz.open()
    for page_text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), page_text, fontsize=11)
    doc.save(path)
    doc.close()
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_audio(tmp_path):
    wav_path = str(tmp_path / "sample.wav")
    _make_wav(wav_path, duration_s=2.0, freq_hz=440.0)
    return wav_path


@pytest.fixture()
def tmp_pdf(tmp_path):
    pdf_path = str(tmp_path / "sample.pdf")
    pages = [
        "We argue that climate change requires urgent action by governments worldwide. "
        "The evidence shows a 1.2°C rise since pre-industrial levels. "
        "Therefore, carbon taxes should be introduced immediately.",

        "According to the IPCC report, emissions must halve by 2030. "
        "Studies show that renewable energy is now cheaper than fossil fuels. "
        "We conclude that a rapid transition is both feasible and necessary.",
    ]
    return _make_pdf(pdf_path, pages)


@pytest.fixture()
def tmp_stores(tmp_path):
    """Isolated ChromaDB + SQLite for each test."""
    chroma_dir = str(tmp_path / "chroma")
    sqlite_path = str(tmp_path / "db" / "agent.db")
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)

    import storage.vector_store as vs_mod
    import storage.relational_store as rs_mod

    original_vs = vs_mod._default_store
    original_rs = rs_mod._default_store

    vs_mod._default_store = None
    rs_mod._default_store = None

    settings_patch = {
        "chroma_path": chroma_dir,
        "sqlite_path": sqlite_path,
        "embedding_model": "all-MiniLM-L6-v2",
        "max_chunk_tokens": 128,
        "chunk_overlap_tokens": 16,
        "retrieval_top_k": 5,
    }
    with patch.multiple("config.settings", **settings_patch):
        yield {"chroma_path": chroma_dir, "sqlite_path": sqlite_path}

    vs_mod._default_store = original_vs
    rs_mod._default_store = original_rs


# ===========================================================================
# 2. STT Pipeline Tests (mocked Whisper)
# ===========================================================================

class TestSTTPipeline:

    def test_transcribe_returns_string(self, tmp_audio):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  Hello world.  "}
        with patch("interaction.stt._get_model", return_value=mock_model):
            from interaction.stt import transcribe
            result = transcribe(tmp_audio)
        assert result == "Hello world."

    def test_language_hint_forwarded(self, tmp_audio):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Bonjour."}
        with patch("interaction.stt._get_model", return_value=mock_model):
            from interaction import stt
            stt.transcribe(tmp_audio, language="fr")
        call_kwargs = mock_model.transcribe.call_args
        assert "fr" in str(call_kwargs)

    def test_fp16_disabled(self, tmp_audio):
        """fp16=False should always be passed (CPU compatibility)."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "test"}
        with patch("interaction.stt._get_model", return_value=mock_model):
            from interaction import stt
            stt.transcribe(tmp_audio)
        call_kwargs = mock_model.transcribe.call_args
        assert "False" in str(call_kwargs) or False in call_kwargs[1].values() \
            or False in [v for v in (call_kwargs.kwargs or {}).values()]

    def test_empty_audio_returns_empty(self, tmp_audio):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": ""}
        with patch("interaction.stt._get_model", return_value=mock_model):
            from interaction.stt import transcribe
            result = transcribe(tmp_audio)
        assert result == ""


# ===========================================================================
# 3. PDF Ingestion Pipeline Tests
# ===========================================================================

class TestPDFIngestionPipeline:

    def test_ingest_returns_chunks(self, tmp_pdf):
        from interaction.pdf_parser import ingest_pdf
        chunks = ingest_pdf(tmp_pdf)
        assert len(chunks) > 0

    def test_chunks_have_required_fields(self, tmp_pdf):
        from interaction.pdf_parser import ingest_pdf
        chunks = ingest_pdf(tmp_pdf)
        for c in chunks:
            assert c.chunk_id
            assert c.text.strip()
            assert c.chunk_type in {"claim", "definition", "evidence", "conclusion"}
            assert c.position_in_pdf > 0

    def test_chunk_ids_are_unique(self, tmp_pdf):
        from interaction.pdf_parser import ingest_pdf
        chunks = ingest_pdf(tmp_pdf)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "All chunk IDs must be unique"

    def test_slide_numbers_assigned(self, tmp_pdf):
        from interaction.pdf_parser import ingest_pdf
        chunks = ingest_pdf(tmp_pdf)
        slide_numbers = {c.slide_number for c in chunks if c.slide_number is not None}
        assert len(slide_numbers) >= 1

    def test_file_not_found_raises(self):
        from interaction.pdf_parser import ingest_pdf
        with pytest.raises(FileNotFoundError):
            ingest_pdf("/nonexistent/file.pdf")

    def test_non_pdf_extension_raises(self, tmp_path):
        from interaction.pdf_parser import ingest_pdf
        txt = tmp_path / "doc.txt"
        txt.write_text("Hello")
        with pytest.raises(ValueError, match="pdf"):
            ingest_pdf(str(txt))

    def test_chunk_type_evidence_detected(self, tmp_path):
        """'According to' should be classified as evidence."""
        pdf_path = str(tmp_path / "evidence.pdf")
        _make_pdf(pdf_path, ["According to recent studies, the data suggest a clear trend."])
        from interaction.pdf_parser import ingest_pdf
        chunks = ingest_pdf(pdf_path)
        types = [c.chunk_type for c in chunks]
        assert "evidence" in types

    def test_chunk_type_conclusion_detected(self, tmp_path):
        """'Therefore' should be classified as conclusion."""
        pdf_path = str(tmp_path / "conc.pdf")
        _make_pdf(pdf_path, ["Therefore, we conclude that the policy must change."])
        from interaction.pdf_parser import ingest_pdf
        chunks = ingest_pdf(pdf_path)
        types = [c.chunk_type for c in chunks]
        assert "conclusion" in types

    def test_deterministic_chunk_ids(self, tmp_pdf):
        """Same PDF ingest twice → same chunk IDs."""
        from interaction.pdf_parser import ingest_pdf
        chunks_1 = ingest_pdf(tmp_pdf)
        chunks_2 = ingest_pdf(tmp_pdf)
        assert [c.chunk_id for c in chunks_1] == [c.chunk_id for c in chunks_2]


# ===========================================================================
# 4. Full Pipeline: PDF → Store → Query
# ===========================================================================

@pytest.mark.slow
class TestPDFStoreAndQuery:

    def test_store_and_retrieve(self, tmp_pdf, tmp_stores):
        from interaction.pdf_parser import ingest_pdf
        from memory.document import DocumentMemory

        chunks = ingest_pdf(tmp_pdf)
        dm = DocumentMemory()
        dm.store(chunks)

        assert dm.count() == len(chunks)

        results = dm.retrieve("carbon taxes renewable energy", top_k=3)
        assert len(results) > 0
        assert all(r.text for r in results)

    def test_retrieve_by_slide_number(self, tmp_pdf, tmp_stores):
        from interaction.pdf_parser import ingest_pdf
        from memory.document import DocumentMemory

        chunks = ingest_pdf(tmp_pdf)
        dm = DocumentMemory()
        dm.store(chunks)

        results = dm.retrieve("emissions", top_k=5, slide_number=2)
        for r in results:
            assert r.slide_number == 2

    def test_upsert_is_idempotent(self, tmp_pdf, tmp_stores):
        """Storing the same PDF twice should not duplicate chunks."""
        from interaction.pdf_parser import ingest_pdf
        from memory.document import DocumentMemory

        chunks = ingest_pdf(tmp_pdf)
        dm = DocumentMemory()
        dm.store(chunks)
        count_1 = dm.count()
        dm.store(chunks)  # second ingest
        count_2 = dm.count()

        assert count_1 == count_2, "Re-ingesting the same PDF must not duplicate chunks"

    def test_clear_removes_all(self, tmp_pdf, tmp_stores):
        from interaction.pdf_parser import ingest_pdf
        from memory.document import DocumentMemory

        chunks = ingest_pdf(tmp_pdf)
        dm = DocumentMemory()
        dm.store(chunks)
        dm.clear()
        assert dm.count() == 0


# ===========================================================================
# 6. Recency Weighting Tests
# ===========================================================================

class TestRecencyWeighting:

    def _make_result(self, session_index: int, distance: float, id_: str) -> dict:
        return {
            "id": id_,
            "document": f"Claim from session {session_index}",
            "metadata": {"session_index": session_index},
            "distance": distance,
        }

    def test_recent_result_scores_higher(self):
        from memory.recency import rerank_with_recency

        results = [
            self._make_result(session_index=0, distance=0.3, id_="old"),  # 3 sessions ago
            self._make_result(session_index=3, distance=0.3, id_="new"),  # current
        ]
        reranked = rerank_with_recency(results, current_session_index=3, decay_factor=0.85)
        ids = [r["id"] for r in reranked]
        assert ids[0] == "new", "Most recent result should rank first when similarity is equal"

    def test_decay_applied_correctly(self):
        from memory.recency import rerank_with_recency

        results = [self._make_result(session_index=0, distance=0.0, id_="r1")]
        reranked = rerank_with_recency(results, current_session_index=2, decay_factor=0.5)
        # similarity = 1 - 0/2 = 1.0, age = 2, score = 1.0 * 0.5^2 = 0.25
        assert abs(reranked[0]["recency_score"] - 0.25) < 1e-6

    def test_missing_session_index_treated_as_oldest(self):
        from memory.recency import rerank_with_recency

        result_no_meta = {"id": "old", "document": "old claim", "metadata": {}, "distance": 0.1}
        result_with_meta = self._make_result(session_index=5, distance=0.1, id_="new")
        reranked = rerank_with_recency(
            [result_no_meta, result_with_meta],
            current_session_index=5,
            decay_factor=0.85,
        )
        assert reranked[0]["id"] == "new"

    def test_empty_results_returned_unchanged(self):
        from memory.recency import rerank_with_recency
        assert rerank_with_recency([], current_session_index=1) == []

    def test_current_session_no_decay(self):
        """A result from the current session (age=0) should not be decayed."""
        from memory.recency import rerank_with_recency
        results = [self._make_result(session_index=5, distance=0.0, id_="r1")]
        reranked = rerank_with_recency(results, current_session_index=5, decay_factor=0.85)
        # score = 1.0 * 0.85^0 = 1.0
        assert abs(reranked[0]["recency_score"] - 1.0) < 1e-6

    def test_resolved_contradictions_filtered(self):
        from memory.recency import filter_resolved_contradictions

        results = [
            {"id": "a", "document": "unresolved", "metadata": {}, "distance": 0.1},
            {"id": "b", "document": "resolved", "metadata": {"contradiction_resolved": "true"}, "distance": 0.05},
        ]
        filtered = filter_resolved_contradictions(results)
        assert len(filtered) == 1
        assert filtered[0]["id"] == "a"

    def test_empty_memory_returns_empty_list(self):
        from memory.recency import handle_empty_memory
        result = handle_empty_memory("episodic_claims")
        assert result == []

    def test_annotate_with_session_index(self):
        from memory.recency import annotate_with_session_index
        meta = {"claim_id": "abc"}
        annotated = annotate_with_session_index(meta, session_index=3)
        assert annotated["session_index"] == 3
        assert annotated["claim_id"] == "abc"  # original preserved


# ===========================================================================
# 7. PDF Re-upload Edge Case
# ===========================================================================

@pytest.mark.slow
class TestPDFReupload:

    def test_detect_reupload_true_when_chunks_exist(self, tmp_pdf, tmp_stores):
        from interaction.pdf_parser import ingest_pdf
        from memory.document import DocumentMemory
        from memory.recency import detect_pdf_reupload

        chunks = ingest_pdf(tmp_pdf)
        dm = DocumentMemory()
        dm.store(chunks)

        assert detect_pdf_reupload(tmp_pdf) is True

    def test_detect_reupload_false_when_empty(self, tmp_pdf, tmp_stores):
        from memory.recency import detect_pdf_reupload
        # No chunks stored yet
        result = detect_pdf_reupload(tmp_pdf)
        assert result is False
