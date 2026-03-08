"""
PDF ingestion pipeline: argument-level chunking with slide/page metadata.

Public API:
    ingest_pdf(file_path: str) -> list[DocumentChunk]

Pipeline:
    1. Open PDF with PyMuPDF (fitz).
    2. Extract text page-by-page, preserving page → slide mapping.
    3. Split each page into argument-level chunks (paragraph / sentence boundaries),
       respecting max_chunk_tokens from config.
    4. Classify each chunk (claim | definition | evidence | conclusion).
    5. Return a list of DocumentChunk Pydantic objects (no DB writes — caller decides).

Design notes:
- "Slide number" maps directly to PDF page number (1-indexed).
- Chunk IDs are deterministic: sha256(file_path + position)[:16].
- Classification is heuristic (regex keyword matching), keeping this module
  completely offline and free of LLM calls.
- Overlap is handled at the sentence level, not character level, to preserve
  semantic coherence.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config shim
# ---------------------------------------------------------------------------

def _get_config():
    try:
        from config import settings  # type: ignore
        return settings
    except Exception:
        class _Defaults:
            max_chunk_tokens: int = 256
            chunk_overlap_tokens: int = 32
        return _Defaults()


# ---------------------------------------------------------------------------
# Chunk-type heuristics
# ---------------------------------------------------------------------------

# Ordered: first match wins.
_CHUNK_TYPE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("definition", re.compile(
        r"\b(is defined as|refers to|means that|by .{1,40} we mean|definition:?)\b",
        re.IGNORECASE,
    )),
    ("evidence", re.compile(
        r"\b(according to|studies show|data suggest|evidence|figure \d|table \d|"
        r"source:|cite|reference|empirically|statistically|survey|experiment)\b",
        re.IGNORECASE,
    )),
    ("conclusion", re.compile(
        r"\b(therefore|thus|in conclusion|in summary|we conclude|it follows that|"
        r"consequently|as a result|to summarise|overall)\b",
        re.IGNORECASE,
    )),
    ("claim", re.compile(
        r"\b(argue|claim|propose|suggest|believe|assert|maintain|contend|"
        r"hypothesis|should|must|will|can)\b",
        re.IGNORECASE,
    )),
]


def _classify_chunk(text: str) -> str:
    """Return one of: claim | definition | evidence | conclusion."""
    for label, pattern in _CHUNK_TYPE_PATTERNS:
        if pattern.search(text):
            return label
    return "claim"  # Default: treat unclassified text as a claim.


# ---------------------------------------------------------------------------
# Tokenisation approximation
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    """
    Rough token count without a tokeniser (4 chars ≈ 1 token for English).
    Accurate enough for windowing; we never need exact counts here.
    """
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_pages(file_path: str) -> list[tuple[int, str]]:
    """
    Return a list of (page_number_1indexed, page_text) tuples.
    Raises RuntimeError if PyMuPDF is not installed.
    """
    try:
        import fitz  # type: ignore  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is not installed. Run: pip install PyMuPDF"
        ) from exc

    with fitz.open(file_path) as doc:
        pages: list[tuple[int, str]] = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            pages.append((i, text))
    return pages


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Sentence-ending punctuation followed by whitespace + uppercase letter.
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'])')


def _split_sentences(text: str) -> list[str]:
    """Split text into individual sentences, stripping empties."""
    sentences = _SENT_SPLIT.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; returns non-empty stripped paragraphs."""
    raw = re.split(r"\n\s*\n", text)
    return [p.strip() for p in raw if p.strip()]


# ---------------------------------------------------------------------------
# Windowing sentences into token-bounded chunks
# ---------------------------------------------------------------------------

def _window_sentences(
    sentences: list[str],
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Group sentences into chunks of ≤ max_tokens with overlap_tokens look-back.

    Returns:
        List of chunk strings, each ≤ max_tokens (approximately).
    """
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _approx_tokens(sent)

        # If a single sentence is larger than the window, emit it as-is.
        if sent_tokens >= max_tokens:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            chunks.append(sent)
            continue

        if current_tokens + sent_tokens > max_tokens and current:
            # Emit current chunk
            chunks.append(" ".join(current))

            # Build overlap: take sentences from the tail of current
            # whose total tokens ≤ overlap_tokens.
            overlap: list[str] = []
            overlap_budget = overlap_tokens
            for s in reversed(current):
                t = _approx_tokens(s)
                if overlap_budget - t < 0:
                    break
                overlap.insert(0, s)
                overlap_budget -= t
            current = overlap
            current_tokens = sum(_approx_tokens(s) for s in current)

        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Chunk ID generation
# ---------------------------------------------------------------------------

def _make_chunk_id(file_path: str, position: int) -> str:
    """Deterministic, collision-resistant chunk ID."""
    key = f"{file_path}::{position}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_pdf(file_path: str) -> list:
    """
    Parse a PDF into argument-level DocumentChunk objects.

    Args:
        file_path: Absolute or relative path to a .pdf file.

    Returns:
        Ordered list of DocumentChunk objects (position_in_pdf = 1-indexed global
        position across all pages).

    Raises:
        FileNotFoundError: if the file does not exist.
        RuntimeError:      if PyMuPDF is not installed.
        ValueError:        if the file is not a PDF or is empty.
    """
    from storage.schemas import DocumentChunk  # type: ignore  # lazy import avoids circular deps

    path = Path(file_path)
    # Check extension first so callers always get ValueError for wrong file types,
    # regardless of whether the file exists.
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix!r}")
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path!r}")

    cfg = _get_config()
    max_tokens: int = cfg.max_chunk_tokens
    overlap_tokens: int = cfg.chunk_overlap_tokens

    pages = _extract_pages(file_path)
    if not pages:
        raise ValueError(f"PDF has no pages: {file_path!r}")

    chunks: list[DocumentChunk] = []
    global_position = 0  # monotonically increasing across all pages

    for slide_number, page_text in pages:
        if not page_text.strip():
            continue  # Skip blank pages

        paragraphs = _split_paragraphs(page_text)

        for paragraph in paragraphs:
            sentences = _split_sentences(paragraph)
            if not sentences:
                # Treat the whole paragraph as one sentence if split yields nothing
                sentences = [paragraph]

            windowed = _window_sentences(sentences, max_tokens, overlap_tokens)

            for chunk_text in windowed:
                if not chunk_text.strip():
                    continue

                global_position += 1
                chunk_id = _make_chunk_id(file_path, global_position)
                chunk_type = _classify_chunk(chunk_text)

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        slide_number=slide_number,
                        chunk_type=chunk_type,
                        text=chunk_text,
                        position_in_pdf=global_position,
                        embedding_id=None,  # set by vector store after embedding
                        source_file=path.name,  # basename for re-upload detection
                    )
                )

    logger.info(
        "Ingested PDF '%s': %d pages → %d chunks.",
        path.name,
        len(pages),
        len(chunks),
    )
    return chunks
