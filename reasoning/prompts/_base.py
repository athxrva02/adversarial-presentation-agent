"""
Shared prompt utilities for the Reasoning Layer.

Design goals:
- Keep prompts consistent across tasks (classification, question gen, summarise, score)
- Enforce groundedness: only use provided context; cite IDs
- Encourage strict structured output where needed
- Keep context compact (avoid dumping huge memory bundles verbatim)

This file contains:
- common system rules
- rendering helpers for DocumentChunk / memories
- schema hint helpers
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

# Keep these rules compact. Put task-specific instructions in the task prompt builders.
BASE_SYSTEM_RULES = """You are the LLM Reasoning Module in an adversarial presentation coaching system.

Hard rules:
- Use ONLY the provided document context and retrieved memory snippets. Do not invent facts or past sessions.
- If you reference a document chunk, cite it using its chunk_id.
- If you reference a prior claim/session/pattern/common-ground entry, cite its id.
- If the user input is ambiguous, ask a clarification question instead of guessing.
- Be concise and operational. Avoid long explanations unless asked.
"""


def json_only_system(schema_hint: str) -> str:
    """
    System prompt for strict JSON tasks.
    Keep it short to reduce token usage and increase compliance.
    """
    return (
        BASE_SYSTEM_RULES
        + "\nOutput format:\n"
        + "- Output ONLY valid JSON.\n"
        + "- No markdown, no code fences, no commentary.\n"
        + "- Follow the schema exactly. If a field is unknown, use null or an empty list.\n\n"
        + "Schema:\n"
        + schema_hint.strip()
        + "\n"
    )


def text_system() -> str:
    """
    System prompt for plain-text tasks (rare; most of the tasks should be structured).
    """
    return BASE_SYSTEM_RULES


def _truncate(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def render_document_context(document_chunks: Optional[Iterable[Any]], *, max_chars: int = 500) -> str:
    """
    Render a compact, ID-citable view of document chunks.

    Expected object shape (duck-typed):
      - chunk_id: str
      - chunk_type: str
      - slide_number: Optional[int]
      - text: str
    """
    if not document_chunks:
        return "DOCUMENT_CONTEXT: (none)\n"

    lines = ["DOCUMENT_CONTEXT:"]
    for ch in document_chunks:
        chunk_id = getattr(ch, "chunk_id", None) or ch.get("chunk_id")
        chunk_type = getattr(ch, "chunk_type", None) or ch.get("chunk_type")
        slide_number = getattr(ch, "slide_number", None) if not isinstance(ch, dict) else ch.get("slide_number")
        text = getattr(ch, "text", None) or ch.get("text", "")

        header = f"- [{chunk_id}]"
        if chunk_type:
            header += f" type={chunk_type}"
        if slide_number is not None:
            header += f" slide={slide_number}"
        snippet = _truncate(str(text), max_chars=max_chars)
        lines.append(f"{header}\n  {snippet}")

    return "\n".join(lines) + "\n"


def render_claims(claims: Optional[Iterable[Any]], *, max_items: int = 6, max_chars: int = 280) -> str:
    """
    Render compact prior claims for grounding / contradiction awareness.

    Expected shape:
      - claim_id: str
      - claim_text: str
      - session_id: str (optional)
      - turn_number: int (optional)
      - alignment: str (optional)
    """
    if not claims:
        return "PRIOR_CLAIMS: (none)\n"

    lines = ["PRIOR_CLAIMS:"]
    count = 0
    for c in claims:
        if count >= max_items:
            break
        claim_id = getattr(c, "claim_id", None) or c.get("claim_id")
        claim_text = getattr(c, "claim_text", None) or c.get("claim_text", "")
        session_id = getattr(c, "session_id", None) if not isinstance(c, dict) else c.get("session_id")
        turn_number = getattr(c, "turn_number", None) if not isinstance(c, dict) else c.get("turn_number")
        alignment = getattr(c, "alignment", None) if not isinstance(c, dict) else c.get("alignment")

        meta = f"- [{claim_id}]"
        if session_id:
            meta += f" session={session_id}"
        if turn_number is not None:
            meta += f" turn={turn_number}"
        if alignment:
            meta += f" alignment={alignment}"
        lines.append(f"{meta}\n  {_truncate(str(claim_text), max_chars=max_chars)}")
        count += 1

    return "\n".join(lines) + "\n"


def render_semantic_patterns(patterns: Optional[Iterable[Any]], *, max_items: int = 6, max_chars: int = 220) -> str:
    """
    Render compact cross-session patterns (weaknesses/strengths) to guide questioning.
    Expected shape:
      - pattern_id
      - category
      - text
      - confidence
      - direction
      - status
    """
    if not patterns:
        return "SEMANTIC_PATTERNS: (none)\n"

    lines = ["SEMANTIC_PATTERNS:"]
    count = 0
    for p in patterns:
        if count >= max_items:
            break
        pattern_id = getattr(p, "pattern_id", None) or p.get("pattern_id")
        category = getattr(p, "category", None) or p.get("category")
        text = getattr(p, "text", None) or p.get("text", "")
        confidence = getattr(p, "confidence", None) if not isinstance(p, dict) else p.get("confidence")
        direction = getattr(p, "direction", None) if not isinstance(p, dict) else p.get("direction")
        status = getattr(p, "status", None) if not isinstance(p, dict) else p.get("status")

        meta = f"- [{pattern_id}]"
        if category:
            meta += f" category={category}"
        if confidence is not None:
            meta += f" conf={confidence:.2f}" if isinstance(confidence, (int, float)) else f" conf={confidence}"
        if direction:
            meta += f" dir={direction}"
        if status:
            meta += f" status={status}"
        lines.append(f"{meta}\n  {_truncate(str(text), max_chars=max_chars)}")
        count += 1

    return "\n".join(lines) + "\n"


def render_common_ground(entries: Optional[Iterable[Any]], *, max_items: int = 6, max_chars: int = 240) -> str:
    """
    Render negotiated common-ground entries to preserve continuity.
    Expected shape:
      - cg_id
      - negotiated_text
      - pdf_chunk_ref (optional)
      - version (optional)
    """
    if not entries:
        return "COMMON_GROUND: (none)\n"

    lines = ["COMMON_GROUND:"]
    count = 0
    for e in entries:
        if count >= max_items:
            break
        cg_id = getattr(e, "cg_id", None) or e.get("cg_id")
        negotiated_text = getattr(e, "negotiated_text", None) or e.get("negotiated_text", "")
        pdf_chunk_ref = getattr(e, "pdf_chunk_ref", None) if not isinstance(e, dict) else e.get("pdf_chunk_ref")
        version = getattr(e, "version", None) if not isinstance(e, dict) else e.get("version")

        meta = f"- [{cg_id}]"
        if pdf_chunk_ref:
            meta += f" doc_ref={pdf_chunk_ref}"
        if version is not None:
            meta += f" v={version}"
        lines.append(f"{meta}\n  {_truncate(str(negotiated_text), max_chars=max_chars)}")
        count += 1

    return "\n".join(lines) + "\n"


def render_memory_bundle(memory_bundle: Optional[Any]) -> str:
    """
    Render a compact memory bundle for prompts.

    Expects MemoryBundle-like object with attributes:
      - document_context
      - episodic_claims
      - semantic_patterns
      - common_ground

    This intentionally ignores episodic_sessions by default (too verbose for most turns).
    If you need sessions, add a separate renderer in the specific prompt.
    """
    if memory_bundle is None:
        return "MEMORY_BUNDLE: (none)\n"

    doc_ctx = getattr(memory_bundle, "document_context", None)
    claims = getattr(memory_bundle, "episodic_claims", None)
    patterns = getattr(memory_bundle, "semantic_patterns", None)
    cg = getattr(memory_bundle, "common_ground", None)

    parts = [
        render_document_context(doc_ctx),
        render_claims(claims),
        render_semantic_patterns(patterns),
        render_common_ground(cg),
    ]
    return "\n".join(parts)


def safe_user_input_block(user_input: str, *, max_chars: int = 1000) -> str:
    """
    Format the current user input clearly and compactly.
    """
    return "USER_INPUT:\n" + _truncate(user_input, max_chars=max_chars) + "\n"