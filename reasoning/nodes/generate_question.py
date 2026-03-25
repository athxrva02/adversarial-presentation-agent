"""
LangGraph node: generate an adversarial follow-up question.

Responsibilities:
- Build question-generation prompt using the current user_input + memory_bundle
- Include recent question history so the model can avoid repetition
- Include richer document context so questions can challenge the uploaded PDF too
- Support both answer-driven and document-driven questioning
- Call the LLM as plain text (NOT JSON)
- Return updated state fragment with agent_response
"""

from __future__ import annotations

from typing import Any, Dict

from reasoning.state import SessionState
from reasoning.llm import call_llm_text, opts_practice_question
from reasoning.prompts.question_generation import build_question_generation_prompt
from reasoning.prompts._base import (
    render_document_context,
    render_claims,
    render_semantic_patterns,
    render_common_ground,
)


def _clean_question(text: str) -> str:
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text.strip()

    first = lines[0]
    prefixes = ("Question:", "Q:", "- ", "• ")
    for p in prefixes:
        if first.startswith(p):
            first = first[len(p):].strip()

    if first and not first.endswith("?") and "?" not in first:
        lowers = first.lower()
        if lowers.startswith(
            (
                "why", "how", "what", "which", "when", "where", "who",
                "can", "could", "would", "do", "does", "did",
                "is", "are", "was", "were", "should"
            )
        ):
            first = first + "?"

    return first


# Context expansion change: keep recent question history so we can diversify
def _get_recent_agent_questions(turns: list[dict[str, Any]], limit: int = 3) -> list[str]:
    questions: list[str] = []
    for turn in reversed(turns):
        if str(turn.get("role", "")).strip().lower() == "agent":
            content = str(turn.get("content", "")).strip()
            if content:
                questions.append(content)
            if len(questions) >= limit:
                break
    return list(reversed(questions))


# Context expansion change: diversify the document snippets shown to the LLM
def _pick_document_chunks(document_chunks: list[Any], limit: int = 5) -> list[Any]:
    chunks = list(document_chunks or [])
    if not chunks:
        return []

    chosen: list[Any] = []
    chosen_ids: set[str] = set()
    seen_types: set[str] = set()

    # First pass: try to diversify by chunk type.
    for ch in chunks:
        chunk_id = str(getattr(ch, "chunk_id", "") or "")
        chunk_type = str(getattr(ch, "chunk_type", "") or "").strip().lower()
        if chunk_type and chunk_type not in seen_types:
            chosen.append(ch)
            if chunk_id:
                chosen_ids.add(chunk_id)
            seen_types.add(chunk_type)
        if len(chosen) >= limit:
            return chosen

    # Second pass: fill remaining slots by rank order.
    for ch in chunks:
        chunk_id = str(getattr(ch, "chunk_id", "") or "")
        if chunk_id and chunk_id in chosen_ids:
            continue
        chosen.append(ch)
        if chunk_id:
            chosen_ids.add(chunk_id)
        if len(chosen) >= limit:
            break

    return chosen


# Context expansion change: pass richer document + memory context into question generation
def _build_focused_context(memory_bundle: Any) -> str:
    if memory_bundle is None:
        return "FOCUSED_CONTEXT: (none)\n"

    doc_chunks = _pick_document_chunks(
        list(getattr(memory_bundle, "document_context", []) or []),
        limit=5,
    )
    episodic_claims = list(getattr(memory_bundle, "episodic_claims", []) or [])[:4]
    semantic_patterns = list(getattr(memory_bundle, "semantic_patterns", []) or [])[:3]
    common_ground = list(getattr(memory_bundle, "common_ground", []) or [])[:3]

    parts = ["FOCUSED_CONTEXT:"]
    parts.append(render_document_context(doc_chunks, max_chars=700))
    parts.append(render_claims(episodic_claims, max_items=4, max_chars=320))
    parts.append(render_semantic_patterns(semantic_patterns, max_items=3, max_chars=240))
    parts.append(render_common_ground(common_ground, max_items=3, max_chars=260))
    return "\n".join(parts)


# Context expansion change: fetch untouched document chunks so some questions come directly from the PDF
def _get_document_driven_chunks(state: SessionState, limit: int = 2) -> list[Any]:
    module = state.get("_memory_module")
    if module is None or not hasattr(module, "get_document_question_candidates"):
        return []

    asked = list(state.get("asked_document_chunk_ids", []) or [])
    return list(
        module.get_document_question_candidates(
            limit=limit,
            exclude_chunk_ids=asked,
        )
        or []
    )


# Context expansion change: track which slides/types have already been covered and prefer under-covered chunks
def _chunk_coverage_keys(chunk: Any) -> list[str]:
    keys: list[str] = []
    chunk_type = str(getattr(chunk, "chunk_type", "") or "").strip().lower()
    slide_number = getattr(chunk, "slide_number", None)

    if chunk_type:
        keys.append(f"type:{chunk_type}")
    if slide_number is not None:
        keys.append(f"slide:{slide_number}")
    return keys


# Context expansion change: rank document-driven chunks by coverage novelty before sending them to the prompt
def _prioritize_document_driven_chunks(
    document_chunks: list[Any],
    coverage_keys: list[str],
    limit: int = 2,
) -> list[Any]:
    used = set(coverage_keys or [])

    def _sort_key(chunk: Any) -> tuple[int, int]:
        novelty = sum(1 for key in _chunk_coverage_keys(chunk) if key not in used)
        position = int(getattr(chunk, "position_in_pdf", 0) or 0)
        return (-novelty, position)

    return sorted(list(document_chunks or []), key=_sort_key)[:limit]


# Context expansion change: alternate between answer-driven and document-driven questions
def _choose_question_mode(
    state: SessionState,
    *,
    document_driven_chunks: list[Any],
    retrieved_doc_chunks: list[Any],
) -> str:
    if not document_driven_chunks:
        return "answer_driven"

    recent_modes = list(state.get("question_modes", []) or [])
    last_mode = recent_modes[-1] if recent_modes else ""
    turn_number = int(state.get("turn_number", 0) or 0)

    if last_mode != "document_driven" and (turn_number % 2 == 0 or not retrieved_doc_chunks):
        return "document_driven"

    return "answer_driven"


# Context expansion change: enforce one sharp attack focus per question while rotating across focuses
def _pick_question_focus(
    state: SessionState,
    *,
    question_mode: str,
    document_priority_chunks: list[Any],
    classification: Any,
    conflict_result: Any,
) -> str:
    recent_focus_history = list(state.get("question_focus_history", []) or [])
    recent_focuses = set(recent_focus_history[-2:])

    conflict_status = str(getattr(conflict_result, "status", "") or "").strip().lower()
    response_class = str(getattr(classification, "response_class", "") or "").strip().lower()
    alignment = str(getattr(classification, "alignment", "") or "").strip().lower()

    if conflict_status == "true_contradiction":
        return "contradiction"
    if response_class == "evasion":
        return "evasion"

    preferred: list[str] = []
    if question_mode == "document_driven":
        doc_types = [
            str(getattr(ch, "chunk_type", "") or "").strip().lower()
            for ch in document_priority_chunks
        ]
        if "definition" in doc_types:
            preferred.append("definition")
        if "evidence" in doc_types:
            preferred.append("evidence")
        if "conclusion" in doc_types:
            preferred.append("limitation")
        if "claim" in doc_types:
            preferred.append("mechanism")
        preferred.extend(["tradeoff", "edge_case", "evidence", "definition", "mechanism", "limitation"])
    else:
        if alignment == "unsupported" or response_class == "weak":
            preferred = ["evidence", "definition", "mechanism", "limitation", "edge_case", "tradeoff"]
        else:
            preferred = ["mechanism", "limitation", "tradeoff", "edge_case", "evidence", "definition"]

    for focus in preferred:
        if focus not in recent_focuses:
            return focus

    return preferred[0] if preferred else "evidence"


def run(state: SessionState) -> Dict[str, Any]:
    utterance = state.get("user_input", "")
    memory_bundle = state.get("memory_bundle")
    classification = state.get("classification")
    conflict_result = state.get("conflict_result")

    turns = list(state.get("turns", []) or [])
    recent_questions = _get_recent_agent_questions(turns, limit=3)
    previous_question = recent_questions[-1] if recent_questions else ""

    focused_context = _build_focused_context(memory_bundle)
    retrieved_doc_chunks = list(getattr(memory_bundle, "document_context", []) or []) if memory_bundle else []
    document_driven_chunks = _get_document_driven_chunks(state, limit=4)
    coverage_keys = list(state.get("document_coverage_keys", []) or [])
    document_driven_chunks = _prioritize_document_driven_chunks(
        document_driven_chunks,
        coverage_keys,
        limit=2,
    )
    question_mode = _choose_question_mode(
        state,
        document_driven_chunks=document_driven_chunks,
        retrieved_doc_chunks=retrieved_doc_chunks,
    )
    required_focus = _pick_question_focus(
        state,
        question_mode=question_mode,
        document_priority_chunks=document_driven_chunks,
        classification=classification,
        conflict_result=conflict_result,
    )

    prompt = build_question_generation_prompt(
        utterance=utterance,
        memory_bundle=memory_bundle,
        classification=classification,
        conflict_result=conflict_result,
        previous_question=previous_question,
        recent_questions=recent_questions,
        focused_context=focused_context,
        question_mode=question_mode,
        document_priority_chunks=document_driven_chunks if question_mode == "document_driven" else [],
        required_focus=required_focus,
    )

    raw = call_llm_text(
        system_prompt=prompt["system"],
        user_prompt=prompt["user"],
        options=opts_practice_question(),
    )

    question = _clean_question(raw)
    out: Dict[str, Any] = {
        "agent_response": question,
        "question_modes": [question_mode],
        "question_focus_history": [required_focus],
    }

    if question_mode == "document_driven":
        out["asked_document_chunk_ids"] = [
            str(getattr(ch, "chunk_id", "") or "")
            for ch in document_driven_chunks
            if str(getattr(ch, "chunk_id", "") or "").strip()
        ]
        coverage_updates: list[str] = []
        for chunk in document_driven_chunks:
            coverage_updates.extend(_chunk_coverage_keys(chunk))
        out["document_coverage_keys"] = coverage_updates

    return out
