"""
Prompt builder for generating an adversarial follow-up question.

Goal:
- Produce ONE high-value follow-up question that is grounded in:
  - the user's latest answer
  - the uploaded document context
  - retrieved session memory
- Increase question diversity across turns
- Keep it short and direct for live practice
"""

from __future__ import annotations

from typing import Any, Optional

from reasoning.prompts._base import (
    text_system,
    render_document_context,
    render_memory_bundle,
    safe_user_input_block,
)


def build_question_generation_prompt(
    *,
    utterance: str,
    memory_bundle: Optional[Any] = None,
    classification: Optional[Any] = None,
    conflict_result: Optional[Any] = None,
    previous_question: str = "",
    recent_questions: Optional[list[str]] = None,
    focused_context: str = "",
    question_mode: str = "answer_driven",
    document_priority_chunks: Optional[list[Any]] = None,
    required_focus: str = "evidence",
) -> dict[str, str]:
    system = (
        text_system()
        + "\nYou are conducting adversarial Q&A practice grounded in BOTH the user's spoken presentation and the uploaded document.\n"
          "When document context is available, treat it as a first-class source for questioning: claims, definitions, evidence, assumptions, limitations, and omissions.\n"
          "Prefer questions that expose a mismatch, unsupported extension, missing justification, or unstated implication between the user's answer and the uploaded document.\n"
          "For this user-facing task, use document content in natural language and do NOT surface internal chunk IDs or other internal identifiers.\n"
          "Output ONLY a single question. No bullet points, no explanations.\n"
          "The question must be concrete, specific, challenging, and tightly scoped to one target.\n"
    )

    if not focused_context:
        focused_context = render_memory_bundle(memory_bundle)

    conflict_block = ""
    if conflict_result is not None:
        conflict_block = (
            "CONFLICT_SIGNAL:\n"
            f"- status: {getattr(conflict_result, 'status', None)}\n"
            f"- action: {getattr(conflict_result, 'action', None)}\n"
            f"- prior_claim: {getattr(conflict_result, 'prior_claim', None)}\n"
            f"- explanation: {getattr(conflict_result, 'explanation', None)}\n\n"
        )
    else:
        conflict_block = (
            "CONFLICT_SIGNAL:\n"
            "- status: no_conflict\n"
            "- action: ignore\n"
            "- prior_claim: null\n"
            "- explanation: none\n\n"
        )

    cls_block = ""
    if classification is not None:
        response_class = getattr(classification, "response_class", None)
        alignment = getattr(classification, "alignment", None)
        conf = getattr(classification, "confidence", None)
        reasoning = getattr(classification, "reasoning", None)

        cls_block = (
            "CLASSIFICATION_SIGNAL:\n"
            f"- response_class: {response_class}\n"
            f"- alignment: {alignment}\n"
            f"- confidence: {conf}\n"
            f"- notes: {reasoning}\n\n"
        )

    recent_questions = list(recent_questions or [])
    if recent_questions:
        recent_questions_block = "RECENT_QUESTION_HISTORY:\n" + "\n".join(
            f"- {q}" for q in recent_questions[-3:]
        ) + "\n\n"
    else:
        recent_questions_block = "RECENT_QUESTION_HISTORY:\n- (none)\n\n"

    # Context expansion change: surface untouched document chunks for document-driven questions
    if document_priority_chunks:
        document_priority_block = (
            "DOCUMENT_PRIORITY_CHUNKS:\n"
            + render_document_context(document_priority_chunks, max_chars=700)
            + "\n"
        )
    else:
        document_priority_block = "DOCUMENT_PRIORITY_CHUNKS: (none)\n\n"

    user = (
        "Task: Ask exactly ONE adversarial follow-up question.\n\n"
        f"QUESTION_MODE: {question_mode}\n\n"
        f"REQUIRED_FOCUS: {required_focus}\n\n"
        "Use BOTH sources when available:\n"
        "- the user's latest spoken answer\n"
        "- the uploaded document context\n\n"
        "Choose exactly ONE attack family:\n"
        "- contradiction with a prior claim\n"
        "- direct non-answer / evasion\n"
        "- mismatch with uploaded document\n"
        "- missing evidence / metric / baseline\n"
        "- undefined or slippery term\n"
        "- unsupported causal leap\n"
        "- unaddressed limitation / trade-off\n"
        "- boundary case / failure mode\n\n"
        "Priority rules:\n"
        "1) If CONFLICT_SIGNAL.status is true_contradiction, ask a reconciliation question.\n"
        "2) If QUESTION_MODE is document_driven, ask from DOCUMENT_PRIORITY_CHUNKS even if the user never mentioned those parts of the document.\n"
        "3) Else if the document context contains a claim, definition, evidence point, or omission that the latest answer ignores, overstates, or conflicts with, ask a document-grounded question.\n"
        "4) Else if the user did not answer the PREVIOUS_QUESTION directly, ask a redirect question.\n"
        "5) Else attack the weakest specific claim in the answer.\n\n"
        "Diversity rules:\n"
        "- Avoid repeating the same attack family as the most recent 1-2 questions unless it is clearly the highest-value next step.\n"
        "- Prefer variety across evidence, definition, mechanism, limitation, trade-off, edge case, and document mismatch.\n"
        "- If the last question asked for evidence, prefer mechanism, limitation, trade-off, or document consistency next unless evidence is still the biggest gap.\n\n"
        "Sharpness rules:\n"
        "- Target REQUIRED_FOCUS only.\n"
        "- Ask about one specific gap, not multiple gaps.\n"
        "- If REQUIRED_FOCUS is evidence, ask for one concrete metric, baseline, comparison, or proof point.\n"
        "- If REQUIRED_FOCUS is definition, ask for one precise definition or operationalization.\n"
        "- If REQUIRED_FOCUS is mechanism, ask how or why the claimed effect happens.\n"
        "- If REQUIRED_FOCUS is limitation, tradeoff, or edge_case, ask about one concrete vulnerability or condition.\n"
        "- Do not combine evidence, limitation, and definition in the same question.\n\n"
        "Document-grounding rules:\n"
        "- In document_driven mode, introduce a new but relevant question from the uploaded document.\n"
        "- It is valid to ask about a definition, evidence claim, limitation, assumption, omission, or implication that was not discussed yet.\n"
        "- Do NOT require the question to be triggered by the latest spoken answer.\n"
        "- If document context is available, anchor the question to a specific statement, definition, evidence snippet, or omission from the uploaded document.\n"
        "- It is acceptable to challenge what the document does NOT justify, not only what it explicitly says.\n"
        "- Use natural language only; do not mention internal IDs or internal system fields.\n\n"
        "Bad questions:\n"
        "- Can you elaborate?\n"
        "- Could you explain more?\n"
        "- What do you mean?\n\n"
        "Constraints:\n"
        "- Ask exactly ONE question.\n"
        "- Keep it <= 2 sentences.\n"
        "- Do not ask multiple sub-questions.\n"
        "- Do not mention internal modules, memory, embeddings, or scoring.\n\n"
        f"PREVIOUS_QUESTION:\n{previous_question or '(none)'}\n\n"
        f"{recent_questions_block}"
        f"{document_priority_block}"
        f"{focused_context}\n"
        f"{conflict_block}"
        f"{cls_block}"
        f"{safe_user_input_block(utterance)}\n"
        "Now output the single best follow-up question."
    )

    return {"system": system, "user": user}
