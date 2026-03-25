from reasoning.prompts.question_generation import build_question_generation_prompt
from storage.schemas import ConflictAction, ConflictResult, ConflictStatus


def test_question_prompt_includes_conflict_signal_when_present():
    conflict = ConflictResult(
        status=ConflictStatus.TRUE_CONTRADICTION,
        action=ConflictAction.CLARIFY,
        current_claim="We improved latency by 50%.",
        prior_claim="Latency got worse under load.",
        explanation="Claims conflict across turns.",
    )

    prompt = build_question_generation_prompt(
        utterance="Latency improved by 50%.",
        memory_bundle=None,
        classification=None,
        conflict_result=conflict,
    )

    user = prompt["user"]
    assert "CONFLICT_SIGNAL" in user
    assert "true_contradiction" in user
    assert "Latency got worse under load." in user
    assert "reconciliation question" in user


def test_question_prompt_includes_default_no_conflict_signal():
    prompt = build_question_generation_prompt(
        utterance="Our model performs well.",
        memory_bundle=None,
        classification=None,
        conflict_result=None,
    )
    user = prompt["user"]
    assert "CONFLICT_SIGNAL" in user
    assert "status: no_conflict" in user


def test_question_prompt_supports_document_driven_mode():
    prompt = build_question_generation_prompt(
        utterance="Our model performs well.",
        memory_bundle=None,
        classification=None,
        conflict_result=None,
        question_mode="document_driven",
        document_priority_chunks=[],
        required_focus="definition",
    )
    user = prompt["user"]
    assert "QUESTION_MODE: document_driven" in user
    assert "REQUIRED_FOCUS: definition" in user
    assert "introduce a new but relevant question from the uploaded document" in user
