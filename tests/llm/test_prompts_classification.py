# tests/test_prompts_classification.py
from reasoning.prompts.classification import build_classification_prompt, CLASSIFICATION_SCHEMA_HINT


def test_classification_prompt_contains_schema_and_input():
    p = build_classification_prompt(utterance="Hello world", memory_bundle=None)
    assert "Schema" in p["system"]
    assert CLASSIFICATION_SCHEMA_HINT.splitlines()[0].strip().startswith("{")
    assert "USER_INPUT" in p["user"]
    assert "Hello world" in p["user"]