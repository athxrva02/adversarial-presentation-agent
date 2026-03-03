# tests/test_json_utils.py
import pytest
from reasoning.json_utils import parse_json, JSONParseError


def test_parse_plain_json():
    obj = parse_json('{"a": 1, "b": 2}')
    assert obj["a"] == 1
    assert obj["b"] == 2


def test_parse_json_in_code_fence():
    obj = parse_json("```json\n{\"a\": 1}\n```")
    assert obj["a"] == 1


def test_parse_json_with_extra_text():
    obj = parse_json("Sure!\n\n{\"a\": 1, \"b\": [1,2]}")
    assert obj["b"] == [1, 2]


def test_parse_raises_when_no_json():
    with pytest.raises(JSONParseError):
        parse_json("No JSON here.")