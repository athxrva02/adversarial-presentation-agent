"""
Utilities for extracting and parsing JSON from LLM outputs reliably.

Why this exists:
LLMs often wrap JSON in Markdown fences, prepend explanations, or return
slightly invalid JSON. This module provides:
- code-fence stripping
- best-effort extraction of the first JSON object/array in text
- strict parsing with helpful error messages
- optional "repair prompt" builder for a single retry
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional


class JSONParseError(ValueError):
    """Raised when JSON cannot be extracted/parsed from model output."""


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    """
    If the model wrapped content in ```json ... ```, return the inside.
    If multiple fenced blocks exist, return the first block's content.
    Otherwise, return the original text.
    """
    if not text:
        return text
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _find_balanced_json_span(text: str, start_idx: int) -> Optional[tuple[int, int]]:
    """
    Given text and an index pointing at '{' or '[', find the matching closing brace/bracket,
    respecting JSON strings and escapes. Returns (start, end_exclusive) or None.
    """
    if start_idx < 0 or start_idx >= len(text):
        return None
    opener = text[start_idx]
    if opener not in "{[":
        return None
    closer = "}" if opener == "{" else "]"

    depth = 0
    in_string = False
    escape = False

    for i in range(start_idx, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        # not in string
        if ch == '"':
            in_string = True
            continue
        if ch == opener:
            depth += 1
            continue
        if ch == closer:
            depth -= 1
            if depth == 0:
                return (start_idx, i + 1)

    return None


def extract_first_json(text: str) -> str:
    """
    Extract the first JSON object/array substring from text.

    Strategy:
    1) Strip code fences if present.
    2) Find first '{' or '[' and then locate a balanced span.
    3) Return that span; raise if not found.
    """
    if not text or not text.strip():
        raise JSONParseError("Empty response; expected JSON.")

    cleaned = strip_code_fences(text)

    # Find first plausible JSON opener
    first_obj = cleaned.find("{")
    first_arr = cleaned.find("[")
    if first_obj == -1 and first_arr == -1:
        raise JSONParseError(
            "No JSON opener found ('{' or '['). Raw output did not contain JSON."
        )

    if first_obj == -1:
        start = first_arr
    elif first_arr == -1:
        start = first_obj
    else:
        start = min(first_obj, first_arr)

    span = _find_balanced_json_span(cleaned, start)
    if span is None:
        raise JSONParseError(
            "Found a JSON opener but could not find a balanced closing brace/bracket. "
            "The output may be truncated or invalid."
        )

    return cleaned[span[0] : span[1]].strip()


def parse_json(text: str) -> Any:
    """
    Parse JSON from text, extracting it first if needed.
    Raises JSONParseError with a useful message on failure.
    """
    candidate = extract_first_json(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        # Provide context near the error position for easier debugging
        start = max(e.pos - 60, 0)
        end = min(e.pos + 60, len(candidate))
        snippet = candidate[start:end].replace("\n", "\\n")
        raise JSONParseError(
            f"JSON decoding failed: {e.msg} at pos {e.pos}. "
            f"Nearby snippet: '{snippet}'"
        ) from e


@dataclass(frozen=True)
class RepairPrompt:
    """
    A simple container for a follow-up prompt used to repair invalid JSON.
    """
    system: str
    user: str


def build_json_repair_prompt(
    schema_hint: str,
    raw_output: str,
    error: str,
) -> RepairPrompt:
    """
    Build a follow-up prompt to ask the model to output valid JSON only.

    schema_hint: a compact schema description (not huge).
    raw_output: the model's previous output.
    error: the parsing error message.

    Return:
      RepairPrompt(system=..., user=...)
    """
    system = (
        "You must output ONLY valid JSON. No markdown, no explanations, no code fences. "
        "If you are unsure about a field, use null or an empty list as appropriate. "
        "Follow the schema exactly."
    )
    user = (
        "The previous output was not valid JSON.\n\n"
        f"Schema:\n{schema_hint}\n\n"
        f"Parsing error:\n{error}\n\n"
        "Previous output:\n"
        f"{raw_output}\n\n"
        "Now return ONLY the corrected JSON that matches the schema."
    )
    return RepairPrompt(system=system, user=user)