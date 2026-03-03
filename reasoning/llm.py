"""
LLM client wrapper for the Reasoning Layer.

Goals:
- Single place to configure and call the local Ollama-served model via LangChain ChatOllama
- Provide two call styles:
    1) call_llm_text(...) -> str
    2) call_llm_structured(...) -> dict/list (JSON), with robust parsing + one repair retry
- Keep decoding options explicit and overridable per call
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from reasoning.json_utils import JSONParseError, build_json_repair_prompt, parse_json


# ----------------------------
# Types
# ----------------------------

@dataclass(frozen=True)
class LLMOptions:
    """
    Ollama generation options. These map to the Ollama `options` object.
    """
    temperature: float = 0.2
    num_predict: int = 350          # max tokens to generate
    num_ctx: int = 4096             # context window
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    seed: Optional[int] = None


# ----------------------------
# Client
# ----------------------------

_client: Optional[ChatOllama] = None


def get_llm_client() -> ChatOllama:
    """
    Create (once) and return a ChatOllama client.
    """
    global _client
    if _client is None:
        _client = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.model_name,
        )
    return _client


def _to_ollama_options(opts: Optional[LLMOptions]) -> dict[str, Any]:
    """
    Convert our LLMOptions into Ollama's `options` dict.
    """
    if opts is None:
        return {}

    o: dict[str, Any] = {
        "temperature": opts.temperature,
        "num_predict": opts.num_predict,
        "num_ctx": opts.num_ctx,
    }
    if opts.top_p is not None:
        o["top_p"] = opts.top_p
    if opts.repeat_penalty is not None:
        o["repeat_penalty"] = opts.repeat_penalty
    if opts.seed is not None:
        o["seed"] = opts.seed
    return o


def call_llm_text(
    system_prompt: str,
    user_prompt: str,
    *,
    options: Optional[LLMOptions] = None,
) -> str:
    """
    Plain text call. Use for tasks where you do not require strict JSON.
    """
    llm = get_llm_client()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    # langchain-ollama expects Ollama parameters under `options=...`
    resp = llm.invoke(messages, options=_to_ollama_options(options))
    content = getattr(resp, "content", "")
    return content if isinstance(content, str) else str(content)


def call_llm_structured(
    system_prompt: str,
    user_prompt: str,
    *,
    schema_hint: str,
    options: Optional[LLMOptions] = None,
    retry_on_failure: bool = True,
) -> Any:
    """
    JSON-structured call. Ensures we return a Python object parsed from JSON.

    Behavior:
    - Calls model once with the provided prompts.
    - Attempts to extract+parse JSON from the response.
    - If parsing fails and retry_on_failure=True, performs ONE repair retry using a
      strict "JSON only" prompt built from the schema + error + raw output.
    """
    raw = call_llm_text(system_prompt, user_prompt, options=options)

    try:
        return parse_json(raw)
    except JSONParseError as e:
        if not retry_on_failure:
            raise

        repair = build_json_repair_prompt(
            schema_hint=schema_hint,
            raw_output=raw,
            error=str(e),
        )

        # More deterministic retry
        base = options or LLMOptions(
            temperature=getattr(settings, "temperature", 0.2),
            num_predict=getattr(settings, "max_tokens", 400),
            num_ctx=getattr(settings, "num_ctx", 4096),
        )
        repair_opts = LLMOptions(
            temperature=min(base.temperature, 0.2),
            num_predict=max(base.num_predict, 300),
            num_ctx=base.num_ctx,
            top_p=base.top_p,
            repeat_penalty=base.repeat_penalty,
            seed=base.seed,
        )

        repaired_raw = call_llm_text(repair.system, repair.user, options=repair_opts)
        return parse_json(repaired_raw)


# Convenience presets

def opts_practice_question() -> LLMOptions:
    # Slightly higher temperature for varied questions
    return LLMOptions(
        temperature=0.5,
        num_predict=250,
        num_ctx=getattr(settings, "num_ctx", 4096),
    )


def opts_judge_or_classify() -> LLMOptions:
    # Low temperature for stable judgments and JSON compliance
    return LLMOptions(
        temperature=0.2,
        num_predict=250,
        num_ctx=getattr(settings, "num_ctx", 4096),
    )


def opts_summarise_or_score() -> LLMOptions:
    return LLMOptions(
        temperature=0.2,
        num_predict=450,
        num_ctx=getattr(settings, "num_ctx", 4096),
    )