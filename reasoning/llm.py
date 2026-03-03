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
from typing import Any, Optional, Sequence

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from reasoning.json_utils import JSONParseError, build_json_repair_prompt, parse_json


# ----------------------------
# Types
# ----------------------------

@dataclass(frozen=True)
class LLMOptions:
    """
    Generation options to balance determinism and speed.
    These map to Ollama options through ChatOllama.model_kwargs.
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
    We keep it global so nodes don't re-create it each call.
    """
    global _client
    if _client is None:
        # ChatOllama supports base_url and model name
        _client = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.model_name,
            temperature=getattr(settings, "temperature", 0.2),
        )
    return _client


def _to_model_kwargs(opts: Optional[LLMOptions]) -> dict[str, Any]:
    if opts is None:
        return {}

    mk: dict[str, Any] = {
        "num_predict": opts.num_predict,
        "num_ctx": opts.num_ctx,
        "temperature": opts.temperature,
    }
    if opts.top_p is not None:
        mk["top_p"] = opts.top_p
    if opts.repeat_penalty is not None:
        mk["repeat_penalty"] = opts.repeat_penalty
    if opts.seed is not None:
        mk["seed"] = opts.seed
    return mk


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

    # Override per-call options without rebuilding the client:
    llm = llm.bind(model_kwargs=_to_model_kwargs(options))

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    resp = llm.invoke(messages)
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
        # Use more deterministic settings for repair
        repair_opts = options or LLMOptions()
        repair_opts = LLMOptions(
            temperature=min(repair_opts.temperature, 0.2),
            num_predict=max(repair_opts.num_predict, 300),
            num_ctx=repair_opts.num_ctx,
            top_p=repair_opts.top_p,
            repeat_penalty=repair_opts.repeat_penalty,
            seed=repair_opts.seed,
        )

        repaired_raw = call_llm_text(repair.system, repair.user, options=repair_opts)
        return parse_json(repaired_raw)


# ----------------------------
# Convenience presets (optional)
# ----------------------------

def opts_practice_question() -> LLMOptions:
    """
    Slightly higher temperature for varied questions.
    """
    return LLMOptions(temperature=0.5, num_predict=250, num_ctx=4096)


def opts_judge_or_classify() -> LLMOptions:
    """
    Low temperature for stable judgments and JSON compliance.
    """
    return LLMOptions(temperature=0.2, num_predict=250, num_ctx=4096)


def opts_summarise_or_score() -> LLMOptions:
    """
    Medium token budget; still low temperature.
    """
    return LLMOptions(temperature=0.2, num_predict=450, num_ctx=4096)