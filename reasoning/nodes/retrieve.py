"""Retrieve node — fetches a MemoryBundle from the MemoryModule.

The MemoryModule instance is passed through the graph state under the
``_memory_module`` key.  If absent or ``None``, the node is a no-op and
sets ``memory_bundle`` to ``None``.
"""
from __future__ import annotations

from typing import Any, Dict

from reasoning.state import SessionState


_ALL_STORES = ["document", "episodic", "semantic", "common_ground"]


def run(state: SessionState) -> Dict[str, Any]:
    module = state.get("_memory_module")
    if module is None:
        return {"memory_bundle": None}

    query = state.get("user_input", "")
    bundle = module.retrieve(
        query=query,
        stores=_ALL_STORES,
    )
    return {"memory_bundle": bundle}
