"""Retrieve node — fetches a MemoryBundle from the MemoryModule.

The MemoryModule instance is passed through the graph state under the
``_memory_module`` key.  If absent or ``None``, the node is a no-op and
sets ``memory_bundle`` to ``None``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from reasoning.state import SessionState

logger = logging.getLogger(__name__)

_ALL_STORES = ["document", "episodic", "semantic", "common_ground"]


def run(state: SessionState) -> Dict[str, Any]:
    module = state.get("_memory_module")
    if module is None:
        logger.debug("retrieve: no memory module, returning None")
        return {"memory_bundle": None}

    query = state.get("user_input", "")
    bundle = module.retrieve(
        query=query,
        stores=_ALL_STORES,
    )
    n_claims = len(bundle.episodic_claims) if bundle else 0
    n_docs = len(bundle.document_context) if bundle else 0
    logger.info(
        "retrieve: query=%r, episodic_claims=%d, document_chunks=%d",
        query[:80], n_claims, n_docs,
    )
    return {"memory_bundle": bundle}
