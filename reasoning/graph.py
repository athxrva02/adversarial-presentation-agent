"""
Minimal LangGraph wiring for the Reasoning Layer (LLM Reasoning Module part only).

Graphs:
- build_practice_graph(): classify -> generate_question
- build_session_end_graph(): summarise -> score

SessionRunner:
- handle_user_input(text): runs practice graph for one turn
- end_session(): runs session-end graph and returns SessionRecord
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from datetime import datetime
from uuid import uuid4

from langgraph.graph import StateGraph, END

from reasoning.state import SessionState
from reasoning.nodes.classify import run as classify_run
from reasoning.nodes.retrieve import run as retrieve_run
from reasoning.nodes.generate_question import run as generate_question_run
from reasoning.nodes.summarise import run as summarise_run
from reasoning.nodes.score import run as score_run
from reasoning.edges import route_after_classification
from reasoning.nodes.detect_contradiction import run as detect_contradiction_run
from reasoning.nodes.negotiate import run as negotiate_run
from storage.schemas import CommonGroundEntry, SemanticPattern, SessionRecord
from reasoning.nodes.mediate_contradiction import run as mediate_contradiction_run
import logging
logger = logging.getLogger(__name__)


def build_practice_graph():
    """
    One practice turn:
    - classify user_input (structured)
    - retrieve memory bundle (if MemoryModule is available)
    - generate one adversarial question (text)
    """
    g = StateGraph(SessionState)
    g.add_node("classify", classify_run)
    g.add_node("retrieve", retrieve_run)
    g.add_node("detect_contradiction", detect_contradiction_run)
    g.add_node("generate_question", generate_question_run)
    g.add_node("mediate_contradiction", mediate_contradiction_run)

    g.set_entry_point("classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "detect_contradiction")
    g.add_conditional_edges(
        "detect_contradiction",
        route_after_classification,
        {
            "probe_weak": "generate_question",
            "escalate_contradiction": "mediate_contradiction",
            "request_evidence": "generate_question",
            "redirect": "generate_question",
        },
    )
    g.add_edge("mediate_contradiction", END)
    g.add_edge("generate_question", END)
    return g.compile()


def build_session_end_graph():
    """
    End-of-session pipeline:
    - summarise (structured)
    - score (structured; updates SessionRecord.overall_score)
    """
    g = StateGraph(SessionState)
    g.add_node("summarise", summarise_run)
    g.add_node("score", score_run)
    g.add_node("negotiate", negotiate_run)

    g.set_entry_point("summarise")
    g.add_edge("summarise", "score")
    g.add_edge("score", "negotiate")
    g.add_edge("negotiate", END)

    return g.compile()


class SessionRunner:
    """
    Thin wrapper the UI can call into.

    This keeps state across turns and gives you a clean interface for:
    - practice loop
    - end session
    """

    def __init__(self, *, session_id: Optional[str] = None, memory_module=None, hybrid_memory: bool = True):
        self.practice_graph = build_practice_graph()
        self.end_graph = build_session_end_graph()
        self._memory = memory_module

        sid = session_id or f"sess_{uuid4().hex[:8]}"
        self.state: SessionState = {
            "session_id": sid,

            "user_input": "",
            "turn_number": 0,

            "classification": None,
            "agent_response": None,

            "phase": "practice",

            "memory_bundle": None,
            "conflict_result": None,

            "turns": [],
            "claims": [],

            "session_summary": None,
            "score_breakdown": {},

            "negotiation_items": None,
            "negotiation_decisions": None,

            "session_active": True,

            "memory_mode": "hybrid" if hybrid_memory else "document_only",

            "_memory_module": memory_module,
            "conflict_prior_claim_id": None,

            #Voice Analysis
            "voice_turn_metrics": [],
            "voice_summary": None,

            "used_question_strategies": [],
        }

        self._started_at = datetime.now()

        # Create a placeholder session record so FK constraints hold for claims
        if self._memory is not None:
            self._memory.store_session(
                SessionRecord(
                    session_id=sid,
                    timestamp=self._started_at,
                    duration_seconds=0.0,
                    overall_score=None,
                    strengths=[],
                    weaknesses=[],
                    claims_count=0,
                    contradictions_detected=0,
                ),
                claims=[],
            )

    def handle_user_input(self, text: str) -> str:
        if not self.state.get("session_active", True):
            raise RuntimeError("Session is not active. Start a new SessionRunner.")

        self.state["turn_number"] = int(self.state.get("turn_number", 0)) + 1
        self.state["user_input"] = text
        current_turn = self.state["turn_number"]

        user_turn_idx = len(self.state["turns"])
        self.state["turns"].append({"role": "user", "content": text, "turn_number": current_turn})

        out: Dict[str, Any] = self.practice_graph.invoke(self.state)

        claims = list(out.get("claims", []))
        prior_conflict_id = out.get("conflict_prior_claim_id")

        if prior_conflict_id:
            patched_claims = []
            for c in claims:
                if getattr(c, "turn_number", None) == current_turn and not getattr(c, "prior_conflict", None):
                    patched_claims.append(c.model_copy(update={"prior_conflict": prior_conflict_id}))
                else:
                    patched_claims.append(c)
            claims = patched_claims
            out["claims"] = claims

        self.state.update(out)

        # Enrich user turn with classification data now that we have it
        classification = self.state.get("classification")
        if classification and user_turn_idx < len(self.state["turns"]):
            turn = self.state["turns"][user_turn_idx]
            turn["response_class"] = classification.response_class.value if hasattr(classification.response_class, "value") else str(classification.response_class)
            turn["alignment"] = classification.alignment.value if hasattr(classification.alignment, "value") else str(classification.alignment)
            turn["confidence"] = classification.confidence

        if self._memory is not None:
            for claim in claims:
                if getattr(claim, "turn_number", None) == current_turn:
                    self._memory.store_claim(claim)

        agent_resp = (out.get("agent_response") or "").strip()
        if agent_resp:
            self.state["turns"].append({"role": "agent", "content": agent_resp, "turn_number": current_turn})

        return agent_resp

    def commit_negotiation(self, decisions: list[dict[str, Any]]) -> None:
        self.state["negotiation_decisions"] = decisions

        if self._memory is None:
            return

        items = {
            i.get("item_id"): i
            for i in (self.state.get("negotiation_items") or [])
            if isinstance(i, dict) and i.get("item_id")
        }

        session_id = self.state.get("session_id", "unknown_session")

        failed: list[tuple[str, str]] = []
        for d in decisions:
            item = items.get(d.get("item_id"))
            try:
                if item is None:
                    continue

                decision = str(d.get("decision", "reject")).lower()
                if decision not in {"accept", "update"}:
                    continue

                kind = str(item.get("kind", "")).lower()

                if kind in {"semantic_strength", "semantic_weakness"}:
                    pattern_text = str(d.get("updated_text") or item.get("proposed_text") or "").strip()
                    if not pattern_text:
                        continue
                    category = "strength" if kind == "semantic_strength" else "weakness"
                    try:
                        confidence = float(item.get("confidence", 0.8))
                    except Exception:
                        confidence = 0.8
                    evidence = item.get("evidence") or []
                    if not isinstance(evidence, list):
                        evidence = []
                    pattern = SemanticPattern(
                        pattern_id=str(item.get("pattern_id") or f"sp_{category}_{uuid4().hex[:10]}"),
                        category=category,
                        text=pattern_text,
                        confidence=confidence,
                        direction=str(item.get("direction") or "stable"),
                        first_seen=str(item.get("first_seen") or session_id),
                        last_updated=session_id,
                        session_count=int(item.get("session_count", 1)),
                        status=str(item.get("status") or "active"),
                        evidence=evidence,
                    )
                    self._memory.store_semantic_pattern(pattern)
                    continue

                if kind != "common_ground":
                    continue

                negotiated_text = str(d.get("updated_text") or item.get("proposed_text") or "").strip()
                if not negotiated_text:
                    continue

                base_version = int(item.get("version", 0) or 0)
                entry_version = max(1, base_version + (1 if decision == "update" else 0))

                entry = CommonGroundEntry(
                    cg_id=str(item.get("cg_id") or f"cg_{uuid4().hex[:12]}"),
                    pdf_chunk_ref=item.get("pdf_chunk_ref"),
                    original_text=item.get("original_text"),
                    negotiated_text=negotiated_text,
                    proposed_by=str(d.get("proposed_by") or item.get("proposed_by") or "agent"),
                    session_agreed=session_id,
                    version=entry_version,
                    timestamp=datetime.now(),
                )
                self._memory.store_common_ground(entry)
            except Exception as exc:
                logger.exception(
                    "Negotiation commit failed for item_id=%s decision=%s",
                    item,
                    d.get("decision"),
                )
                failed.append((item or "<missing_item_id>", str(exc)))
        if failed:
            failed_ids = [fid for fid, _ in failed]
            raise RuntimeError(f"Negotiation commit partial failure: {failed_ids}")

    def end_session(self) -> Any:
        """
        End session and produce SessionRecord (with overall_score filled).
        """
        if not self.state.get("session_active", True):
            return self.state.get("session_summary")

        self.state["session_active"] = False
        self.state["phase"] = "assessment"

        # optional duration for SessionRecord
        duration = (datetime.now() - self._started_at).total_seconds()
        self.state["duration_seconds"] = float(duration)

        out: Dict[str, Any] = self.end_graph.invoke(self.state)
        self.state.update(out)

        # Persist session record and promote patterns
        if self._memory is not None:
            session_record = self.state.get("session_summary")
            session_record.duration_seconds = duration
            if session_record is not None:
                self._memory.store_session(
                    session_record, self.state.get("claims", [])
                )
                self._memory.promote_patterns(
                    self.state.get("session_id", "")
                )

        return self.state.get("session_summary")
    
    def reset_state(self, *, new_session_id: Optional[str] = None) -> None:
        sid = new_session_id or f"sess_{uuid4().hex[:8]}"
        self.state = {
            "session_id": sid,
            "user_input": "",
            "turn_number": 0,
            "classification": None,
            "agent_response": None,
            "phase": "practice",
            "memory_bundle": None,
            "conflict_result": None,
            "turns": [],
            "claims": [],
            "session_summary": None,
            "score_breakdown": {},
            "negotiation_items": None,
            "negotiation_decisions": None,
            "session_active": True,
            "memory_mode": self.state.get("memory_mode", "hybrid"),
            "_memory_module": self._memory,
            "conflict_prior_claim_id": None,
            "used_question_strategies": [],
        }
        self._started_at = datetime.now()
