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

from storage.schemas import SessionRecord
from reasoning.state import SessionState
from reasoning.nodes.classify import run as classify_run
from reasoning.nodes.retrieve import run as retrieve_run
from reasoning.nodes.generate_question import run as generate_question_run
from reasoning.nodes.summarise import run as summarise_run
from reasoning.nodes.score import run as score_run


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
    g.add_node("generate_question", generate_question_run)

    g.set_entry_point("classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "generate_question")
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

    g.set_entry_point("summarise")
    g.add_edge("summarise", "score")
    g.add_edge("score", END)

    return g.compile()


class SessionRunner:
    """
    Thin wrapper the UI can call into.

    This keeps state across turns and gives you a clean interface for:
    - practice loop
    - end session
    """

    def __init__(self, *, session_id: Optional[str] = None, memory_module=None):
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

            "_memory_module": memory_module,
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
        """
        Process a single practice turn and return the agent's next question.
        """
        if not self.state.get("session_active", True):
            raise RuntimeError("Session is not active. Start a new SessionRunner.")

        self.state["turn_number"] = int(self.state.get("turn_number", 0)) + 1
        self.state["user_input"] = text

        # Track turns (for summarisation/scoring later)
        self.state["turns"].append({"role": "user", "content": text})

        # Run one practice step (includes retrieve node if memory_module set)
        out: Dict[str, Any] = self.practice_graph.invoke(self.state)
        self.state.update(out)

        # Persist new claims from this turn
        if self._memory is not None:
            for claim in out.get("claims", []):
                self._memory.store_claim(claim)

        # Store assistant question in turn history too
        agent_resp = (out.get("agent_response") or "").strip()
        if agent_resp:
            self.state["turns"].append({"role": "assistant", "content": agent_resp})

        return agent_resp

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
            if session_record is not None:
                self._memory.store_session(
                    session_record, self.state.get("claims", [])
                )
                self._memory.promote_patterns(
                    self.state.get("session_id", "")
                )

        return self.state.get("session_summary")