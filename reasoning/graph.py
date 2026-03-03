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
from reasoning.nodes.generate_question import run as generate_question_run
from reasoning.nodes.summarise import run as summarise_run
from reasoning.nodes.score import run as score_run


def build_practice_graph():
    """
    One practice turn:
    - classify user_input (structured)
    - generate one adversarial question (text)
    """
    g = StateGraph(SessionState)
    g.add_node("classify", classify_run)
    g.add_node("generate_question", generate_question_run)

    g.set_entry_point("classify")
    g.add_edge("classify", "generate_question")
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

    def __init__(self, *, session_id: Optional[str] = None):
        self.practice_graph = build_practice_graph()
        self.end_graph = build_session_end_graph()

        sid = session_id or f"sess_{uuid4().hex[:8]}"
        self.state: SessionState = {
            "session_id": sid,

            "user_input": "",
            "turn_number": 0,

            "classification": None,
            "agent_response": None,

            "phase": "practice",
            
            "memory_bundle": None,        # someone else may set this later
            "conflict_result": None,      # someone else may set this later

            "turns": [],
            "claims": [],

            "session_summary": None,
            "score_breakdown": {},

            "negotiation_items": None,          # someone else may set this later
            "negotiation_decisions": None,      # someone else may set this later
            
            "session_active": True,
        }

        self._started_at = datetime.now()

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

        # Run one practice step
        out: Dict[str, Any] = self.practice_graph.invoke(self.state)
        self.state.update(out)

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
        return self.state.get("session_summary")