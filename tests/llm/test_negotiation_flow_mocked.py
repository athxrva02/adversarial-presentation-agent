from datetime import datetime

from reasoning.graph import SessionRunner
from reasoning.nodes.negotiate import _stable_cg_id, run as negotiate_run
from storage.schemas import (
    CommonGroundEntry,
    ConflictAction,
    ConflictResult,
    ConflictStatus,
    SessionRecord,
)


class _MemoryStub:
    def __init__(self):
        self.common_ground = []
        self.patterns = []

    def store_session(self, session, claims):
        return None

    def store_common_ground(self, entry):
        self.common_ground.append(entry)

    def store_semantic_pattern(self, pattern):
        self.patterns.append(pattern)

    def store_claim(self, claim):
        return None

    def promote_patterns(self, session_id):
        return []


def _summary() -> SessionRecord:
    return SessionRecord(
        session_id="s1",
        timestamp=datetime(2026, 1, 1),
        duration_seconds=120.0,
        overall_score=70.0,
        strengths=["Good structure"],
        weaknesses=["Vague metric definitions"],
        claims_count=4,
        contradictions_detected=1,
    )


def _conflict() -> ConflictResult:
    return ConflictResult(
        status=ConflictStatus.TRUE_CONTRADICTION,
        action=ConflictAction.CLARIFY,
        current_claim="A",
        prior_claim="B",
        explanation="A conflicts with B.",
    )


def test_negotiate_node_generates_versioned_common_ground_items():
    next_step = "State baseline before reporting improvements."
    existing = CommonGroundEntry(
        cg_id=_stable_cg_id(next_step),
        pdf_chunk_ref=None,
        original_text="old",
        negotiated_text="State baseline first.",
        proposed_by="agent",
        session_agreed="s0",
        version=4,
        timestamp=datetime(2026, 1, 1),
    )

    class _Bundle:
        common_ground = [existing]

    state = {
        "session_summary": _summary(),
        "score_breakdown": {
            "notes": {"most_important_next_step": next_step},
            "open_issues": ["Define train/val/test split explicitly."],
        },
        "memory_bundle": _Bundle(),
        "conflict_result": _conflict(),
    }

    out = negotiate_run(state)
    items = out["negotiation_items"]

    assert any(i["kind"] == "semantic_strength" for i in items)
    assert any(i["kind"] == "semantic_weakness" for i in items)

    next_step_item = next(i for i in items if i.get("source") == "score_breakdown.notes.most_important_next_step")
    assert next_step_item["kind"] == "common_ground"
    assert next_step_item["cg_id"] == _stable_cg_id(next_step)
    assert next_step_item["version"] == 4
    assert next_step_item["original_text"] == "State baseline first."

    assert any(i.get("source") == "conflict_result" for i in items)


def test_commit_negotiation_persists_common_ground_and_semantic_items():
    mem = _MemoryStub()
    runner = SessionRunner(session_id="s_commit", memory_module=mem)

    runner.state["negotiation_items"] = [
        {
            "item_id": "n1",
            "kind": "common_ground",
            "cg_id": "cg_x",
            "version": 2,
            "proposed_text": "Agree on baseline details.",
            "proposed_by": "agent",
            "pdf_chunk_ref": None,
            "original_text": "old text",
        },
        {
            "item_id": "n2",
            "kind": "semantic_strength",
            "proposed_text": "Structured answer flow.",
        },
        {
            "item_id": "n3",
            "kind": "semantic_weakness",
            "proposed_text": "Missing metric definition.",
        },
    ]

    runner.commit_negotiation(
        [
            {
                "item_id": "n1",
                "decision": "update",
                "updated_text": "Agree baseline = ResNet-50 on CIFAR-10.",
                "proposed_by": "user",
            },
            {
                "item_id": "n2",
                "decision": "accept",
            },
            {
                "item_id": "n3",
                "decision": "reject",
            },
        ]
    )

    assert len(mem.common_ground) == 1
    entry = mem.common_ground[0]
    assert entry.cg_id == "cg_x"
    assert entry.version == 3
    assert entry.proposed_by == "user"
    assert "ResNet-50" in entry.negotiated_text

    assert len(mem.patterns) == 1
    pattern = mem.patterns[0]
    assert pattern.category == "strength"
    assert "Structured answer flow" in pattern.text
