import builtins

import session


class _RunnerStub:
    def __init__(self, items):
        self.state = {"negotiation_items": items}
        self.committed = None

    def commit_negotiation(self, decisions):
        self.committed = decisions


def test_session_negotiation_clarify_default_becomes_update(monkeypatch, capsys):
    runner = _RunnerStub(
        [
            {
                "item_id": "i1",
                "kind": "common_ground",
                "proposed_text": "Define baseline and metric.",
                "source": "test",
                "rationale": "test",
                "default_decision": "clarify",
            }
        ]
    )

    answers = iter(["", "Clarified baseline: ResNet-50, metric: top-1 accuracy"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    session._run_negotiation_phase(runner, voice=False)
    out = capsys.readouterr().out

    assert runner.committed is not None
    assert len(runner.committed) == 1
    d = runner.committed[0]
    assert d["item_id"] == "i1"
    assert d["decision"] == "update"
    assert "ResNet-50" in d["updated_text"]
    assert "No contradictions detected." in out


def test_session_negotiation_clarify_empty_becomes_reject(monkeypatch, capsys):
    runner = _RunnerStub(
        [
            {
                "item_id": "i1",
                "kind": "common_ground",
                "proposed_text": "Define baseline and metric.",
                "source": "test",
                "rationale": "test",
                "default_decision": "clarify",
            }
        ]
    )

    answers = iter(["", ""])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    session._run_negotiation_phase(runner, voice=False)
    out = capsys.readouterr().out

    assert runner.committed is not None
    assert len(runner.committed) == 1
    d = runner.committed[0]
    assert d["item_id"] == "i1"
    assert d["decision"] == "reject"
    assert "No contradictions detected." in out


def test_session_negotiation_reports_contradiction_items(monkeypatch, capsys):
    runner = _RunnerStub(
        [
            {
                "item_id": "i1",
                "kind": "common_ground",
                "source": "conflict_result",
                "proposed_text": "Resolve contradiction: claim mismatch.",
                "conflict_explanation": "Claim mismatch.",
                "current_claim": "Model accuracy improved.",
                "past_claim": "Model accuracy decreased.",
                "default_decision": "clarify",
            }
        ]
    )

    answers = iter(["a"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    session._run_negotiation_phase(runner, voice=False)
    out = capsys.readouterr().out

    assert runner.committed is not None
    assert len(runner.committed) == 1
    assert runner.committed[0]["decision"] == "accept"
    assert "contradiction item(s) to review" in out
    assert "Contradiction detected: Claim mismatch." in out
