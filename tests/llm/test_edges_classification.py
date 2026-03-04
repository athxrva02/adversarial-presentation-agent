# tests/test_edges_classification.py
from datetime import datetime

from reasoning.edges import route_after_classification
from storage.schemas import (
    Classification,
    ResponseClass,
    ClaimAlignment,
    ConflictResult,
    ConflictStatus,
    ConflictAction,
)


def _cls(rc: ResponseClass) -> Classification:
    return Classification(
        response_class=rc,
        alignment=ClaimAlignment.NOVEL,
        confidence=0.9,
        reasoning="test",
    )


def test_route_escalates_on_true_contradiction_even_if_classification_strong():
    state = {
        "classification": _cls(ResponseClass.STRONG),
        "conflict_result": ConflictResult(
            status=ConflictStatus.TRUE_CONTRADICTION,
            action=ConflictAction.CLARIFY,
            current_claim="A",
            prior_claim="B",
            explanation="conflict",
        ),
    }
    assert route_after_classification(state) == "escalate_contradiction"


def test_route_redirect_when_classification_missing():
    state = {"classification": None, "conflict_result": None}
    assert route_after_classification(state) == "redirect"


def test_route_redirect_on_evasion():
    state = {"classification": _cls(ResponseClass.EVASION)}
    assert route_after_classification(state) == "redirect"


def test_route_probe_weak_on_weak():
    state = {"classification": _cls(ResponseClass.WEAK)}
    assert route_after_classification(state) == "probe_weak"


def test_route_request_evidence_on_strong():
    state = {"classification": _cls(ResponseClass.STRONG)}
    assert route_after_classification(state) == "request_evidence"


def test_route_escalate_on_classification_contradiction():
    state = {"classification": _cls(ResponseClass.CONTRADICTION)}
    assert route_after_classification(state) == "escalate_contradiction"