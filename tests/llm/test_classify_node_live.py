# tests/test_classify_node_live.py
import pytest
import requests

from reasoning.nodes.classify import run


def _ollama_reachable() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def test_classify_node_live(live_enabled):
    if not live_enabled:
        pytest.skip("Live test disabled. Run with: pytest --live")
    if not _ollama_reachable():
        pytest.skip("Ollama not reachable at http://localhost:11434")

    state = {
        "user_input": "Our approach improves accuracy by 15% compared to baseline.",
        "turn_number": 1,
        "session_id": "live_sess",
        "memory_bundle": None,
    }

    result = run(state)

    # We don't assert exact labels since the model may vary.
    assert "classification" in result
    assert 0.0 <= float(result["classification"].confidence) <= 1.0
    assert result["classification"].response_class.value in {"strong", "weak", "contradiction", "evasion"}
    assert result["classification"].alignment.value in {"supported", "contradicted", "unsupported", "novel", "negotiated"}
    assert "claims" in result
    assert len(result["claims"]) == 1