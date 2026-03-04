# tests/test_generate_question_node_live.py
import pytest
from urllib.request import urlopen, Request
from urllib.error import URLError

from reasoning.nodes.generate_question import run


def _ollama_reachable() -> bool:
    try:
        req = Request("http://localhost:11434/api/tags", headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except URLError:
        return False
    except Exception:
        return False


def test_generate_question_node_live(live_enabled):
    if not live_enabled:
        pytest.skip("Live test disabled. Run with: pytest --live")
    if not _ollama_reachable():
        pytest.skip("Ollama not reachable at http://localhost:11434")

    state = {
        "user_input": "Our approach improves accuracy by 15% compared to baseline.",
        "turn_number": 1,
        "session_id": "live_sess",
        "memory_bundle": None,
        "classification": None,
    }

    result = run(state)
    assert "agent_response" in result
    q = result["agent_response"].strip()

    assert len(q) > 0
    # Should look like a question
    assert q.endswith("?") or q.lower().startswith(("why", "how", "what", "which", "when", "where", "who", "can", "could", "would", "do", "does", "did", "is", "are", "should"))
    # Keep it concise for a live agent
    assert len(q) <= 300