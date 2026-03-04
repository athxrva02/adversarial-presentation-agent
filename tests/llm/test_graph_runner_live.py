# tests/test_graph_runner_live.py
import pytest
from urllib.request import urlopen, Request
from urllib.error import URLError

from reasoning.graph import SessionRunner


def _ollama_reachable() -> bool:
    try:
        req = Request("http://localhost:11434/api/tags", headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except URLError:
        return False
    except Exception:
        return False


def test_session_runner_end_to_end_live(live_enabled):
    if not live_enabled:
        pytest.skip("Live test disabled. Run with: pytest --live")
    if not _ollama_reachable():
        pytest.skip("Ollama not reachable at http://localhost:11434")

    r = SessionRunner(session_id="s_live")

    q = r.handle_user_input("We improve accuracy by 15% compared to baseline.")
    assert isinstance(q, str) and len(q.strip()) > 0

    rec = r.end_session()
    assert rec is not None
    assert rec.session_id == "s_live"
    assert rec.overall_score is not None
    assert 0.0 <= float(rec.overall_score) <= 100.0