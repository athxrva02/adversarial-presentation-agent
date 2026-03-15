from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from reasoning.graph import SessionRunner


@dataclass
class ScenarioTurn:
    user: str
    note: str = ""


@dataclass
class Scenario:
    name: str
    turns: List[ScenarioTurn]


SCENARIOS: list[Scenario] = [
    Scenario(
        name="Definitions",
        turns=[
            ScenarioTurn("Our method improves fairness.", "Expect: ask to define 'fairness' metric."),
            ScenarioTurn("Fairness means equal opportunity.", "Expect: ask to formalize / how measured."),
            ScenarioTurn("Equal opportunity = equal true positive rate across groups.", "Expect: which groups / threshold / tradeoffs."),
        ],
    ),
    Scenario(
        name="Evasion",
        turns=[
            ScenarioTurn("We reduce runtime by 3×.", "Expect: ask measurement details."),
            ScenarioTurn("That's not important; let's talk about something else.", "Expect: redirect back; evasion classification."),
            ScenarioTurn("Okay—runtime measured on 1M samples; same hardware.", "Expect: baseline / complexity / scaling."),
        ],
    ),
    Scenario(
        name="Evidence vs method",
        turns=[
            ScenarioTurn("We improve accuracy by 15% compared to baseline.", "Expect: baseline + metric + dataset."),
            ScenarioTurn("Baseline is logistic regression; metric is accuracy on held-out test set.", "Expect: dataset size / leakage / split protocol."),
        ],
    ),
    Scenario(
        name="Contradiction",
        turns=[
            ScenarioTurn("Our model achieves 95% accuracy on the test set.", "Expect: ask about dataset/metric details."),
            ScenarioTurn("We used a 80/20 train-test split on 10,000 samples.", "Expect: follow-up on methodology."),
            ScenarioTurn("Actually, our model only reaches 78% accuracy.", "Expect: contradiction detected with turn 1 (95% vs 78%)."),
        ],
    ),
]


def _make_memory_module():
    """Create a MemoryModule backed by a temporary directory."""
    from memory.module import MemoryModule
    from storage.vector_store import VectorStore
    from storage.relational_store import RelationalStore

    data_dir = os.path.join(tempfile.gettempdir(), "adversarial_scenarios")
    os.makedirs(os.path.join(data_dir, "db"), exist_ok=True)
    vs = VectorStore(chroma_path=os.path.join(data_dir, "chroma"))
    rs = RelationalStore(db_path=os.path.join(data_dir, "db", "scenarios.db"))
    return MemoryModule(vector_store=vs, relational_store=rs)


def run_scenario(s: Scenario) -> None:
    mm = _make_memory_module()
    runner = SessionRunner(memory_module=mm)
    print("=" * 80)
    print(f"SCENARIO: {s.name}")
    print(f"Session: {runner.state['session_id']}")
    print("=" * 80)

    for i, t in enumerate(s.turns, start=1):
        print(f"\nTurn {i} NOTE: {t.note}")
        print(f"You: {t.user}")
        try:
            q = runner.handle_user_input(t.user)
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            break
        cls = runner.state.get("classification")
        print(f"Classification: {cls}")
        print(f"Agent: {q}")

    rec = runner.end_session()
    print("\n" + "-" * 80)
    print("SESSION SUMMARY + SCORE")
    print(f"overall_score: {rec.overall_score}")
    print("strengths:", rec.strengths)
    print("weaknesses:", rec.weaknesses)
    print("breakdown:", runner.state.get("score_breakdown"))
    print("-" * 80)


def main():
    for s in SCENARIOS:
        run_scenario(s)


if __name__ == "__main__":
    main()