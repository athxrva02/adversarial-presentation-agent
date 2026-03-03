from __future__ import annotations

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
]


def run_scenario(s: Scenario) -> None:
    runner = SessionRunner()
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