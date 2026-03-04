"""
Manual CLI script for testing the agent in a local environment, without the UI.

It allows you to have a conversation with the agent, 
and then end the session to see the final summary and score.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from reasoning.graph import SessionRunner


def _print_header(session_id: str) -> None:
    print("=" * 80)
    print("Adversarial Presentation Agent — Dev CLI")
    print(f"Session: {session_id}")
    print("Commands: /end to finish session, /state to inspect current state, /help")
    print("=" * 80)


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, default=str, ensure_ascii=False)
    except Exception:
        return str(obj)


def main() -> None:
    runner = SessionRunner()
    _print_header(runner.state["session_id"])

    while True:
        try:
            user = input("\nYou: ").strip()
        except EOFError:
            user = "/end"

        if not user:
            continue

        if user.lower() in {"/help", "help"}:
            print("\nCommands:")
            print("  /end   - end session, summarise + score")
            print("  /state - print current internal state (debug)")
            print("  /help  - show this help")
            continue

        if user.lower() == "/state":
            # Print a compact debug view
            s = runner.state.copy()
            # avoid dumping large objects verbatim
            if "memory_bundle" in s and s["memory_bundle"] is not None:
                s["memory_bundle"] = "(memory_bundle present)"
            print("\nSTATE (debug):")
            print(_safe_json(s))
            continue

        if user.lower() in {"/end", "end", "quit", "exit"}:
            print("\nEnding session…\n")
            rec = runner.end_session()
            if rec is None:
                print("No session summary produced.")
                return

            print("SESSION SUMMARY:")
            print(f"- session_id: {rec.session_id}")
            print(f"- timestamp: {rec.timestamp}")
            print(f"- duration_seconds: {rec.duration_seconds:.1f}")
            print(f"- overall_score: {rec.overall_score}")
            print(f"- contradictions_detected: {rec.contradictions_detected}")
            print("\nSTRENGTHS:")
            for x in rec.strengths:
                print(f"  - {x}")
            print("\nWEAKNESSES:")
            for x in rec.weaknesses:
                print(f"  - {x}")

            print("\nSCORE BREAKDOWN (debug):")
            print(_safe_json(runner.state.get("score_breakdown", {})))
            print("\nDone.")
            return

        # Normal practice turn
        try:
            q = runner.handle_user_input(user)
        except Exception as e:
            print(f"\n[ERROR] {type(e).__name__}: {e}")
            continue

        print(f"\nAgent: {q}")


if __name__ == "__main__":
    main()