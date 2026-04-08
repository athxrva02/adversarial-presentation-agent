"""
Shared fixtures for analysis pipeline tests.

`synthetic_project` creates a minimal but complete project layout in a
temporary directory so tests never touch the real results/ or survey.csv.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic participant data
# ---------------------------------------------------------------------------
PARTICIPANTS = [
    # pid, survey_before, survey_after, s1_dir, s2_dir, condition
    ("P01", "Alice Smith", "Alice Smith", "2026-01-01_10-00-00", "2026-01-01_10-30-00", "hybrid-memory"),
    ("P02", "Bob Jones", "Bob Jones", "2026-01-01_11-00-00", "2026-01-01_11-30-00", "hybrid-memory"),
    ("P03", "Carol Lee", "Carol Lee", "2026-01-01_12-00-00", "2026-01-01_12-30-00", "hybrid-memory"),
    ("P04", "David Kim", "David Kim", "2026-01-01_13-00-00", "2026-01-01_13-30-00", "non-hybrid-memory"),
    ("P05", "Eva Brown", "Eva Brown", "2026-01-01_14-00-00", "2026-01-01_14-30-00", "non-hybrid-memory"),
    ("P06", "Frank Wu", "Frank Wu", "2026-01-01_15-00-00", "2026-01-01_15-30-00", "non-hybrid-memory"),
]

# Overall scores for each (participant, session) — varied to give ANOVA something to work with
SCORES = {
    ("P01", 1): 55.0, ("P01", 2): 70.0,
    ("P02", 1): 60.0, ("P02", 2): 75.0,
    ("P03", 1): 50.0, ("P03", 2): 65.0,
    ("P04", 1): 45.0, ("P04", 2): 55.0,
    ("P05", 1): 40.0, ("P05", 2): 50.0,
    ("P06", 1): 42.0, ("P06", 2): 52.0,
}

# Preparedness labels for session-1 scores (mapped to 1–7)
PREPAREDNESS = {
    "P01": "Somewhat unprepared",      # → 3
    "P02": "Neither prepared nor unprepared",  # → 4
    "P03": "Not prepared",             # → 2
    "P04": "Somewhat unprepared",      # → 3
    "P05": "Neither prepared nor unprepared",  # → 4
    "P06": "Somewhat unprepared",      # → 3
}

# After-survey Likert responses (all 13 questions, same answer per participant for simplicity)
AFTER_LIKERT_RESPONSES: dict[str, str] = {
    "Alice Smith": "Agree",           # mean = 6.0
    "Bob Jones": "Somewhat Agree",    # mean = 5.0
    "Carol Lee": "Strongly Agree",    # mean = 7.0
    "David Kim": "Agree",             # mean = 6.0
    "Eva Brown": "Neutral",           # mean = 4.0
    "Frank Wu": "Somewhat Agree",     # mean = 5.0
}

AFTER_QUESTION_COLS = [
    "Using the agent would enhance my effectiveness in giving presentations",
    "Using the agent would enhance my public speaking skills",
    "I would find the agent useful for my work",
    "My interaction with the agent was clear and understandable",
    "I find the agent easy to use",
    "I would interact with the agent again",
    "I feel involved in the interaction with the agent",
    "I believe the agent is reliable and truthful",
    "I believe I formed a beneficial association with the agent for my presentation",
    "I find the agent logical and consistent",
    "I find the agent's feedback actionable",
    "I believe that my current preparedness is affected by the interaction with the agent",
    "I believe the agent made an impact on me",
]

SUMMARY_COLS = [
    "run_timestamp", "session_id", "memory_type", "pdf_file", "session_timestamp",
    "duration_seconds", "overall_score", "claims_count", "contradictions_detected",
    "contradictions_accepted", "contradictions_rejected", "strengths", "weaknesses",
    "top_priority", "voice_analysis_available", "voice_delivery_score",
    "voice_speaking_rate_wpm", "voice_articulation_rate_wpm", "voice_pause_count",
    "voice_long_pause_count", "voice_mean_pause_s", "voice_silence_ratio",
    "voice_pitch_mean_hz", "voice_pitch_std_hz", "voice_pitch_range_semitones",
    "voice_volume_mean_dbfs", "voice_volume_std_db", "voice_clipping_ratio",
    "voice_feedback",
    "rubric_clarity_structure", "rubric_evidence_specificity",
    "rubric_definition_precision", "rubric_logical_coherence",
    "rubric_handling_adversarial_questions", "rubric_depth_of_understanding",
    "rubric_concession_and_qualification", "rubric_recovery_from_challenge",
]


def _make_summary_row(pid: str, session: int, condition: str, ts: str) -> dict:
    memory_type = "hybrid" if condition == "hybrid-memory" else "document_only"
    return {
        "run_timestamp": ts,
        "session_id": f"sess_{pid}_{session}",
        "memory_type": memory_type,
        "pdf_file": "test.pdf",
        "session_timestamp": ts,
        "duration_seconds": 300.0,
        "overall_score": SCORES[(pid, session)],
        "claims_count": 4,
        "contradictions_detected": 1,
        "contradictions_accepted": 1,
        "contradictions_rejected": 0,
        "strengths": "Clear claim",
        "weaknesses": "Lacks examples",
        "top_priority": "Add evidence",
        "voice_analysis_available": "no",
        **{c: "" for c in SUMMARY_COLS if c.startswith("voice_") and c != "voice_analysis_available"},
        "rubric_clarity_structure": 3,
        "rubric_evidence_specificity": 2,
        "rubric_definition_precision": 3,
        "rubric_logical_coherence": 4,
        "rubric_handling_adversarial_questions": 3,
        "rubric_depth_of_understanding": 3,
        "rubric_concession_and_qualification": 3,
        "rubric_recovery_from_challenge": 3,
    }


@pytest.fixture()
def synthetic_project(tmp_path: Path) -> Path:
    """
    Build a minimal project tree under tmp_path and return its root.

    Structure:
        analysis/
            participants.csv
            Questionnaire Before Interacting.csv
            Questionnaire After Interacting.csv
        results/{timestamp}/
            summary.csv          (one per session, un-augmented)
        analysis.ipynb           (copied from real project)
        analysis_nb.py           (copied from real project)
    """
    root = tmp_path
    analysis_dir = root / "analysis"
    analysis_dir.mkdir()
    results_dir = root / "results"
    results_dir.mkdir()

    # --- participants.csv ---
    parts_rows = []
    for pid, sb, sa, d1, d2, cond in PARTICIPANTS:
        parts_rows.append({
            "participant_id": pid,
            "name": sb.split()[0],
            "survey_name_before": sb,
            "survey_name_after": sa,
            "session_dir_1": d1,
            "session_dir_2": d2,
            "condition": cond,
        })
    pd.DataFrame(parts_rows).to_csv(analysis_dir / "participants.csv", index=False)

    # --- before-survey CSV ---
    before_rows = []
    for i, (pid, sb, sa, d1, d2, cond) in enumerate(PARTICIPANTS):
        before_rows.append({
            "ID": i + 1,
            "Start time": "1/1/26 10:00:00",
            "Completion time": "1/1/26 10:01:00",
            "Email": "anonymous",
            "Name": "",
            "Last modified time": "",
            "Name2": sb,
            "Level of Study": "Master",
            "Field of Study": "Computer Science",
            "Age": "18<30",
            "Column": "Average",
            "2": PREPAREDNESS[pid],
        })
    pd.DataFrame(before_rows).to_csv(
        analysis_dir / "Questionnaire Before Interacting with Agent.csv", index=False
    )

    # --- after-survey CSV ---
    after_rows = []
    for i, (pid, sb, sa, d1, d2, cond) in enumerate(PARTICIPANTS):
        response = AFTER_LIKERT_RESPONSES[sb]
        row = {
            "ID": i + 1,
            "Start time": "1/1/26 11:00:00",
            "Completion time": "1/1/26 11:01:00",
            "Email": "anonymous",
            "Name": "",
            "Last modified time": "",
            "Name2": sa,
        }
        for q in AFTER_QUESTION_COLS:
            row[q] = response
        after_rows.append(row)
    pd.DataFrame(after_rows).to_csv(
        analysis_dir / "Questionnaire After Interacting with Agent.csv", index=False
    )

    # --- results/{timestamp}/summary.csv ---
    for pid, sb, sa, d1, d2, cond in PARTICIPANTS:
        for session, ts_dir in [(1, d1), (2, d2)]:
            sess_dir = results_dir / ts_dir
            sess_dir.mkdir(parents=True)
            row = _make_summary_row(pid, session, cond, ts_dir)
            pd.DataFrame([row]).to_csv(sess_dir / "summary.csv", index=False)

    # --- Copy analysis.ipynb and analysis.py from real project ---
    real_root = Path(__file__).parent.parent.parent
    for fname in ("analysis.ipynb", "analysis_nb.py"):
        src = real_root / fname
        if src.exists():
            shutil.copy(src, root / fname)

    return root
