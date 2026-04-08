"""
Tests for analysis/prepare_data.py and analysis/pipeline.py.

Unit tests use the `synthetic_project` fixture (no network, no real data).
The integration test (marked `integration`) executes the full notebook and
is skipped by default — run with:

    pytest tests/analysis/ -m integration
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis.prepare_data import (
    LIKERT_7,
    PREPAREDNESS_7,
    after_likert_cols,
    augment_summary_csvs,
    build_survey_csv,
    preparedness_to_score,
    run as prepare_run,
)
from analysis.pipeline import prepare, run as pipeline_run, run_notebook

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.parent


def _load_participants(project_root: Path) -> pd.DataFrame:
    return pd.read_csv(project_root / "analysis" / "participants.csv")


def _summary_paths(project_root: Path) -> list[Path]:
    return sorted((project_root / "results").glob("*/summary.csv"))


# ---------------------------------------------------------------------------
# Unit tests — pure functions
# ---------------------------------------------------------------------------

class TestPreparednessToScore:
    @pytest.mark.parametrize("label,expected", [
        ("Very unprepared", 1.0),
        ("Not prepared", 2.0),
        ("Somewhat unprepared", 3.0),
        ("Neither prepared nor unprepared", 4.0),
        ("Somewhat prepared", 5.0),
        ("Prepared", 6.0),
        ("Very prepared", 7.0),
    ])
    def test_known_labels(self, label, expected):
        assert preparedness_to_score(label) == pytest.approx(expected)

    def test_unknown_label_raises(self):
        with pytest.raises(ValueError, match="Unknown preparedness label"):
            preparedness_to_score("Totally unprepared")

    def test_output_always_in_range(self):
        for label in PREPAREDNESS_7:
            score = preparedness_to_score(label)
            assert 1.0 <= score <= 7.0


class TestAfterLikertCols:
    def test_excludes_meta_columns(self, synthetic_project):
        after_csv = next((synthetic_project / "analysis").glob("Questionnaire After*"))
        df = pd.read_csv(after_csv)
        cols = after_likert_cols(df)
        assert "Name2" not in cols
        assert "ID" not in cols
        assert len(cols) == 13  # synthetic fixture has exactly 13 questions


# ---------------------------------------------------------------------------
# Unit tests — augment_summary_csvs
# ---------------------------------------------------------------------------

class TestAugmentSummaryCsvs:
    def test_injects_participant_id_and_session(self, synthetic_project):
        participants = _load_participants(synthetic_project)
        augmented, skipped = augment_summary_csvs(
            participants, synthetic_project / "results"
        )

        assert augmented == len(participants) * 2  # 2 sessions each
        assert skipped == []

        for path in _summary_paths(synthetic_project):
            df = pd.read_csv(path)
            assert "participant_id" in df.columns
            assert "session" in df.columns

    def test_participant_id_values_match_participants_csv(self, synthetic_project):
        participants = _load_participants(synthetic_project)
        augment_summary_csvs(participants, synthetic_project / "results")

        for path in _summary_paths(synthetic_project):
            df = pd.read_csv(path)
            pid = df["participant_id"].iloc[0]
            assert pid in participants["participant_id"].values

    def test_session_values_are_1_or_2(self, synthetic_project):
        participants = _load_participants(synthetic_project)
        augment_summary_csvs(participants, synthetic_project / "results")

        for path in _summary_paths(synthetic_project):
            df = pd.read_csv(path)
            assert df["session"].iloc[0] in (1, 2)

    def test_memory_type_normalised_for_no_memory(self, synthetic_project):
        participants = _load_participants(synthetic_project)
        augment_summary_csvs(participants, synthetic_project / "results")

        no_mem = participants[participants["condition"] == "non-hybrid-memory"]
        for _, row in no_mem.iterrows():
            for dir_col in ("session_dir_1", "session_dir_2"):
                path = synthetic_project / "results" / row[dir_col] / "summary.csv"
                df = pd.read_csv(path)
                assert df["memory_type"].iloc[0] == "document_only"

    def test_skips_missing_directories(self, synthetic_project):
        participants = _load_participants(synthetic_project)

        # Remove one session directory to simulate a missing session
        first_dir = participants.iloc[0]["session_dir_1"]
        missing = synthetic_project / "results" / first_dir / "summary.csv"
        missing.unlink()
        missing.parent.rmdir()

        _, skipped = augment_summary_csvs(participants, synthetic_project / "results")
        assert len(skipped) == 1
        assert first_dir in skipped[0]

    def test_idempotent_on_second_run(self, synthetic_project):
        participants = _load_participants(synthetic_project)
        results_dir = synthetic_project / "results"

        augment_summary_csvs(participants, results_dir)
        first_states = {p: pd.read_csv(p) for p in _summary_paths(synthetic_project)}

        augment_summary_csvs(participants, results_dir)
        for path in _summary_paths(synthetic_project):
            second = pd.read_csv(path)
            assert first_states[path].equals(second), f"Second run changed {path}"


# ---------------------------------------------------------------------------
# Unit tests — build_survey_csv
# ---------------------------------------------------------------------------

class TestBuildSurveyCsv:
    def test_creates_survey_csv(self, synthetic_project):
        participants = _load_participants(synthetic_project)
        analysis_dir = synthetic_project / "analysis"
        before_csv = next(analysis_dir.glob("Questionnaire Before*"))
        after_csv = next(analysis_dir.glob("Questionnaire After*"))
        out = synthetic_project / "survey.csv"

        build_survey_csv(participants, before_csv, after_csv, out)
        assert out.exists()

    def test_row_count(self, synthetic_project):
        participants = _load_participants(synthetic_project)
        analysis_dir = synthetic_project / "analysis"
        before_csv = next(analysis_dir.glob("Questionnaire Before*"))
        after_csv = next(analysis_dir.glob("Questionnaire After*"))
        out = synthetic_project / "survey.csv"

        df = build_survey_csv(participants, before_csv, after_csv, out)
        # 2 sessions per participant
        assert len(df) == len(participants) * 2

    def test_columns(self, synthetic_project):
        participants = _load_participants(synthetic_project)
        analysis_dir = synthetic_project / "analysis"
        before_csv = next(analysis_dir.glob("Questionnaire Before*"))
        after_csv = next(analysis_dir.glob("Questionnaire After*"))
        out = synthetic_project / "survey.csv"

        df = build_survey_csv(participants, before_csv, after_csv, out)
        assert list(df.columns) == ["participant_id", "session", "preparedness_score"]

    def test_preparedness_score_range(self, synthetic_project):
        participants = _load_participants(synthetic_project)
        analysis_dir = synthetic_project / "analysis"
        before_csv = next(analysis_dir.glob("Questionnaire Before*"))
        after_csv = next(analysis_dir.glob("Questionnaire After*"))
        out = synthetic_project / "survey.csv"

        df = build_survey_csv(participants, before_csv, after_csv, out)
        assert df["preparedness_score"].between(1.0, 7.0).all(), (
            f"Scores out of [1, 7]: {df[~df['preparedness_score'].between(1, 7)]}"
        )

    def test_session1_score_comes_from_preparedness(self, synthetic_project):
        participants = _load_participants(synthetic_project)
        analysis_dir = synthetic_project / "analysis"
        before_csv = next(analysis_dir.glob("Questionnaire Before*"))
        after_csv = next(analysis_dir.glob("Questionnaire After*"))
        out = synthetic_project / "survey.csv"

        df = build_survey_csv(participants, before_csv, after_csv, out)
        s1 = df[df["session"] == 1]

        # "Somewhat unprepared" → 3, "Neither..." → 4, "Not prepared" → 2
        expected = {"P01": 3.0, "P02": 4.0, "P03": 2.0, "P04": 3.0, "P05": 4.0, "P06": 3.0}
        for pid, exp in expected.items():
            actual = s1.loc[s1["participant_id"] == pid, "preparedness_score"].iloc[0]
            assert actual == pytest.approx(exp), f"{pid} S1: expected {exp}, got {actual}"

    def test_session2_score_is_mean_likert(self, synthetic_project):
        """
        Carol Lee answered "Strongly Agree" (7) to all 13 questions.
        mean = 7.0
        """
        participants = _load_participants(synthetic_project)
        analysis_dir = synthetic_project / "analysis"
        before_csv = next(analysis_dir.glob("Questionnaire Before*"))
        after_csv = next(analysis_dir.glob("Questionnaire After*"))
        out = synthetic_project / "survey.csv"

        df = build_survey_csv(participants, before_csv, after_csv, out)
        carol_s2 = df[(df["participant_id"] == "P03") & (df["session"] == 2)]
        assert carol_s2["preparedness_score"].iloc[0] == pytest.approx(7.0)

    def test_sessions_present_for_all_participants(self, synthetic_project):
        participants = _load_participants(synthetic_project)
        analysis_dir = synthetic_project / "analysis"
        before_csv = next(analysis_dir.glob("Questionnaire Before*"))
        after_csv = next(analysis_dir.glob("Questionnaire After*"))
        out = synthetic_project / "survey.csv"

        df = build_survey_csv(participants, before_csv, after_csv, out)
        for pid in participants["participant_id"]:
            for sess in (1, 2):
                assert ((df["participant_id"] == pid) & (df["session"] == sess)).any(), (
                    f"Missing {pid} session {sess}"
                )


# ---------------------------------------------------------------------------
# Integration test — prepare_run and pipeline.prepare
# ---------------------------------------------------------------------------

class TestPrepareRun:
    def test_prepare_run_returns_correct_counts(self, synthetic_project):
        result = prepare_run(synthetic_project)
        n = len(pd.read_csv(synthetic_project / "analysis" / "participants.csv"))
        assert result["augmented_count"] == n * 2
        assert result["skipped"] == []
        assert result["survey_rows"] == n * 2

    def test_prepare_creates_survey_csv(self, synthetic_project):
        prepare_run(synthetic_project)
        assert (synthetic_project / "survey.csv").exists()

    def test_pipeline_prepare_equivalent_to_prepare_run(self, synthetic_project):
        r1 = prepare_run(synthetic_project)
        # Remove outputs so we can re-run via pipeline.prepare
        (synthetic_project / "survey.csv").unlink()
        r2 = prepare(synthetic_project)
        assert r1["augmented_count"] == r2["augmented_count"]
        assert r1["survey_rows"] == r2["survey_rows"]


# ---------------------------------------------------------------------------
# Integration test — full notebook execution
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestNotebookExecution:
    def test_notebook_runs_without_error(self, synthetic_project):
        """
        Runs the full analysis pipeline (data prep + notebook) against synthetic
        data and asserts that results_summary.csv is produced.
        """
        result = pipeline_run(
            root=synthetic_project,
            execute_notebook=True,
            notebook_output=synthetic_project / "analysis_executed.ipynb",
            timeout=300,
        )

        summary_csv = synthetic_project / "results_summary.csv"
        assert summary_csv.exists(), "results_summary.csv was not produced by the notebook"

        df = pd.read_csv(summary_csv)
        assert len(df) > 0, "results_summary.csv is empty"
        assert "DV" in df.columns
        assert "p" in df.columns
        assert "Significant" in df.columns

    def test_executed_notebook_saved(self, synthetic_project):
        nb_out = synthetic_project / "analysis_executed.ipynb"
        pipeline_run(
            root=synthetic_project,
            execute_notebook=True,
            notebook_output=nb_out,
            timeout=300,
        )
        assert nb_out.exists()

    def test_figures_produced(self, synthetic_project):
        pipeline_run(
            root=synthetic_project,
            execute_notebook=True,
            timeout=300,
        )
        assert (synthetic_project / "fig_line_plots.png").exists()
        assert (synthetic_project / "fig_box_plots.png").exists()
