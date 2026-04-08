"""
prepare_data.py — ETL to link experimental data sources for analysis.ipynb

Run from the project root:
    python analysis/prepare_data.py

What it does:
1. Reads analysis/participants.csv (canonical participant → session directory mapping)
2. Augments each results/{session_dir}/summary.csv with participant_id, session, and
   memory_type columns (writes in-place)
3. Builds survey.csv with participant_id, session, preparedness_score (range 1–7):
   - session 1 → before-interaction preparedness question mapped to 1–7
   - session 2 → mean of 13 after-interaction Likert items (each 1–7)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Encoding tables
# ---------------------------------------------------------------------------
LIKERT_7: dict[str, int] = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Somewhat Disagree": 3,
    "Neutral": 4,
    "Somewhat Agree": 5,
    "Agree": 6,
    "Strongly Agree": 7,
}

PREPAREDNESS_7: dict[str, int] = {
    "Not prepared": 1,
    "Very unprepared": 2,
    "Somewhat unprepared": 3,
    "Neither prepared nor unprepared": 4,
    "Somewhat prepared": 5,
    "Well prepared": 6,
    "Very prepared": 7,
}

# After-survey metadata columns that are NOT Likert items
_AFTER_META_COLS = {"ID", "Start time", "Completion time", "Email", "Name",
                    "Last modified time", "Name2"}


def preparedness_to_score(label: str) -> float:
    """Map a preparedness label directly onto the 1–7 scale."""
    val = PREPAREDNESS_7.get(str(label).strip())
    if val is None:
        raise ValueError(f"Unknown preparedness label: {label!r}. "
                         f"Expected one of: {list(PREPAREDNESS_7)}")
    return float(val)


def after_likert_cols(after_df: pd.DataFrame) -> list[str]:
    """Return the ordered list of Likert question columns from the after-survey."""
    return [c for c in after_df.columns if c not in _AFTER_META_COLS]


def augment_summary_csvs(
    participants: pd.DataFrame,
    results_dir: Path,
) -> tuple[int, list[str]]:
    """
    For each participant's session directories, inject participant_id, session,
    and (if absent) memory_type into the corresponding summary.csv in-place.

    Returns (augmented_count, list_of_skipped_descriptions).
    """
    augmented = 0
    skipped: list[str] = []

    for _, row in participants.iterrows():
        pid = str(row["participant_id"])
        condition = str(row["condition"])

        for session_num, dir_col in [(1, "session_dir_1"), (2, "session_dir_2")]:
            session_dir = str(row[dir_col]).strip()
            summary_path = results_dir / session_dir / "summary.csv"

            if not summary_path.exists():
                skipped.append(f"{pid} S{session_num}: {session_dir}")
                continue

            df = pd.read_csv(summary_path)

            # Inject / overwrite identification columns at the front
            df["participant_id"] = pid
            df["session"] = session_num

            # Normalise memory_type so CONDITION_MAP in the notebook resolves correctly
            if "memory_type" not in df.columns or df["memory_type"].isna().all():
                df["memory_type"] = "hybrid" if condition == "hybrid-memory" else "document_only"

            # Reorder: put participant_id and session first for readability
            leading = ["participant_id", "session"]
            other = [c for c in df.columns if c not in leading]
            df = df[leading + other]

            df.to_csv(summary_path, index=False)
            augmented += 1

    return augmented, skipped


def build_survey_csv(
    participants: pd.DataFrame,
    before_csv: Path,
    after_csv: Path,
    out: Path,
) -> pd.DataFrame:
    """
    Build survey.csv with columns: participant_id, session, preparedness_score.

    session 1 score  — preparedness question from before-survey, mapped to 1–7
    session 2 score  — mean of 13 Likert items from after-survey (each 1–7)

    Returns the DataFrame (also written to `out`).
    """
    _read = pd.read_excel if before_csv.suffix == ".xlsx" else pd.read_csv
    before_raw = _read(before_csv).rename(
        columns={"Column": "speaking_ability", "2": "preparedness"}
    )
    _read_after = pd.read_excel if after_csv.suffix == ".xlsx" else pd.read_csv
    after_raw = _read_after(after_csv)
    likert_cols = after_likert_cols(after_raw)

    rows: list[dict] = []
    warnings: list[str] = []

    for _, p in participants.iterrows():
        pid = str(p["participant_id"])
        name_before = str(p["survey_name_before"]).strip().lower()
        name_after = str(p["survey_name_after"]).strip().lower()

        # --- Session 1: baseline from before-survey preparedness ---
        before_match = before_raw[before_raw["Name2"].str.strip().str.lower() == name_before]
        if before_match.empty:
            warnings.append(f"No before-survey match for {pid} ({p['survey_name_before']!r})")
        else:
            prep_label = before_match.iloc[0]["preparedness"]
            rows.append({
                "participant_id": pid,
                "session": 1,
                "preparedness_score": preparedness_to_score(prep_label),
            })

        # --- Session 2: rescaled 13-item after-survey composite ---
        after_match = after_raw[after_raw["Name2"].str.strip().str.lower() == name_after]
        if after_match.empty:
            warnings.append(f"No after-survey match for {pid} ({p['survey_name_after']!r})")
        else:
            row_after = after_match.iloc[0]
            raw_scores: list[int] = []
            for col in likert_cols:
                val = str(row_after[col]).strip()
                encoded = LIKERT_7.get(val)
                if encoded is None:
                    warnings.append(
                        f"Unknown Likert value {val!r} in column {col!r} for {pid}"
                    )
                else:
                    raw_scores.append(encoded)

            if len(raw_scores) == len(likert_cols):
                rows.append({
                    "participant_id": pid,
                    "session": 2,
                    "preparedness_score": round(sum(raw_scores) / len(raw_scores), 3),
                })

    for w in warnings:
        print(f"  WARNING: {w}")

    survey_df = pd.DataFrame(rows, columns=["participant_id", "session", "preparedness_score"])
    survey_df.to_csv(out, index=False)
    return survey_df


def run(root: Path) -> dict:
    """
    Full ETL pipeline.

    Expects under `root`:
        analysis/participants.csv
        analysis/Questionnaire Before*.csv
        analysis/Questionnaire After*.csv
        results/{timestamp}/summary.csv  (one per session)

    Writes:
        results/{timestamp}/summary.csv  (augmented in-place)
        survey.csv
    """
    analysis_dir = root / "analysis"
    results_dir = root / "results"
    survey_out = root / "survey.csv"

    participants = pd.read_csv(analysis_dir / "participants.csv")
    print(f"Loaded {len(participants)} participants")

    before_csv = next(analysis_dir.glob("Questionnaire Before*"), None)
    if before_csv is None:
        raise FileNotFoundError(
            f"No 'Questionnaire Before*' CSV found in {analysis_dir}. "
            "Download the pre-interaction survey export and place it there."
        )
    after_csv = next(analysis_dir.glob("Questionnaire After*"), None)
    if after_csv is None:
        raise FileNotFoundError(
            f"No 'Questionnaire After*' CSV found in {analysis_dir}. "
            "Download the post-interaction survey export and place it there."
        )

    augmented, skipped = augment_summary_csvs(participants, results_dir)
    print(f"Augmented {augmented} summary CSV(s).")
    if skipped:
        print(f"Skipped {len(skipped)} missing session director(ies):")
        for s in skipped:
            print(f"  {s}")

    survey_df = build_survey_csv(participants, before_csv, after_csv, survey_out)
    print(f"Wrote {len(survey_df)} rows to {survey_out.relative_to(root)}")

    return {
        "augmented_count": augmented,
        "skipped": skipped,
        "survey_rows": len(survey_df),
        "survey_path": survey_out,
    }


if __name__ == "__main__":
    _root = Path(__file__).parent.parent
    run(_root)
