# Analysis Pipeline — Replication Guide

## Prerequisites

```bash
pip install pandas scipy pingouin matplotlib seaborn nbformat nbconvert jupytext
```

## Directory Layout

```
project-root/
├── analysis/
│   ├── participants.csv                          # participant → session mapping
│   ├── Questionnaire Before Interacting*.csv     # pre-interaction survey
│   └── Questionnaire After Interacting*.csv      # post-interaction survey
├── results/
│   └── {YYYY-MM-DD_HH-MM-SS}/
│       └── summary.csv                           # one dir per session
├── analysis.py                                   # source (edit this, not the notebook)
└── analysis.ipynb                                # generated from analysis.py
```

## Steps

### 1. Add a new participant

Edit `analysis/participants.csv` and add a row:

| participant_id | name | survey_name_before | survey_name_after | session_dir_1 | session_dir_2 | condition |
|---|---|---|---|---|---|---|
| P09 | Name | Full Name as in survey | Full Name as in survey | 2026-XX-XX_HH-MM-SS | 2026-XX-XX_HH-MM-SS | memory **or** no-memory |

### 2. Drop session directories into `results/`

Copy the timestamped session folders (containing `summary.csv` and `turns.csv`) into `results/`.

### 3. Run the pipeline

```bash
python analysis/pipeline.py
```

This does two things in sequence:
1. **Prepares data** — injects `participant_id`/`session` into each `results/*/summary.csv` and builds `survey.csv` from the questionnaires
2. **Executes the notebook** — runs `analysis.ipynb` and writes `results_summary.csv`, `fig_line_plots.png`, `fig_box_plots.png`, and `analysis_executed.ipynb`

To regenerate data only (skip notebook):
```bash
python analysis/pipeline.py --no-notebook
```

### 4. Regenerate the notebook after editing analysis logic

`analysis.py` is the source of truth. After editing it, regenerate the notebook:

```bash
jupytext --to notebook analysis.py -o analysis.ipynb
```

## Outputs

| File | Contents |
|---|---|
| `survey.csv` | `participant_id, session, confidence_score` — auto-generated, do not edit |
| `results_summary.csv` | ANOVA / non-parametric test results table |
| `fig_line_plots.png` | Mean ± SE line plots for both DVs |
| `fig_box_plots.png` | Distribution box plots by condition/session |
| `analysis_executed.ipynb` | Notebook with all cell outputs |

## Confidence Score Derivation

| Session | Source | Method |
|---|---|---|
| 1 (baseline) | Before-survey: preparedness question | Label → 1–5 scale × 3 → range [3, 15] |
| 2 (post) | After-survey: all 13 Likert items | Sum (range 13–91) rescaled linearly to [3, 15] |

## Tests

```bash
pytest tests/analysis/               # all 32 tests including notebook execution
pytest tests/analysis/ -m "not integration"   # fast unit tests only (~1s)
```
