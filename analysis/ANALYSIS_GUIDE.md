# Analysis Pipeline — Replication Guide

## Prerequisites

```bash
pip install pandas openpyxl scipy pingouin matplotlib seaborn nbformat nbconvert jupytext ipykernel
```

## Directory Layout

```
project-root/
├── analysis/
│   ├── participants.csv                                               # participant → session mapping
│   ├── Questionnaire Before Interacting with Adversarial Presentation Agent(1-30).xlsx
│   └── Questionnaire After Interacting with Adversarial Presentation Agent(1-30).xlsx
├── results/
│   └── {YYYY-MM-DD_HH-MM-SS}/
│       └── summary.csv                                               # one dir per session
├── survey.csv                                                        # perceived preparedness scores (auto-generated)
├── analysis_nb.py                                                    # source — edit this, not the notebook
└── analysis_executed.ipynb                                           # generated from analysis_nb.py
```

## Required Input Files

| File | Required columns |
|---|---|
| `analysis/participants.csv` | `participant_id`, `name`, `survey_name_before`, `survey_name_after`, `session_dir_1`, `session_dir_2`, `condition` |
| `results/{dir}/summary.csv` | `overall_score`, `contradictions_detected`, `memory_type` |
| `survey.csv` | `participant_id`, `session`, `preparedness_score` |

## Running the Pipeline

### Option A — Automated (generates `survey.csv` from xlsx, then executes notebook)

```bash
python analysis/pipeline.py
```

Data prep only (skip notebook execution):
```bash
python analysis/pipeline.py --no-notebook
```

### Option B — Manual steps

```bash
# 1. Regenerate notebook from source
jupytext --to notebook analysis_nb.py -o analysis_executed.ipynb

# 2. Execute
jupyter nbconvert --to notebook --execute analysis_executed.ipynb --inplace
```

## Perceived Preparedness Score Derivation (1–7 scale)

| Session | Source | Method |
|---|---|---|
| 1 (baseline) | Before-survey: preparedness question | Map label → 7-point scale (see table below) |
| 2 (post) | After-survey: all 13 Likert items | `score = mean(13 items)` — raw 1–7, no rescaling |

**Session 1 label → score mapping:**

| Label | Score |
|---|---|
| Very unprepared | 1 |
| Not prepared | 2 |
| Somewhat unprepared | 3 |
| Neither prepared nor unprepared | 4 |
| Somewhat prepared | 5 |
| Very prepared | 7 |

**Session 2 item coding:** Strongly Agree = 7, Agree = 6, Somewhat Agree = 5, Neutral = 4, Somewhat Disagree = 3, Disagree = 2, Strongly Disagree = 1

## Outputs

| File | Contents |
|---|---|
| `survey.csv` | `participant_id, session, preparedness_score` — overwritten by pipeline |
| `results_summary.csv` | All statistical test results (H1, H2, H3, correlation) |
| `fig_line_plots.png` | Mean ± SE line plots — composite score and perceived preparedness by condition/session |
| `fig_box_plots.png` | Box plots for all 3 DVs by condition/session |
| `fig_correlation.png` | Session-2 perception vs performance scatter + regression line |
| `analysis_executed.ipynb` | Notebook with all cell outputs |

## Adding a New Participant

1. Append a row to `analysis/participants.csv`
2. Drop the two session directories into `results/`
3. Re-run the pipeline

## Tests

```bash
pytest tests/analysis/                          # all tests including notebook execution
pytest tests/analysis/ -m "not integration"    # fast unit tests only
```
