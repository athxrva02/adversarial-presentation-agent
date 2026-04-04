# Statistical Analysis — Adversarial Presentation Agent

## Study Design

**Type:** 2 × 2 mixed design  
**Between-subjects:** `condition` — `memory` (hybrid conversational memory) vs `no-memory` (session-only context)  
**Within-subjects:** `session` — Session 1 (baseline) vs Session 2 (post-practice)  
**N:** 30 participants, 15 per condition  
**Note:** Pre-registered design specified 3 sessions; only 2 were completed (noted as a limitation).

---

## Hypotheses

| # | Hypothesis | DV |
|---|---|---|
| H1 | Memory-enabled participants report higher perceived preparedness after repeated practice | Perceived Preparedness Score (1–7) |
| H2 | Memory-enabled participants produce fewer contradictions | Contradictions Detected |
| H3 | Memory-enabled agent perceived as more helpful and asking more relevant questions | Perceived Preparedness Score at Session 2, 1–7 (between conditions) |

---

## Metrics & How They Are Computed

### 1. Composite Performance Score (0–100)
Automated rubric score exported per session by `export.py`. Combines:
- Rubric sub-scores (clarity/structure, evidence specificity, definition precision, logical coherence, handling adversarial questions, depth of understanding, concession & qualification, recovery from challenge) — each scored 1–5
- Voice delivery score (where available) derived from pace, pause rate, pitch variance, volume, silence ratio

Source: `results/<session_dir>/summary.csv` → column `overall_score`

### 2. Perceived Preparedness Score (1–7)
Derived from questionnaire responses on a 7-point Likert scale. Measures perceived preparedness and agent perception.

**Session 1 score** — mapped from the Before-interaction questionnaire "preparedness level" item onto the 7-point scale:

| Response | Score |
|---|---|
| Very unprepared | 1 |
| Not prepared | 2 |
| Somewhat unprepared | 3 |
| Neither prepared nor unprepared | 4 |
| Somewhat prepared | 5 |
| Very prepared | 7 |

**Session 2 score** — mean of the 13-item After-interaction Likert questionnaire (raw, no rescaling):

```
score = mean_of_13_items
```

where each item is coded: Strongly Agree = 7, Agree = 6, Somewhat Agree = 5, Neutral = 4, Somewhat Disagree = 3, Disagree = 2, Strongly Disagree = 1.

Both session scores are on the same 1–7 scale, allowing direct within-subjects comparison.

Source: `survey.csv` → column `preparedness_score`

### 3. Contradictions Detected (count)
Number of contradictions flagged by the agent during a session (user's claims that conflict with prior statements or the slide content).

Source: `results/<session_dir>/summary.csv` → column `contradictions_detected`

---

## Assumption Checks (run per DV)

| Check | Test | Action if failed |
|---|---|---|
| Normality | Shapiro-Wilk per condition × session cell | Switch to non-parametric tests |
| Homogeneity of variance | Levene's test per session | Noted; non-parametric path tolerates this |
| Sphericity | Mauchly's test | N/A — only 2 within-subjects levels |

---

## Statistical Tests

### Parametric path (if all normality checks pass)
- **Mixed ANOVA** (`pingouin.mixed_anova`): condition (between) × session (within)
  - Reports: main effect of condition, main effect of session, condition × session interaction
  - Effect size: partial η² (η²_p)
- **Post-hoc** (only if interaction is significant): independent-samples t-test at each session with Bonferroni correction (α / N_sessions)
  - Effect size: Cohen's d

### Non-parametric path (if any normality check fails)
- **Mann-Whitney U** (`scipy.stats.mannwhitneyu`, two-sided): between conditions at each session separately → addresses H2 (session-by-session) and H3 (session 2)
- **Wilcoxon signed-rank** (`scipy.stats.wilcoxon`): within each condition across sessions → addresses H1 and H2 within-group trend
  - If all paired differences are zero, the test is skipped and p = 1.0 is reported

### Correlation
- **Spearman's ρ** (`scipy.stats.spearmanr`): between session-2 perceived preparedness score and session-2 composite performance score (N = 30)
  - Tests whether participants who perceived the agent more positively also performed better

**Significance threshold:** α = 0.05 throughout. Bonferroni-corrected α = 0.025 for post-hoc pairwise tests.

---

## Results Summary

| DV | Effect | Test | p | Significant |
|---|---|---|---|---|
| Composite Performance Score | Between conditions — Session 1 | Mann-Whitney U | 0.709 | No |
| Composite Performance Score | Between conditions — Session 2 | Mann-Whitney U | 0.561 | No |
| Composite Performance Score | Within memory (S1→S2) | Wilcoxon | 0.083 | No |
| Composite Performance Score | Within no-memory (S1→S2) | Wilcoxon | 0.208 | No |
| **H2** Contradictions | Between conditions — Session 1 | Mann-Whitney U | 0.164 | No |
| **H2** Contradictions | Between conditions — Session 2 | Mann-Whitney U | **0.0015** | **Yes** |
| **H2** Contradictions | Within memory (S1→S2) | Wilcoxon | **0.030** | **Yes** |
| **H2** Contradictions | Within no-memory (S1→S2) | Wilcoxon | 1.0 (all zero) | No |
| **H1** Preparedness | Between conditions — Session 1 | Mann-Whitney U | 0.483 | No |
| **H3** Preparedness | Between conditions — Session 2 | Mann-Whitney U | 0.129 | No |
| **H1** Preparedness | Within memory (S1→S2) | Wilcoxon | **0.0001** | **Yes** |
| **H1** Preparedness | Within no-memory (S1→S2) | Wilcoxon | **0.0019** | **Yes** |
| **Corr.** Perception × Performance | Spearman ρ = 0.444 | Spearman | **0.014** | **Yes** |

### Interpretation
- **H1 supported:** Perceived preparedness (1–7) increased significantly from session 1 to session 2 in both conditions (memory p = 0.0001, no-memory p = 0.0019). The improvement is not unique to the memory condition.
- **H2 partially supported:** Memory-enabled participants had significantly fewer contradictions at session 2 (between-conditions, p = 0.0015) and reduced contradictions within the memory group over sessions (p = 0.030). No-memory participants showed no within-session change.
- **H3 not supported:** No significant difference in overall agent perception between conditions at session 2 (p = 0.129).
- **Correlation significant:** Agent perception score (session 2, 1–7) positively correlates with composite performance (ρ = 0.44, p = 0.014) — participants who rated the agent more positively also performed better.

---

## Output Files

| File | Description |
|---|---|
| `results_summary.csv` | Full results table for all tests |
| `fig_line_plots.png` | Mean ± SE line plots for composite score and perceived preparedness by condition/session |
| `fig_box_plots.png` | Distribution box plots for all three DVs |
| `fig_correlation.png` | Scatter plot of session-2 perception vs performance with regression line |
| `analysis_nb.py` | Source notebook (convert via `jupytext --to notebook analysis_nb.py`) |
| `analysis_executed.ipynb` | Executed notebook with all outputs |
