# %% [markdown]
# # Adversarial Presentation Agent — Statistical Analysis
#
# **Design:** 2×2 mixed ANOVA
# - Between-subjects: `condition` (memory / no-memory)
# - Within-subjects: `session` (1, 2)
# - DVs: composite performance score (0–100) and perceived confidence score (3–15)
#
# **Data sources:**
# - `results/*/summary.csv` — one row per participant-session (exported by export.py)
# - `survey.csv` — flat CSV with `participant_id`, `session`, `confidence_score`
#
# **Conversion to Jupyter:** `jupytext --to notebook analysis.py`
#
# pip install pingouin scipy pandas numpy matplotlib seaborn jupytext

# %% [markdown]
# ## Section 0: Imports & Configuration

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Paths (edit as needed) ---
RESULTS_DIR = Path("results")      # directory containing timestamped run subdirs
SURVEY_CSV = Path("survey.csv")    # flat CSV: participant_id, session, confidence_score

# --- Study constants ---
ALPHA = 0.05
N_PARTICIPANTS = 30
N_SESSIONS = 2
N_PER_CONDITION = 15
N_COMPARISONS = N_SESSIONS          # Bonferroni denominator for post-hoc t-tests

ALPHA_BONFERRONI = ALPHA / N_COMPARISONS

CONDITION_MAP = {
    "hybrid": "memory",
    "document_only": "no-memory",
}

sns.set_theme(style="whitegrid", palette="Set2")

# %% [markdown]
# ## Section 1: Data Loading & Merging

# %%
def load_summary_csvs(results_dir: Path) -> pd.DataFrame:
    """Glob all summary.csv files under results_dir and concatenate."""
    paths = sorted(results_dir.glob("*/summary.csv"))
    if not paths:
        raise FileNotFoundError(f"No summary.csv files found under {results_dir.resolve()}")
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(paths)} summary CSV(s) → {len(df)} rows")
    return df


def validate_data(df: pd.DataFrame, label: str = "merged") -> None:
    """Print warnings for any data quality issues."""
    issues = []

    n_participants = df["participant_id"].nunique()
    if n_participants != N_PARTICIPANTS:
        issues.append(
            f"Expected {N_PARTICIPANTS} unique participants, found {n_participants}"
        )

    session_counts = df.groupby("participant_id")["session"].nunique()
    bad = session_counts[session_counts != N_SESSIONS]
    if not bad.empty:
        issues.append(
            f"{len(bad)} participant(s) don't have exactly {N_SESSIONS} sessions: "
            + ", ".join(bad.index.astype(str))
        )

    cond_counts = df.drop_duplicates("participant_id")["condition"].value_counts()
    for cond, expected in [("memory", N_PER_CONDITION), ("no-memory", N_PER_CONDITION)]:
        actual = cond_counts.get(cond, 0)
        if actual != expected:
            issues.append(f"Expected {expected} participants in '{cond}', found {actual}")

    if "confidence_score" in df.columns:
        bad_conf = df[~df["confidence_score"].between(3, 15)]
        if not bad_conf.empty:
            issues.append(
                f"{len(bad_conf)} confidence_score value(s) outside [3, 15]"
            )

    if issues:
        print(f"\n[DATA WARNINGS — {label}]")
        for issue in issues:
            print(f"  ⚠  {issue}")
    else:
        print(f"[{label}] All validation checks passed.")


# Load performance data
perf_df = load_summary_csvs(RESULTS_DIR)
perf_df["condition"] = perf_df["memory_type"].map(CONDITION_MAP)

# Load confidence survey data
if not SURVEY_CSV.exists():
    raise FileNotFoundError(
        f"Survey CSV not found at {SURVEY_CSV.resolve()}. "
        "Expected columns: participant_id, session, confidence_score"
    )
survey_df = pd.read_csv(SURVEY_CSV)
survey_df["session"] = survey_df["session"].astype(int)
print(f"Loaded survey CSV → {len(survey_df)} rows")

# Merge
df = perf_df.merge(
    survey_df[["participant_id", "session", "confidence_score"]],
    on=["participant_id", "session"],
    how="inner",
    validate="1:1",
)
df["session"] = df["session"].astype(int)
print(f"Merged dataset: {len(df)} rows, {df['participant_id'].nunique()} participants")

validate_data(df, label="merged dataset")

# %% [markdown]
# ## Section 2: Descriptive Statistics

# %%
DVS = {
    "overall_score": "Composite Performance Score (0–100)",
    "confidence_score": "Perceived Confidence Score (3–15)",
}

for col, label in DVS.items():
    print(f"\n{'='*60}")
    print(f"Descriptive statistics — {label}")
    print("="*60)
    desc = (
        df.groupby(["condition", "session"])[col]
        .agg(n="count", mean="mean", sd="std", se=lambda x: x.std() / np.sqrt(len(x)),
             min="min", max="max")
        .round(3)
    )
    print(desc.to_string())

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (col, label) in zip(axes, DVS.items()):
    summary = (
        df.groupby(["session", "condition"])[col]
        .agg(mean="mean", se=lambda x: x.std() / np.sqrt(len(x)))
        .reset_index()
    )
    for cond, grp in summary.groupby("condition"):
        ax.errorbar(
            grp["session"], grp["mean"], yerr=grp["se"],
            marker="o", capsize=4, label=cond,
        )
    ax.set_xticks([1, 2])
    ax.set_xlabel("Session")
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.legend(title="Condition")

plt.tight_layout()
plt.savefig("fig_line_plots.png", dpi=150)
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (col, label) in zip(axes, DVS.items()):
    plot_df = df.copy()
    plot_df["group"] = plot_df["condition"] + " / S" + plot_df["session"].astype(str)
    order = ["memory / S1", "memory / S2", "no-memory / S1", "no-memory / S2"]
    sns.boxplot(data=plot_df, x="group", y=col, order=order, ax=ax)
    ax.set_xlabel("Condition / Session")
    ax.set_ylabel(label)
    ax.set_title(f"Distribution — {label}")
    ax.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("fig_box_plots.png", dpi=150)
plt.show()

# %% [markdown]
# ## Section 3: Assumption Checks

# %%
def check_normality(df: pd.DataFrame, col: str, label: str) -> bool:
    """Shapiro-Wilk on each condition×session cell. Returns True if all cells pass."""
    print(f"\nShapiro-Wilk normality test — {label}")
    print(f"{'Cell':<25} {'W':>8} {'p':>10} {'Result'}")
    print("-" * 55)
    all_ok = True
    for (cond, sess), grp in df.groupby(["condition", "session"]):
        vals = grp[col].dropna().values
        if len(vals) < 3:
            print(f"  {cond} / S{sess:<18} {'—':>8} {'—':>10}  SKIP (n<3)")
            continue
        W, p = stats.shapiro(vals)
        ok = p > ALPHA
        if not ok:
            all_ok = False
        flag = "PASS" if ok else "FAIL ⚠"
        print(f"  {cond} / S{sess:<18} {W:>8.4f} {p:>10.4f}  {flag}")
    return all_ok


def check_levene(df: pd.DataFrame, col: str, label: str) -> bool:
    """Levene's test at each session. Returns True if all pass."""
    print(f"\nLevene's homogeneity of variance test — {label}")
    print(f"{'Session':<12} {'stat':>8} {'p':>10} {'Result'}")
    print("-" * 40)
    all_ok = True
    for sess in sorted(df["session"].unique()):
        groups = [
            grp[col].dropna().values
            for _, grp in df[df["session"] == sess].groupby("condition")
        ]
        stat, p = stats.levene(*groups)
        ok = p > ALPHA
        if not ok:
            all_ok = False
        flag = "PASS" if ok else "FAIL ⚠"
        print(f"  Session {sess:<7} {stat:>8.4f} {p:>10.4f}  {flag}")
    return all_ok


normality_ok = {}
levene_ok = {}

for col, label in DVS.items():
    normality_ok[col] = check_normality(df, col, label)
    levene_ok[col] = check_levene(df, col, label)

print("\n" + "="*60)
print("Sphericity")
print("="*60)
print(
    "Sphericity assumption is NOT applicable: the within-subjects factor\n"
    "(session) has only 2 levels. Greenhouse-Geisser correction is not needed."
)

# %% [markdown]
# ## Section 4: Analysis — Composite Performance Score

# %%
def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Pooled-SD Cohen's d."""
    nx, ny = len(x), len(y)
    pooled_sd = np.sqrt(((nx - 1) * x.std(ddof=1)**2 + (ny - 1) * y.std(ddof=1)**2) / (nx + ny - 2))
    return (x.mean() - y.mean()) / pooled_sd if pooled_sd > 0 else np.nan


def run_mixed_anova(df: pd.DataFrame, col: str, label: str) -> dict:
    """Run mixed ANOVA and return a dict of results for the summary table."""
    print(f"\n{'='*60}")
    print(f"Mixed ANOVA — {label}")
    print("="*60)

    results = []

    if normality_ok[col]:
        aov = pg.mixed_anova(
            data=df, dv=col, within="session", between="condition", subject="participant_id"
        )
        print(aov[["Source", "F", "DF1", "DF2", "p_unc", "np2"]].to_string(index=False))

        for _, row in aov.iterrows():
            source = row["Source"]
            if source == "Intercept":
                continue
            sig = "Yes" if row["p_unc"] < ALPHA else "No"
            print(
                f"\n  [{source}] F({int(row['DF1'])},{int(row['DF2'])}) = {row['F']:.3f}, "
                f"p = {row['p_unc']:.4f}, η²_p = {row['np2']:.3f} → Significant: {sig}"
            )
            results.append({
                "DV": label, "Effect": source,
                "F": round(row["F"], 3), "df": f"{int(row['DF1'])},{int(row['DF2'])}",
                "p": round(row["p_unc"], 4), "eta_p2": round(row["np2"], 3),
                "Significant": sig, "Test": "Mixed ANOVA",
            })

        # Post-hoc if interaction significant
        interaction_row = aov[aov["Source"] == "Interaction"]
        if not interaction_row.empty and interaction_row.iloc[0]["p_unc"] < ALPHA:
            print(f"\n  Interaction is significant → post-hoc t-tests at each session")
            print(f"  Bonferroni α = {ALPHA}/{N_COMPARISONS} = {ALPHA_BONFERRONI:.4f}")
            for sess in sorted(df["session"].unique()):
                sess_df = df[df["session"] == sess]
                mem = sess_df[sess_df["condition"] == "memory"][col].dropna().values
                nomem = sess_df[sess_df["condition"] == "no-memory"][col].dropna().values
                t, p_unc = stats.ttest_ind(mem, nomem)
                p_bonf = min(p_unc * N_COMPARISONS, 1.0)
                d = cohen_d(mem, nomem)
                sig = "Yes" if p_bonf < ALPHA else "No"
                print(
                    f"    Session {sess}: t = {t:.3f}, p = {p_unc:.4f}, "
                    f"p_bonf = {p_bonf:.4f}, d = {d:.3f} → Significant: {sig}"
                )
                results.append({
                    "DV": label, "Effect": f"Post-hoc Session {sess}",
                    "F": np.nan, "df": f"{len(mem)+len(nomem)-2}",
                    "p": round(p_bonf, 4), "eta_p2": np.nan,
                    "d": round(d, 3), "Significant": sig, "Test": "t-test (Bonferroni)",
                })
        else:
            print("\n  Interaction not significant — no post-hoc tests required.")

    else:
        print(
            f"\n  Normality assumption failed for {label}. "
            "Using non-parametric alternatives."
        )
        print(
            "  Note: Friedman test applies for ≥3 within-subjects levels; "
            "Wilcoxon signed-rank is used here (2 sessions)."
        )

        # Between-group: Mann-Whitney U at each session
        print("\n  Mann-Whitney U (between conditions, per session):")
        for sess in sorted(df["session"].unique()):
            sess_df = df[df["session"] == sess]
            mem = sess_df[sess_df["condition"] == "memory"][col].dropna().values
            nomem = sess_df[sess_df["condition"] == "no-memory"][col].dropna().values
            U, p = stats.mannwhitneyu(mem, nomem, alternative="two-sided")
            sig = "Yes" if p < ALPHA else "No"
            print(f"    Session {sess}: U = {U:.1f}, p = {p:.4f} → Significant: {sig}")
            results.append({
                "DV": label, "Effect": f"Between (Session {sess})",
                "F": np.nan, "df": np.nan, "p": round(p, 4), "eta_p2": np.nan,
                "Significant": sig, "Test": "Mann-Whitney U",
            })

        # Within-group: Wilcoxon signed-rank per condition
        print("\n  Wilcoxon signed-rank (within condition across sessions):")
        for cond in ["memory", "no-memory"]:
            cond_df = df[df["condition"] == cond].sort_values(["participant_id", "session"])
            s1 = cond_df[cond_df["session"] == 1].set_index("participant_id")[col]
            s2 = cond_df[cond_df["session"] == 2].set_index("participant_id")[col]
            common = s1.index.intersection(s2.index)
            stat, p = stats.wilcoxon(s1[common].values, s2[common].values)
            sig = "Yes" if p < ALPHA else "No"
            print(f"    {cond}: W = {stat:.1f}, p = {p:.4f} → Significant: {sig}")
            results.append({
                "DV": label, "Effect": f"Within '{cond}'",
                "F": np.nan, "df": np.nan, "p": round(p, 4), "eta_p2": np.nan,
                "Significant": sig, "Test": "Wilcoxon signed-rank",
            })

    return results


perf_results = run_mixed_anova(df, "overall_score", DVS["overall_score"])

# %% [markdown]
# ## Section 5: Analysis — Perceived Confidence Score

# %%
conf_results = run_mixed_anova(df, "confidence_score", DVS["confidence_score"])

# %% [markdown]
# ## Section 6: Results Summary Table

# %%
all_results = perf_results + conf_results
summary_table = pd.DataFrame(all_results)

# Reorder columns sensibly
col_order = ["DV", "Effect", "Test", "df", "F", "p", "eta_p2", "d", "Significant"]
col_order = [c for c in col_order if c in summary_table.columns]
summary_table = summary_table[col_order]

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(summary_table.fillna("—").to_string(index=False))

# Save to CSV for reporting
summary_table.to_csv("results_summary.csv", index=False)
print("\nResults saved to results_summary.csv")
