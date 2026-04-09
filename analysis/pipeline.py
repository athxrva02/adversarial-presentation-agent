"""
pipeline.py — end-to-end analysis pipeline

Prepares data then converts analysis_nb.py → notebook via jupytext and executes it.

Usage:
    python analysis/pipeline.py              # full pipeline
    python analysis/pipeline.py --no-notebook  # data prep only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `analysis.*` imports work whether
# this script is invoked as `python analysis/pipeline.py` or as a module.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import jupytext
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from analysis.prepare_data import run as prepare_data


def prepare(root: Path) -> dict:
    """Run the data-preparation step and return its result dict."""
    return prepare_data(root)


def run_notebook(
    root: Path,
    *,
    output_path: Path | None = None,
    timeout: int = 600,
) -> nbformat.NotebookNode:
    """
    Convert analysis_nb.py → notebook via jupytext, execute it, and return the result.

    The notebook's relative file paths (results/, survey.csv) are resolved
    against `root` because the kernel is launched with that as its working
    directory.

    Parameters
    ----------
    root:
        Project root — must contain analysis_nb.py, results/, and survey.csv.
    output_path:
        If given, write the executed notebook here (e.g. analysis_executed.ipynb).
    timeout:
        Per-cell execution timeout in seconds.

    Returns
    -------
    The executed NotebookNode (in-memory).
    """
    nb_src = root / "analysis_nb.py"
    if not nb_src.exists():
        raise FileNotFoundError(f"Source not found: {nb_src}")

    nb = jupytext.read(nb_src)

    # Force non-interactive matplotlib backend so plt.show() is a no-op
    prev_backend = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = "Agg"

    try:
        ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": str(root)}})
    finally:
        if prev_backend is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = prev_backend

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            nbformat.write(nb, f)
        print(f"Executed notebook saved → {output_path}")

    return nb


def run(
    root: Path | None = None,
    *,
    execute_notebook: bool = True,
    notebook_output: Path | None = None,
    timeout: int = 600,
) -> dict:
    """
    Full pipeline: prepare data, then (optionally) execute the notebook.

    Parameters
    ----------
    root:
        Project root.  Defaults to the parent of this file's parent directory.
    execute_notebook:
        Set False to skip notebook execution (data prep only).
    notebook_output:
        Path to write the executed notebook.  Defaults to
        ``root/analysis_executed.ipynb``.
    timeout:
        Per-cell notebook execution timeout in seconds.

    Returns
    -------
    dict with keys: augmented_count, skipped, survey_rows, survey_path,
    and (if notebook ran) notebook_path, results_summary_path.
    """
    if root is None:
        root = Path(__file__).parent.parent

    print("=" * 60)
    print("Step 1/2 — Data preparation")
    print("=" * 60)
    result = prepare(root)

    if not execute_notebook:
        return result

    print("\n" + "=" * 60)
    print("Step 2/2 — Notebook execution")
    print("=" * 60)

    if notebook_output is None:
        notebook_output = root / "analysis_executed.ipynb"

    run_notebook(root, output_path=notebook_output, timeout=timeout)

    result["notebook_path"] = notebook_output
    result["results_summary_path"] = root / "results_summary.csv"

    print("\nPipeline complete.")
    if result["results_summary_path"].exists():
        print(f"Results summary → {result['results_summary_path']}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full analysis pipeline.")
    parser.add_argument(
        "--no-notebook",
        action="store_true",
        help="Skip notebook execution (data prep only).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-cell notebook execution timeout in seconds (default: 600).",
    )
    args = parser.parse_args()

    run(execute_notebook=not args.no_notebook, timeout=args.timeout)
