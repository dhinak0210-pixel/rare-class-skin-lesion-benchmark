"""
src/stats.py
============
Statistical aggregation and paired hypothesis testing:
  - Mean, std, and 95% bootstrap confidence intervals across 15 evaluations (5 folds x 3 seeds).
  - Paired Wilcoxon signed-rank tests and paired bootstrap difference CIs.
  - Generates LaTeX and Markdown result tables.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats

METRIC_COLUMNS = [
    "test_minority_recall",
    "test_minority_f1",
    "test_balanced_accuracy",
    "test_macro_f1",
    "test_auroc_macro",
    "test_minority_auroc",
    "test_auprc_macro",
    "test_minority_auprc",
    "test_accuracy",
    "test_ece_calibrated"
]

def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float]:
    """Computes non-parametric empirical bootstrap confidence interval for mean."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or np.all(values == values[0]):
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    boot_means = []
    n = len(values)
    for _ in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(np.mean(sample))

    alpha = 1.0 - confidence_level
    low = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return low, high


def compute_paired_comparison(
    baseline_values: np.ndarray,
    candidate_values: np.ndarray,
    metric_name: str = "minority_recall",
    n_resamples: int = 2000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes paired difference statistics:
      - Delta mean = mean(candidate - baseline)
      - 95% bootstrap confidence interval of delta
      - Wilcoxon signed-rank test p-value
      - Statistical significance decision (p < 0.05 and CI excludes 0)
    """
    diffs = np.asarray(candidate_values, dtype=np.float64) - np.asarray(baseline_values, dtype=np.float64)
    mean_diff = float(np.mean(diffs))
    low_ci, high_ci = bootstrap_ci(diffs, n_resamples=n_resamples, seed=seed)

    # Wilcoxon signed-rank test
    try:
        if np.all(diffs == 0):
            w_stat, p_val = 0.0, 1.0
        else:
            w_res = stats.wilcoxon(candidate_values, baseline_values, alternative="two-sided")
            w_stat = float(w_res.statistic)
            p_val = float(w_res.pvalue)
    except Exception:
        w_stat, p_val = float("nan"), float("nan")

    is_significant = (p_val < 0.05) and (low_ci > 0 or high_ci < 0)

    return {
        "metric": metric_name,
        "delta_mean": mean_diff,
        "delta_ci_95": (low_ci, high_ci),
        "wilcoxon_stat": w_stat,
        "p_value": p_val,
        "is_significant": is_significant,
        "conclusion": (
            f"Statistically significant improvement (p={p_val:.4f})"
            if (is_significant and mean_diff > 0)
            else (f"Statistically significant degradation (p={p_val:.4f})"
                  if (is_significant and mean_diff < 0)
                  else f"No statistically significant difference (p={p_val:.4f}, 95% CI [{low_ci:.4f}, {high_ci:.4f}])")
        )
    }


def aggregate_benchmark_runs(
    runs_csv: str = "results/logs/runs.csv",
    output_tables_dir: str = "results/tables"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads raw run logs across 15 evaluations per arm and generates:
      1. Aggregated summary table (mean +- std, 95% CI)
      2. Paired statistical test matrix against B0 baseline
    """
    if not os.path.exists(runs_csv):
        raise FileNotFoundError(f"Runs log not found at: {runs_csv}")

    df_runs = pd.read_csv(runs_csv)
    os.makedirs(output_tables_dir, exist_ok=True)

    summary_rows = []
    arms = df_runs["arm_id"].unique()

    for arm in arms:
        arm_df = df_runs[df_runs["arm_id"] == arm]
        n_evals = len(arm_df)

        row = {"arm_id": arm, "n_evals": n_evals}

        for metric in METRIC_COLUMNS:
            if metric in arm_df.columns:
                vals = arm_df[metric].dropna().values
                if len(vals) > 0:
                    mean_val = float(np.mean(vals))
                    std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    ci_low, ci_high = bootstrap_ci(vals)
                    row[f"{metric}_mean"] = mean_val
                    row[f"{metric}_std"] = std_val
                    row[f"{metric}_ci_low"] = ci_low
                    row[f"{metric}_ci_high"] = ci_high
                    row[f"{metric}_formatted"] = f"{mean_val:.3f} +- {std_val:.3f} [{ci_low:.3f}, {ci_high:.3f}]"

        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_tables_dir, "aggregated_benchmark_summary.csv")
    df_summary.to_csv(summary_csv, index=False)

    # Generate Markdown Table
    md_lines = [
        "# Rare-Class Benchmark: Aggregated Results across 15 Evaluations\n",
        "| Arm ID | N | Minority Recall (df) | Balanced Accuracy | Macro F1 | Minority AUPRC | ECE (Calibrated) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for _, r in df_summary.iterrows():
        min_rec = r.get("test_minority_recall_formatted", "N/A")
        bal_acc = r.get("test_balanced_accuracy_formatted", "N/A")
        mac_f1 = r.get("test_macro_f1_formatted", "N/A")
        min_pr = r.get("test_minority_auprc_formatted", "N/A")
        ece = r.get("test_ece_calibrated_formatted", "N/A")
        md_lines.append(f"| **{r['arm_id']}** | {r['n_evals']} | {min_rec} | {bal_acc} | {mac_f1} | {min_pr} | {ece} |")

    summary_md_path = os.path.join(output_tables_dir, "benchmark_table.md")
    with open(summary_md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    return df_summary, df_runs
