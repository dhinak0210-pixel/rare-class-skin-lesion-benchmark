"""Benefit predictor, cross-modality transfer evaluation, and publication figures/tables.

Answers:
- RQ1: How does generative augmentation compare against count-matched classical controls across scarcity levels N in [10, 20, 50, 100]?
- RQ2: Do cheap pre-training/quality signals (FID, KID, Diversity, Copying, Confidence) correlate with downstream gain Delta F1?
- RQ3: Can an interpretable Ridge / shallow Decision Tree predict Delta F1 and zero-shot transfer across distinct medical modalities?
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree

from src.config import ExperimentConfig, cfg, ensure_dirs


def load_and_prepare_deltas(csv_path: str = "results/runs.csv") -> pd.DataFrame:
    """Compute Delta F1 (val and test) relative to method='none' for each (dataset, N, seed)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found at {csv_path}. Run experiments first.")

    df = pd.read_csv(csv_path)

    # Exclude full_data baseline for delta calculations
    scarcity_df = df[df["N"] > 0].copy()

    # Separate baseline (method == 'none')
    none_df = scarcity_df[scarcity_df["method"] == "none"].set_index(["dataset", "N", "seed"])

    rows = []
    aug_df = scarcity_df[scarcity_df["method"].isin(["classical_k", "gan_k"])].copy()

    for _, row in aug_df.iterrows():
        key = (row["dataset"], row["N"], row["seed"])
        if key in none_df.index:
            base_row = none_df.loc[key]
            if isinstance(base_row, pd.DataFrame):
                base_row = base_row.iloc[0]

            delta_val_f1 = row["val_rare_f1"] - base_row["val_rare_f1"]
            delta_val_pr_auc = row["val_pr_auc"] - base_row["val_pr_auc"]
            delta_test_f1 = row["test_rare_f1"] - base_row["test_rare_f1"]
            delta_test_pr_auc = row["test_pr_auc"] - base_row["test_pr_auc"]

            r_dict = row.to_dict()
            r_dict["delta_val_f1"] = float(delta_val_f1)
            r_dict["delta_val_pr_auc"] = float(delta_val_pr_auc)
            r_dict["delta_test_f1"] = float(delta_test_f1)
            r_dict["delta_test_pr_auc"] = float(delta_test_pr_auc)
            r_dict["base_val_rare_f1"] = float(base_row["val_rare_f1"])
            r_dict["base_val_pr_auc"] = float(base_row["val_pr_auc"])
            r_dict["log10_N"] = float(np.log10(row["N"]))
            rows.append(r_dict)

    delta_df = pd.DataFrame(rows)
    return delta_df


# ---------------------------------------------------------------------------
# RQ1: Gain vs. Scarcity (N) Summary Table & Figure 1
# ---------------------------------------------------------------------------
def generate_rq1_analysis(runs_df: pd.DataFrame, delta_df: pd.DataFrame) -> str:
    """Generate LaTeX summary table and Figure 1 (Gain vs N)."""
    ensure_dirs()
    tables_dir = Path(cfg.tables_dir)
    figures_dir = Path(cfg.figures_dir)

    latex_table_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Empirical evaluation across scarcity levels $N \in \{10, 20, 50, 100\}$ on DermaMNIST and BloodMNIST. We report mean $\pm$ std over 3 independent seeds. Count-matched classical control matches sample budget $K=200$.}",
        r"\label{tab:rq1_main_results}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Method} & \textbf{N=10} & \textbf{N=20} & \textbf{N=50} & \textbf{N=100} \\",
        r"\midrule",
    ]

    datasets = sorted(runs_df["dataset"].unique())
    for dset in datasets:
        latex_table_lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{dset.upper()}}}}} \\\\")
        for method, m_label in [("none", "None (Baseline)"), ("classical_k", "Classical ($K=200$)"), ("gan_k", "DCGAN ($K=200$)")]:
            vals = []
            for N in [10, 20, 50, 100]:
                sub = runs_df[(runs_df["dataset"] == dset) & (runs_df["method"] == method) & (runs_df["N"] == N)]
                if len(sub) > 0:
                    m_f1 = sub["val_rare_f1"].mean()
                    s_f1 = sub["val_rare_f1"].std()
                    s_str = f"{s_f1:.3f}" if not np.isnan(s_f1) else "0.000"
                    vals.append(f"${m_f1:.3f} \\pm {s_str}$")
                else:
                    vals.append("--")
            latex_table_lines.append(f" & {m_label} & " + " & ".join(vals) + r" \\")
        latex_table_lines.append(r"\midrule")

    latex_table_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    table_content = "\n".join(latex_table_lines)
    tex_path = tables_dir / "rq1_main_results.tex"
    with open(tex_path, "w") as f:
        f.write(table_content)

    # Plot Figure 1: Line plot of val_rare_f1 vs N
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5), dpi=300, sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    colors = {"none": "#7f7f7f", "classical_k": "#1f77b4", "gan_k": "#d62728"}
    markers = {"none": "o", "classical_k": "s", "gan_k": "^"}
    labels = {"none": "None (Baseline)", "classical_k": "Count-Matched Classical", "gan_k": "DCGAN+DiffAugment"}

    for idx, dset in enumerate(datasets):
        ax = axes[idx]
        for m in ["none", "classical_k", "gan_k"]:
            m_means = []
            m_stds = []
            ns = [10, 20, 50, 100]
            for n_val in ns:
                sub = runs_df[(runs_df["dataset"] == dset) & (runs_df["method"] == m) & (runs_df["N"] == n_val)]
                if len(sub) > 0:
                    m_means.append(sub["val_rare_f1"].mean())
                    m_stds.append(sub["val_rare_f1"].std() if len(sub) > 1 else 0.0)
                else:
                    m_means.append(np.nan)
                    m_stds.append(0.0)

            m_means = np.array(m_means)
            m_stds = np.array(m_stds)

            ax.plot(ns, m_means, marker=markers[m], color=colors[m], label=labels[m], linewidth=2.2, markersize=7)
            ax.fill_between(ns, m_means - m_stds, m_means + m_stds, color=colors[m], alpha=0.18)

        ax.set_title(f"Rare-Class F1 vs. Scarcity — {dset.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of Real Rare Training Samples ($N$)", fontsize=11, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Validation Rare-Class F1", fontsize=11, fontweight="bold")
        ax.set_xticks(ns)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    fig1_path = figures_dir / "fig1_gain_vs_N.png"
    plt.savefig(fig1_path, dpi=300)
    plt.savefig(str(fig1_path).replace(".png", ".pdf"), dpi=300)
    plt.close()

    print(f"RQ1 Summary Table saved: {tex_path}")
    print(f"RQ1 Figure 1 saved: {fig1_path}")
    return table_content


# ---------------------------------------------------------------------------
# RQ2: Cheap Signals vs. Utility Correlation Analysis & Figure 2
# ---------------------------------------------------------------------------
def generate_rq2_analysis(delta_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute Spearman rho, Pearson r, and produce Figure 2."""
    ensure_dirs()
    figures_dir = Path(cfg.figures_dir)

    signals = [
        ("signal_kid", "Kernel Inception Dist (KID)", True),
        ("signal_diversity", "Diversity (Cosine Dist)", False),
        ("signal_nn_copy_dist", "NN Copying Distance", False),
        ("signal_clf_conf", "Classifier Confidence", False),
    ]

    corr_results = {}
    valid_df = delta_df.dropna(subset=["delta_val_f1"]).copy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=300)
    axes = axes.flatten()

    for idx, (sig_col, sig_title, lower_is_better) in enumerate(signals):
        ax = axes[idx]
        sig_data = valid_df.dropna(subset=[sig_col])

        if len(sig_data) >= 3:
            x = sig_data[sig_col].values
            y = sig_data["delta_val_f1"].values

            s_rho, s_pval = spearmanr(x, y)
            p_r, p_pval = pearsonr(x, y)

            corr_results[sig_col] = {
                "spearman_rho": float(s_rho),
                "spearman_pval": float(s_pval),
                "pearson_r": float(p_r),
                "pearson_pval": float(p_pval),
            }

            # Scatter points styled by N and method
            scatter = ax.scatter(
                x, y,
                c=sig_data["N"],
                cmap="viridis",
                s=70,
                alpha=0.85,
                edgecolors="black",
                linewidth=0.8,
            )

            # Fit trendline
            if len(np.unique(x)) > 1:
                poly = np.polyfit(x, y, deg=1)
                x_line = np.linspace(np.min(x), np.max(x), 50)
                y_line = np.polyval(poly, x_line)
                ax.plot(x_line, y_line, color="#d62728", linestyle="--", linewidth=1.8, label=f"Trend ($\\rho={s_rho:.2f}$)")

            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Scarcity $N$", fontsize=9)
            ax.legend(frameon=True, fontsize=9, loc="upper right" if lower_is_better else "lower right")
        else:
            corr_results[sig_col] = {
                "spearman_rho": 0.0,
                "spearman_pval": 1.0,
                "pearson_r": 0.0,
                "pearson_pval": 1.0,
            }

        ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_title(f"{sig_title} vs. $\\Delta$ Rare F1", fontsize=11, fontweight="bold")
        ax.set_xlabel(sig_title, fontsize=10)
        ax.set_ylabel("$\\Delta$ Validation Rare F1", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig2_path = figures_dir / "fig2_signals_vs_gain.png"
    plt.savefig(fig2_path, dpi=300)
    plt.savefig(str(fig2_path).replace(".png", ".pdf"), dpi=300)
    plt.close()

    print(f"RQ2 Figure 2 saved: {fig2_path}")
    print("Signal Correlation with Delta F1:")
    for sig, metrics in corr_results.items():
        print(f"  {sig:20s}: Spearman rho = {metrics['spearman_rho']:.3f} (p={metrics['spearman_pval']:.3e}) | Pearson r = {metrics['pearson_r']:.3f}")

    return corr_results


# ---------------------------------------------------------------------------
# RQ3: Benefit Predictor & Cross-Modality Transfer Evaluation & Figure 3
# ---------------------------------------------------------------------------
def generate_rq3_analysis(delta_df: pd.DataFrame) -> Dict[str, object]:
    """Train Ridge and shallow Decision Tree, evaluate cross-dataset transfer, and plot Figure 3."""
    ensure_dirs()
    figures_dir = Path(cfg.figures_dir)

    feature_cols = [
        "log10_N",
        "signal_kid",
        "signal_diversity",
        "signal_nn_copy_dist",
        "signal_clf_conf",
    ]

    valid_df = delta_df.dropna(subset=feature_cols + ["delta_val_f1"]).copy()
    datasets = list(valid_df["dataset"].unique())

    dset_train_name = "dermamnist" if "dermamnist" in datasets else datasets[0]
    train_subset = valid_df[valid_df["dataset"] == dset_train_name].copy()
    test_subset = valid_df[valid_df["dataset"] != dset_train_name].copy()

    X_train = train_subset[feature_cols].values
    y_train = train_subset["delta_val_f1"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 1. Fit Ridge Regressor
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)

    # 2. Fit Interpretable Decision Tree (max_depth=2)
    dt = DecisionTreeRegressor(max_depth=2, random_state=42)
    dt.fit(X_train, y_train)  # Unscaled for intuitive human readability in decision rules

    results = {
        "ridge_coefs": dict(zip(feature_cols, ridge.coef_.tolist())),
        "ridge_intercept": float(ridge.intercept_),
    }

    # Cross-dataset transfer test if second dataset exists
    if len(test_subset) > 0:
        X_test = test_subset[feature_cols].values
        y_test = test_subset["delta_val_f1"].values
        X_test_scaled = scaler.transform(X_test)

        preds_ridge = ridge.predict(X_test_scaled)
        preds_dt = dt.predict(X_test)

        # Decision Accuracy: % of correct decisions whether to turn augmentation ON (gain > 0) or OFF (gain <= 0)
        dec_acc_ridge = np.mean((preds_ridge > 0) == (y_test > 0)) * 100.0
        dec_acc_dt = np.mean((preds_dt > 0) == (y_test > 0)) * 100.0

        mae_ridge = mean_absolute_error(y_test, preds_ridge)
        mae_dt = mean_absolute_error(y_test, preds_dt)

        r2_ridge = r2_score(y_test, preds_ridge)
        r2_dt = r2_score(y_test, preds_dt)

        results["transfer"] = {
            "source_dataset": dset_train_name,
            "target_dataset": test_subset["dataset"].iloc[0],
            "ridge_mae": float(mae_ridge),
            "ridge_r2": float(r2_ridge),
            "ridge_decision_acc": float(dec_acc_ridge),
            "dt_mae": float(mae_dt),
            "dt_r2": float(r2_dt),
            "dt_decision_acc": float(dec_acc_dt),
        }
    else:
        # In-sample validation metrics when only 1 dataset is currently executed
        preds_dt = dt.predict(X_train)
        dec_acc_dt = np.mean((preds_dt > 0) == (y_train > 0)) * 100.0
        results["in_sample"] = {
            "dataset": dset_train_name,
            "dt_decision_acc": float(dec_acc_dt),
        }

    # Plot Figure 3: Decision tree rule and predicted vs actual scatter plot
    fig = plt.figure(figsize=(12, 5), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0])

    ax1 = fig.add_subplot(gs[0])
    plot_tree(
        dt,
        feature_names=feature_cols,
        filled=True,
        rounded=True,
        ax=ax1,
        fontsize=9,
        precision=3,
    )
    ax1.set_title("Interpretable Decision Rule (Depth-2 Tree)", fontsize=11, fontweight="bold")

    ax2 = fig.add_subplot(gs[1])
    if len(test_subset) > 0:
        ax2.scatter(
            y_test, preds_dt,
            c="#1f77b4", s=60, alpha=0.9, edgecolors="black", label="Transfer (BloodMNIST)"
        )
        lims = [min(np.min(y_test), np.min(preds_dt)) - 0.05, max(np.max(y_test), np.max(preds_dt)) + 0.05]
    else:
        ax2.scatter(
            y_train, preds_dt,
            c="#2ca02c", s=60, alpha=0.9, edgecolors="black", label=f"Train ({dset_train_name})"
        )
        lims = [min(np.min(y_train), np.min(preds_dt)) - 0.05, max(np.max(y_train), np.max(preds_dt)) + 0.05]

    ax2.plot(lims, lims, "r--", linewidth=1.5, label="Ideal ($y = \\hat{y}$)")
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.7)
    ax2.axvline(0, color="gray", linestyle=":", alpha=0.7)
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel("Actual $\\Delta$ Rare F1", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Predicted $\\Delta$ Rare F1", fontsize=10, fontweight="bold")
    ax2.set_title("Predicted vs. Actual Utility Gain", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    fig3_path = figures_dir / "fig3_decision_tree_rule.png"
    plt.savefig(fig3_path, dpi=300)
    plt.savefig(str(fig3_path).replace(".png", ".pdf"), dpi=300)
    plt.close()

    print(f"RQ3 Figure 3 saved: {fig3_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze runs.csv and generate paper figures & tables.")
    parser.add_argument("--csv", type=str, default="results/runs.csv", help="Path to runs.csv")
    args = parser.parse_args()

    ensure_dirs()
    if not os.path.exists(args.csv):
        print(f"CSV file not found: {args.csv}. Please run experiments first.")
        return

    runs_df = pd.read_csv(args.csv)
    delta_df = load_and_prepare_deltas(args.csv)

    print("\n--- Generating RQ1 Main Results ---")
    generate_rq1_analysis(runs_df, delta_df)

    print("\n--- Generating RQ2 Cheap Signals Correlation ---")
    generate_rq2_analysis(delta_df)

    print("\n--- Generating RQ3 Benefit Predictor & Transfer ---")
    generate_rq3_analysis(delta_df)

    print("\n[SUCCESS] All paper tables and figures generated successfully.")


if __name__ == "__main__":
    main()
