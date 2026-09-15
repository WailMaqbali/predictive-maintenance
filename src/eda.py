"""Exploratory data analysis plots for the predictive maintenance dataset.

Run as a script to regenerate all EDA figures into ``figures/``:

    python -m src.eda
"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FAILURE_TYPE_COL, FIGURES_DIR, TARGET_COL, TYPE_COL
from src.data import load_processed_data

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Power [W]",
    "Temp difference [K]",
]


def plot_class_balance(df: pd.DataFrame, out_dir=FIGURES_DIR) -> None:
    """Save a bar chart of the Target class balance with counts and percentages."""
    counts = df[TARGET_COL].value_counts().sort_index()
    pct = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["No Failure (0)", "Failure (1)"],
        counts.values,
        color=["#4C72B0", "#C44E52"],
    )
    for bar, count, p in zip(bars, counts.values, pct.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n({p:.1f}%)",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.set_title("Class balance: machine failure (Target)")
    fig.tight_layout()
    fig.savefig(out_dir / "class_balance.png", dpi=150)
    plt.close(fig)


def plot_failure_type_breakdown(df: pd.DataFrame, out_dir=FIGURES_DIR) -> None:
    """Save a horizontal bar chart of failure counts by Failure Type."""
    counts = (
        df[df[TARGET_COL] == 1][FAILURE_TYPE_COL]
        .value_counts()
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(counts.index, counts.values, color="#C44E52")
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {v}", va="center")
    ax.set_xlabel("Count")
    ax.set_title("Failure type breakdown (rows with Target = 1)")
    fig.tight_layout()
    fig.savefig(out_dir / "failure_type_breakdown.png", dpi=150)
    plt.close(fig)


def plot_feature_distributions(df: pd.DataFrame, out_dir=FIGURES_DIR) -> None:
    """Save histograms of each numeric feature split by Target class."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_FEATURES):
        ax = axes[i]
        sns.histplot(
            data=df,
            x=col,
            hue=TARGET_COL,
            bins=40,
            stat="density",
            common_norm=False,
            palette={0: "#4C72B0", 1: "#C44E52"},
            ax=ax,
            legend=(i == 0),
        )
        ax.set_title(col)
    for j in range(len(NUMERIC_FEATURES), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Feature distributions by failure status", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_matrix(df: pd.DataFrame, out_dir=FIGURES_DIR) -> None:
    """Save a heatmap of the Pearson correlation matrix for numeric features + Target."""
    cols = NUMERIC_FEATURES + [TARGET_COL]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_matrix.png", dpi=150)
    plt.close(fig)


def plot_type_vs_failure(df: pd.DataFrame, out_dir=FIGURES_DIR) -> None:
    """Save a bar chart of failure rate broken down by product quality Type (L/M/H)."""
    rate = df.groupby(TYPE_COL)[TARGET_COL].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(rate.index, rate.values, color="#55A868")
    for i, v in enumerate(rate.values):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom")
    ax.set_ylabel("Failure rate (%)")
    ax.set_title("Failure rate by product quality type")
    fig.tight_layout()
    fig.savefig(out_dir / "failure_rate_by_type.png", dpi=150)
    plt.close(fig)


def run_all(out_dir=FIGURES_DIR) -> None:
    """Generate and save every EDA figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_processed_data()
    plot_class_balance(df, out_dir)
    plot_failure_type_breakdown(df, out_dir)
    plot_feature_distributions(df, out_dir)
    plot_correlation_matrix(df, out_dir)
    plot_type_vs_failure(df, out_dir)
    print(f"Saved EDA figures to {out_dir}")


if __name__ == "__main__":
    run_all()
