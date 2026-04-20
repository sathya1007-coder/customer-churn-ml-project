"""Exploratory data analysis and visualization routines."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run_eda(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save a small EDA report as image files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x="Churn")
    ax.set_title("Churn Distribution")
    ax.set_xticks([0, 1], ["No", "Yes"])
    ax.set_xlabel("Churn")
    plt.tight_layout()
    plt.savefig(output_dir / "churn_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
    ax.set_title("Monthly Charges by Churn")
    ax.set_xticks([0, 1], ["No", "Yes"])
    ax.set_xlabel("Churn")
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_charges_by_churn.png", dpi=200)
    plt.close()

    correlation_columns = ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df[correlation_columns].corr(numeric_only=True),
        annot=True,
        cmap="Blues",
        fmt=".2f",
    )
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=200)
    plt.close()

    contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize="index")
    contract_churn.columns = ["No", "Yes"]
    contract_churn.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="Set2")
    plt.title("Contract Type vs Churn Rate")
    plt.xlabel("Contract")
    plt.ylabel("Share of Customers")
    plt.legend(title="Churn")
    plt.tight_layout()
    plt.savefig(output_dir / "contract_vs_churn.png", dpi=200)
    plt.close()
