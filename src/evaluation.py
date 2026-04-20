"""Model evaluation and artifact generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.config import METRICS_PATH, METRICS_TABLE_PATH, PREDICTIONS_PATH
from src.utils import save_json


def evaluate_models(
    trained_models: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> tuple[str, dict[str, dict[str, float]]]:
    """Evaluate candidate models, save metrics, and return the best model name."""

    metrics: dict[str, dict[str, float]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    for model_name, pipeline in trained_models.items():
        predictions = pipeline.predict(X_test)
        metrics[model_name] = {
            "accuracy": round(accuracy_score(y_test, predictions), 4),
            "precision": round(precision_score(y_test, predictions), 4),
            "recall": round(recall_score(y_test, predictions), 4),
            "f1_score": round(f1_score(y_test, predictions), 4),
        }

        report = classification_report(y_test, predictions, output_dict=True)
        save_json(report, output_dir / f"{model_name}_classification_report.json")

        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            display_labels=["No Churn", "Churn"],
            cmap="Blues",
        )
        disp.ax_.set_title(f"Confusion Matrix: {model_name.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=200)
        plt.close()

    comparison_df = pd.DataFrame(metrics).T.sort_values("f1_score", ascending=False)
    comparison_df.to_csv(METRICS_TABLE_PATH, index=True)
    save_json(comparison_df.to_dict(orient="index"), METRICS_PATH)

    best_model_name = comparison_df.index[0]
    sample_predictions = X_test.head(10).copy()
    sample_predictions["actual_churn"] = y_test.head(10).values
    sample_predictions["predicted_churn"] = trained_models[best_model_name].predict(
        X_test.head(10)
    )
    sample_predictions.to_csv(PREDICTIONS_PATH, index=False)

    return best_model_name, metrics
