"""Model evaluation and artifact generation."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
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


def _format_feature_name(feature_name: str) -> str:
    """Convert transformed feature names into a more readable label."""

    cleaned = feature_name
    prefixes = [
        "engineered_numeric__",
        "numeric__",
        "engineered_categorical__",
        "categorical__",
    ]
    for prefix in prefixes:
        cleaned = cleaned.replace(prefix, "")
    cleaned = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cleaned)
    cleaned = cleaned.replace("_", " ")
    return cleaned.title()


def export_feature_importance(pipeline: Pipeline, output_dir: Path) -> None:
    """Create a feature importance CSV and plot for the trained pipeline."""

    preprocessor = pipeline.named_steps["preprocessing"].named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "coef_"):
        importances = model.coef_.ravel()
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "display_feature": [_format_feature_name(name) for name in feature_names],
            "importance": importances,
            "absolute_importance": np.abs(importances),
        }
    ).sort_values("absolute_importance", ascending=False)

    top_features = importance_df.head(15).sort_values("importance")
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    plt.figure(figsize=(10, 7))
    colors = ["#c44e52" if value < 0 else "#4c72b0" for value in top_features["importance"]]
    plt.barh(top_features["display_feature"], top_features["importance"], color=colors)
    plt.title("Top Feature Importance: Best Model")
    plt.xlabel("Coefficient / Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=200)
    plt.close()


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
    export_feature_importance(trained_models[best_model_name], output_dir)

    return best_model_name, metrics
