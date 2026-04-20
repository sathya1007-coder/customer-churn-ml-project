"""Inference entrypoint for scoring new customer rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_PATH


def predict(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Load the trained model and generate churn predictions for a CSV file."""

    model = joblib.load(MODEL_PATH)
    payload = pd.read_csv(input_path)
    predictions = payload.copy()
    predictions["predicted_churn"] = model.predict(payload)
    if hasattr(model, "predict_proba"):
        predictions["churn_probability"] = model.predict_proba(payload)[:, 1]
    predictions.to_csv(output_path, index=False)
    return predictions


def main() -> None:
    """CLI wrapper around the prediction helper."""

    parser = argparse.ArgumentParser(description="Run churn prediction inference.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", required=True, help="Path to the output CSV file.")
    args = parser.parse_args()
    predict(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
