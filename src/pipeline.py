"""Complete end-to-end project pipeline."""

from __future__ import annotations

from src.config import (
    DATA_DIR,
    DATA_URL,
    METRICS_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)
from src.data_ingestion import download_dataset
from src.eda import run_eda
from src.evaluation import evaluate_models
from src.modeling import save_model, train_models
from src.preprocessing import clean_dataset
from src.utils import ensure_directories


def main() -> None:
    """Execute the project from ingestion through model persistence."""

    ensure_directories([DATA_DIR, MODELS_DIR, OUTPUTS_DIR])

    raw_df = download_dataset(DATA_URL, RAW_DATA_PATH)
    cleaned_df = clean_dataset(raw_df)
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)

    run_eda(cleaned_df, OUTPUTS_DIR)

    artifacts = train_models(cleaned_df)
    best_model_name, metrics = evaluate_models(
        artifacts.trained_models,
        artifacts.X_test,
        artifacts.y_test,
        OUTPUTS_DIR,
    )
    save_model(artifacts.trained_models[best_model_name])

    print("Pipeline completed successfully.")
    print(f"Best model: {best_model_name}")
    print(f"Metrics saved to: {METRICS_PATH}")
    for model_name, model_metrics in metrics.items():
        print(f"{model_name}: {model_metrics}")


if __name__ == "__main__":
    main()
