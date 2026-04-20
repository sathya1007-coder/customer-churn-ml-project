"""Model training utilities for churn classification."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH, RANDOM_STATE, TEST_SIZE
from src.preprocessing import create_preprocessor


@dataclass
class TrainingArtifacts:
    """Container for the objects produced during model training."""

    X_test: pd.DataFrame
    y_test: pd.Series
    trained_models: dict[str, Pipeline]


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the input DataFrame into model features and labels."""

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def train_models(df: pd.DataFrame) -> TrainingArtifacts:
    """Train multiple candidate models and return them with the holdout set."""

    X, y = split_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    base_preprocessor = create_preprocessor()
    candidate_models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
    }

    trained_models: dict[str, Pipeline] = {}
    for model_name, estimator in candidate_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessing", base_preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline

    return TrainingArtifacts(X_test=X_test, y_test=y_test, trained_models=trained_models)


def save_model(model: Pipeline) -> None:
    """Persist the chosen model pipeline to disk."""

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
