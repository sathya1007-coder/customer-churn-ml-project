"""Data cleaning and preprocessing utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    ENGINEERED_CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
)
from src.features import FeatureEngineer


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize data types and target labels from the raw churn dataset."""

    cleaned = df.copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype(str)
    cleaned["Churn"] = cleaned["Churn"].map({"Yes": 1, "No": 0})
    cleaned = cleaned.drop(columns=["customerID"])
    return cleaned


def create_preprocessor() -> Pipeline:
    """Create the full modeling preprocessor with feature engineering."""

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("engineered_numeric", numeric_transformer, ENGINEERED_NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
            (
                "engineered_categorical",
                categorical_transformer,
                ENGINEERED_CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("preprocessor", preprocessor),
        ]
    )
