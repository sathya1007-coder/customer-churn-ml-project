"""Feature engineering helpers for customer churn modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add domain-inspired features before preprocessing."""

    addon_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        """Return self because this transformer is stateless."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create engineered columns from the raw customer data."""

        transformed = X.copy()
        transformed["is_month_to_month"] = (
            transformed["Contract"].eq("Month-to-month").astype(int)
        )
        transformed["has_fiber_optic"] = (
            transformed["InternetService"].eq("Fiber optic").astype(int)
        )
        transformed["charges_per_tenure"] = (
            transformed["MonthlyCharges"] / transformed["tenure"].replace(0, np.nan)
        ).fillna(transformed["MonthlyCharges"])
        transformed["num_addon_services"] = (
            transformed[self.addon_columns].eq("Yes").sum(axis=1)
        )
        transformed["tenure_group"] = pd.cut(
            transformed["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["0-12", "13-24", "25-48", "49-72"],
        ).astype(str)
        return transformed
