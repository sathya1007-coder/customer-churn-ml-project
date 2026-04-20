"""Download the public churn dataset used by the project."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


def download_dataset(url: str, destination: Path, force: bool = False) -> pd.DataFrame:
    """Download the CSV dataset if needed and return it as a DataFrame."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if force or not destination.exists():
        urlretrieve(url, destination)
    return pd.read_csv(destination)
