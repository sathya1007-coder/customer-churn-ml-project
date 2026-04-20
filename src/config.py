"""Project configuration and filesystem paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw_telco_churn.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_telco_churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
METRICS_TABLE_PATH = OUTPUTS_DIR / "model_comparison.csv"
PREDICTIONS_PATH = OUTPUTS_DIR / "sample_predictions.csv"

DATA_URL = (
    "https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
ENGINEERED_NUMERIC_FEATURES = [
    "is_month_to_month",
    "has_fiber_optic",
    "charges_per_tenure",
    "num_addon_services",
]
ENGINEERED_CATEGORICAL_FEATURES = ["tenure_group"]
