from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Data directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# Model directory
MODEL_DIR = PROJECT_ROOT / "models"


# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"


# Raw dataset files
RED_WINE_PATH = RAW_DATA_DIR / "winequality-red.csv"
WHITE_WINE_PATH = RAW_DATA_DIR / "winequality-white.csv"


# Saved model artifacts
RANDOM_FOREST_MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"


def create_project_directories():
    """
    Create project directories if they do not already exist.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)