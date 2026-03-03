from pathlib import Path
import json
import joblib
import numpy as np
from datetime import datetime, timezone
import sklearn

REPO_ROOT         = Path(__file__).resolve().parent.parent
MODELS_DIR        = REPO_ROOT / "models"
MODEL_PATH        = MODELS_DIR / "digits_model.joblib"
META_PATH         = MODELS_DIR / "metadata.json"
EXPECTED_FEATURES = 64


def save_model(model) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run train.py first.")
    return joblib.load(MODEL_PATH)


def write_metadata(test_accuracy: float) -> None:
    metadata = {
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "test_accuracy": round(float(test_accuracy), 6),
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)


def parse_and_validate_input(raw: str) -> np.ndarray:
    tokens = [t.strip() for t in raw.split(",")]
    if len(tokens) != EXPECTED_FEATURES:
        raise ValueError(
            f"Error: Expected {EXPECTED_FEATURES} feature values, got {len(tokens)}."
        )
    try:
        values = [float(t) for t in tokens]
    except ValueError:
        raise ValueError("Error: All feature values must be numeric.")
    return np.array(values, dtype=np.float64).reshape(1, -1)
