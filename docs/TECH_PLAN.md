# TECH_PLAN.md — Digit Classifier: Model Persistence + Prediction CLI

**Date:** 2026-03-03
**Sprint:** 1
**Derived from:** PRD.md, STORIES.md

---

## 1. Proposed File Structure After Implementation

```
digit_classifier/
├── .gitignore                                         # Updated: excludes models/ artifacts
├── README.md                                          # Updated: Setup / Train / Predict sections
├── requirements.txt                                   # NEW: pinned dependencies
├── DIgit_Classifier using Logistic Regression.ipynb  # Unchanged: original exploration notebook
├── docs/
│   ├── PRD.md
│   ├── STORIES.md
│   └── TECH_PLAN.md
├── models/                                            # NEW directory, git-ignored contents
│   └── .gitkeep                                       # Tracks empty dir in git only
└── src/
    ├── model_utils.py                                 # Shared utilities (paths, save/load, validate)
    ├── train.py                                       # Training + serialization script
    └── predict.py                                     # CLI prediction script
```

---

## 2. Implementation Details

### 2.1 `src/model_utils.py` — Shared Utilities

**Purpose:** Single module imported by both `train.py` and `predict.py`.

**Libraries:** `joblib`, `json`, `datetime`, `pathlib`, `sklearn`, `numpy`

**Exact Function Signatures:**

```python
from pathlib import Path
import json, joblib, numpy as np
from datetime import datetime, timezone
import sklearn

REPO_ROOT         = Path(__file__).resolve().parent.parent
MODELS_DIR        = REPO_ROOT / "models"
MODEL_PATH        = MODELS_DIR / "digits_model.joblib"
META_PATH         = MODELS_DIR / "metadata.json"
EXPECTED_FEATURES = 64

def save_model(model) -> None:
    """Persist a fitted sklearn estimator to MODEL_PATH using joblib."""

def load_model():
    """Load and return the persisted model. Raises FileNotFoundError if missing."""

def write_metadata(test_accuracy: float) -> None:
    """Write sklearn_version, trained_at (ISO-8601 UTC), test_accuracy to metadata.json."""

def parse_and_validate_input(raw: str) -> np.ndarray:
    """Parse comma-separated string. Returns shape (1,64) array.
    Raises ValueError for wrong count or non-numeric values."""
```

---

### 2.2 `src/train.py` — Training Script

**Purpose:** Replicates notebook training, evaluates accuracy, saves model + metadata.

**Libraries:** `sklearn.datasets`, `sklearn.linear_model`, `sklearn.model_selection`, `model_utils`

**Exact Function Signatures:**

```python
ACCURACY_THRESHOLD = 0.96
RANDOM_STATE       = 42

def load_data() -> tuple:
    """Returns (X_train, X_test, y_train, y_test) with random_state=42, stratified."""

def train_model(X_train, y_train) -> LogisticRegression:
    """Trains LogisticRegression(max_iter=1_000_000). Matches notebook exactly."""

def evaluate_model(model, X_test, y_test) -> float:
    """Returns accuracy. Prints warning if < 0.96 but does not abort."""

def main() -> None:
    """Orchestrates: load → train → evaluate → save_model → write_metadata."""
```

---

### 2.3 `src/predict.py` — Prediction CLI

**Purpose:** CLI that validates input, loads model, prints predicted digit (0-9).

**Libraries:** `argparse`, `sys`, `model_utils`

**Exact Function Signatures:**

```python
def build_parser() -> argparse.ArgumentParser:
    """Returns parser with required --input argument."""

def predict(raw_input: str) -> int:
    """Validates input then loads model. Returns predicted digit as int."""
    # Order matters: validate BEFORE loading model
    features = model_utils.parse_and_validate_input(raw_input)
    model    = model_utils.load_model()
    return int(model.predict(features)[0])

def main() -> None:
    """Parses args, calls predict(), catches FileNotFoundError/ValueError, exits with code 1 on error."""
```

---

### 2.4 `.gitignore` Additions

```gitignore
# Trained model artifacts
models/*.joblib
models/metadata.json

# Python caches
__pycache__/
*.py[cod]
.pytest_cache/

# Jupyter
.ipynb_checkpoints/

# Virtual environments
venv/
.venv/
env/
```

---

### 2.5 `requirements.txt`

```text
scikit-learn==1.4.2   # MUST match training environment
joblib==1.3.2
numpy>=1.24,<2.0
matplotlib>=3.7
seaborn>=0.12
pandas>=2.0
```

> Dev3: replace versions with output of `pip freeze | grep -E "scikit-learn|joblib|numpy"`

---

## 3. Dev Task Assignments

| Dev | Tasks | Deliverables |
|-----|-------|-------------|
| **Dev1** | TASK-01, TASK-02, TASK-03 | `models/.gitkeep`, `.gitignore`, `src/model_utils.py`, `src/train.py` |
| **Dev2** | TASK-04, TASK-05, TASK-06 | `src/predict.py`, `parse_and_validate_input()` in model_utils |
| **Dev3** | TASK-07, TASK-08, TASK-09 | `requirements.txt`, updated `README.md`, smoke test sign-off |

### Dev1 Tasks
- **TASK-01:** Create `models/` dir, add `.gitkeep`, update `.gitignore`
- **TASK-02:** Implement `src/model_utils.py` (save/load) + `src/train.py`
- **TASK-03:** Add `write_metadata()` to model_utils, wire into train.py

### Dev2 Tasks
- **TASK-04:** Implement `src/predict.py` with argparse, model loading, output
- **TASK-05:** Verify model-not-found guard (FileNotFoundError → exit code 1)
- **TASK-06:** Implement `parse_and_validate_input()` in model_utils

### Dev3 Tasks
- **TASK-07:** Create `requirements.txt` with pinned versions from actual env
- **TASK-08:** Rewrite `README.md` with Setup/Train/Predict/Error sections
- **TASK-09:** End-to-end smoke test on fresh clone

---

## 4. Integration Points

```
         src/model_utils.py
         (paths, save, load, validate)
              /          \
        imports           imports
           /                \
  src/train.py         src/predict.py
  (writes artifacts)   (reads artifacts)
           \                /
            models/
            digits_model.joblib
            metadata.json
            (git-ignored, runtime only)
```

**Key contract:** Both scripts run from repo root as `python src/train.py` / `python src/predict.py`. The `sys.path.insert` uses `Path(__file__).resolve().parent` (absolute, cwd-independent).

---

## 5. Technical Risks

| Risk | Mitigation |
|------|-----------|
| sklearn version mismatch on model load | Exact pin in requirements.txt; metadata.json records version |
| Non-deterministic accuracy | `random_state=42` + `stratify=digits.target` in `load_data()` |
| `models/` missing on fresh clone | `mkdir(parents=True, exist_ok=True)` in save functions + `.gitkeep` committed |
| Merge conflict on `model_utils.py` | Dev1 commits stubs (`raise NotImplementedError`); Dev2 fills them in a focused PR |
| User passes `"0, 1, 5, ..."` with spaces | `t.strip()` in `parse_and_validate_input()` handles it |

---

## 6. Parallel Implementation Timeline

```
Day 1-2 (all parallel):
  Dev1: TASK-01 → TASK-02
  Dev2: TASK-04
  Dev3: TASK-07

Day 3-4 (all parallel):
  Dev1: TASK-03
  Dev2: TASK-05 → TASK-06
  Dev3: TASK-08

Day 5 (sequential, all merged):
  Dev3: TASK-09 (smoke test)
```
