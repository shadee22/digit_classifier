# Technical Implementation Plan: Model Persistence + Prediction CLI

**Status:** Draft
**Date:** 2026-03-03
**Author:** Tech Lead
**References:** PRD.md, STORIES.md
**Sprint:** 1

---

## 1. Proposed File Structure After Implementation

```
digit_classifier/
├── DIgit_Classifier using Logistic Regression.ipynb   # existing notebook, unchanged
├── README.md                                           # updated by Dev3 (US-04)
├── requirements.txt                                    # new, created by Dev3
├── .gitignore                                          # updated by Dev3 (US-01 AC)
├── docs/
│   ├── PRD.md
│   ├── STORIES.md
│   └── TECH_PLAN.md                                   # this document
├── models/                                             # git-ignored at contents level
│   ├── .gitkeep                                        # committed — tracks empty dir in git
│   ├── digits_model.joblib                             # runtime artifact, produced by train.py
│   └── metadata.json                                   # runtime artifact, produced by train.py
└── src/
    ├── model_utils.py                                  # Dev1: shared save/load utilities
    ├── train.py                                        # Dev1: training entry point
    └── predict.py                                      # Dev2: CLI prediction entry point
```

**Notes:**
- The original notebook is intentionally left untouched. It remains the exploratory reference artifact and the canonical baseline for the >= 96% accuracy target.
- `src/` is a flat directory, not a Python package. No `__init__.py` is required because `train.py` and `predict.py` are executed directly as scripts (`python src/train.py` from the repo root). Python adds `src/` to `sys.path` automatically when running a file inside it, so `from model_utils import ...` resolves without any `PYTHONPATH` manipulation.
- `models/.gitkeep` is committed so that the `models/` directory appears in a fresh clone, removing the need for any manual setup step. `models/*.joblib` and `models/metadata.json` are git-ignored.

---

## 2. Implementation Details

### 2.1 `src/model_utils.py` (Dev1)

Shared serialization utilities consumed by both `train.py` and `predict.py`. Centralizing save/load logic here ensures a single serialization protocol and single source of truth for the model path constant.

**Dependencies:** `joblib`, `os` (stdlib)

**Exact function signatures:**

```python
"""
src/model_utils.py

Shared utilities for saving and loading the digit classifier model.
Imported by both train.py and predict.py.
"""

import os
import joblib


DEFAULT_MODEL_PATH = "models/digits_model.joblib"


def save_model(model, path: str) -> None:
    """
    Serialize a fitted sklearn estimator to disk using joblib.

    Parameters
    ----------
    model : sklearn estimator
        A fitted model object (e.g., a trained LogisticRegression instance).
    path : str
        Destination file path, e.g. "models/digits_model.joblib".
        The parent directory must already exist before this call.
        train.py is responsible for creating it via os.makedirs.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If the parent directory does not exist or is not writable.
    """
    joblib.dump(model, path)


def load_model(path: str):
    """
    Deserialize a fitted sklearn estimator from disk using joblib.

    Parameters
    ----------
    path : str
        Path to a .joblib file produced by save_model().

    Returns
    -------
    sklearn estimator
        The deserialized fitted model, ready for .predict() calls.

    Raises
    ------
    FileNotFoundError
        If no file exists at `path`. Callers (predict.py) catch this
        and surface the user-friendly message:
        "Model not found. Run train.py first."
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at '{path}'.")
    return joblib.load(path)
```

**Key decisions:**
- `joblib` is preferred over `pickle` for sklearn models: it handles large numpy arrays via memory-mapped files and is the serialization format recommended by the sklearn documentation.
- `load_model` raises `FileNotFoundError` explicitly (rather than letting joblib surface a generic `OSError`) so `predict.py` can match on a specific, documented exception type.
- Path constants live here so both `train.py` and `predict.py` reference a single definition. If the path ever changes, it changes in one place.

---

### 2.2 `src/train.py` (Dev1)

Standalone training script. Replicates the notebook's `LogisticRegression(max_iter=1000000)` training on the full sklearn digits dataset, evaluates on a 25% held-out test split, persists the model via `model_utils.save_model`, and writes a metadata sidecar satisfying US-01 AC.

**Dependencies:** `sklearn`, `joblib`, `json`, `datetime`, `os`, `argparse` (stdlib), `model_utils`

**Exact function signatures:**

```python
"""
src/train.py

Train a LogisticRegression model on the sklearn digits dataset and save it to disk.

Usage
-----
    python src/train.py
    python src/train.py --model-path models/digits_model.joblib
"""

import argparse
import datetime
import json
import os
import sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from model_utils import save_model


def train_and_save(model_path: str = "models/digit_classifier.pkl") -> dict:
    """
    Load the sklearn digits dataset, train a LogisticRegression model,
    evaluate on the test split, save the model to disk, and write a
    metadata sidecar file.

    Parameters
    ----------
    model_path : str
        Destination file path for the serialized model.
        Default matches the project brief signature; the CLI overrides
        this to "models/digits_model.joblib" per PRD / US-01 AC.

    Returns
    -------
    dict
        Metadata dictionary with the following keys:
            sklearn_version : str   — e.g. "1.4.2"
            trained_at      : str   — ISO 8601 UTC timestamp, e.g. "2026-03-03T10:00:00Z"
            test_accuracy   : float — accuracy on the 25% test split
            model_path      : str   — absolute path where the model was saved

    Side effects
    ------------
    - Creates the parent directory of model_path if it does not exist
      (via os.makedirs with exist_ok=True).
    - Writes the serialized model to model_path.
    - Writes a metadata sidecar to <parent_dir>/metadata.json.
    - Prints progress lines to stdout.
    """
    # 1. Ensure output directory exists (self-bootstrapping; no manual mkdir needed)
    model_dir = os.path.dirname(model_path) or "."
    os.makedirs(model_dir, exist_ok=True)

    # 2. Load dataset — identical to notebook
    digits = load_digits()

    # 3. Split — random_state=42 added for reproducibility (notebook did not set a seed)
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.25, random_state=42
    )

    # 4. Train — max_iter=1000000 matches notebook exactly
    model = LogisticRegression(max_iter=1_000_000)
    model.fit(X_train, y_train)

    # 5. Evaluate
    test_accuracy = float(model.score(X_test, y_test))

    # 6. Save model
    save_model(model, model_path)

    # 7. Write metadata sidecar (US-01 AC: sklearn_version, trained_at, test_accuracy)
    metadata = {
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
        "test_accuracy": test_accuracy,
        "model_path": os.path.abspath(model_path),
    }
    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved     : {model_path}")
    print(f"Metadata saved  : {metadata_path}")
    print(f"Test accuracy   : {test_accuracy:.4f}")
    return metadata


def main() -> None:
    """
    Entry point. Parses --model-path argument and delegates to train_and_save().
    """
    parser = argparse.ArgumentParser(
        description="Train digit classifier and save model + metadata to disk."
    )
    parser.add_argument(
        "--model-path",
        default="models/digits_model.joblib",
        help="Destination path for the saved model (default: models/digits_model.joblib).",
    )
    args = parser.parse_args()
    train_and_save(model_path=args.model_path)


if __name__ == "__main__":
    main()
```

**Key decisions:**
- `random_state=42` is added to `train_test_split`. The notebook used no seed, making accuracy non-deterministic across runs. Fixing the seed guarantees the >= 96% acceptance criterion is reproducible.
- `os.makedirs(model_dir, exist_ok=True)` makes the script self-bootstrapping — no prerequisite `mkdir models/` step is required.
- The metadata sidecar satisfies the medium-likelihood / high-impact sklearn version skew risk from the PRD.

---

### 2.3 `src/predict.py` (Dev2)

CLI prediction script. Validates a 64-value comma-separated input string, loads the persisted model, and prints the predicted digit. All error paths exit with code 1 and a human-readable message — no raw Python tracebacks surface under any invalid-input scenario (US-03 AC).

**Dependencies:** `argparse`, `sys` (stdlib), `model_utils`

**Exact function signatures:**

```python
"""
src/predict.py

Load a saved digit classifier model and predict from a 64-value feature vector.

Usage
-----
    python src/predict.py --input "0,0,5,13,9,1,0,0,..."
    python src/predict.py --input "0,0,5,..." --model-path models/digits_model.joblib
"""

import argparse
import sys

from model_utils import load_model


EXPECTED_FEATURES = 64


def validate_input(raw_input: str) -> list:
    """
    Parse and validate a comma-separated string of pixel feature values.

    Parameters
    ----------
    raw_input : str
        Comma-separated string of numeric values, e.g. "0,0,5,13,9,1,...".
        Leading/trailing whitespace around each value is stripped before parsing.

    Returns
    -------
    list of float
        Parsed feature values, guaranteed to contain exactly 64 elements,
        all successfully coerced to float.

    Raises
    ------
    ValueError
        With a user-friendly message string (no traceback context) if:
        - Any value is non-numeric:
          "Error: All feature values must be numeric."
        - The count of values is not exactly 64:
          "Error: Expected 64 feature values, got <N>."
    """
    try:
        values = [float(v.strip()) for v in raw_input.split(",")]
    except ValueError:
        raise ValueError("Error: All feature values must be numeric.")

    if len(values) != EXPECTED_FEATURES:
        raise ValueError(
            f"Error: Expected {EXPECTED_FEATURES} feature values, got {len(values)}."
        )

    return values


def predict_digit(pixel_values: list, model_path: str) -> int:
    """
    Load the serialized model and return the predicted digit class.

    Parameters
    ----------
    pixel_values : list of float
        Exactly 64 numeric feature values representing a flattened 8x8 pixel
        image. Pixel intensities should be in range 0-16 (sklearn digits
        dataset convention); values outside this range are accepted but
        may produce unreliable predictions.
    model_path : str
        Path to the .joblib file produced by train.py.

    Returns
    -------
    int
        Predicted digit class in the range 0-9.

    Raises
    ------
    FileNotFoundError
        Propagated from model_utils.load_model() if no file exists at
        model_path. Caller (main) catches this and prints:
        "Model not found. Run train.py first."
    """
    model = load_model(model_path)
    prediction = model.predict([pixel_values])
    return int(prediction[0])


def main() -> None:
    """
    Entry point. Parses --input and --model-path, calls validate_input()
    then predict_digit(), catches all exceptions, and exits with code 1
    on any error. No raw traceback is ever shown to the user.
    """
    parser = argparse.ArgumentParser(
        description="Predict a digit (0-9) from a 64-value pixel feature vector."
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "64 comma-separated float values representing pixel intensities "
            "(range 0-16, matching the sklearn digits dataset format)."
        ),
    )
    parser.add_argument(
        "--model-path",
        default="models/digits_model.joblib",
        help="Path to the saved model file (default: models/digits_model.joblib).",
    )
    args = parser.parse_args()

    try:
        pixel_values = validate_input(args.input)
        result = predict_digit(pixel_values, args.model_path)
    except FileNotFoundError:
        print("Model not found. Run train.py first.")
        sys.exit(1)
    except ValueError as e:
        print(str(e))
        sys.exit(1)
    except Exception as e:
        # Catch-all: suppress raw traceback; surface a minimal friendly message
        print(f"Unexpected error: {e}")
        sys.exit(1)

    print(result)


if __name__ == "__main__":
    main()
```

**Key decisions:**
- `validate_input` and `predict_digit` are separated so Dev2 can unit-test input parsing independently without needing a model file on disk.
- The single `try/except` block in `main()` wraps both `validate_input` and `predict_digit` calls, guaranteeing no code path surfaces a raw traceback regardless of failure mode.
- `validate_input` raises `ValueError` with the exact error string specified in US-03 AC, so `main()` can simply `print(str(e))` without reformatting.
- `sys.exit(1)` is used on all error paths per US-02 and US-03 ACs.

---

### 2.4 `.gitignore` Additions (Dev3)

Append the following block to the root `.gitignore` (create the file if it does not exist):

```gitignore
# -------------------------------------------------------
# Digit Classifier — generated model artifacts
# Model files must be regenerated locally via train.py.
# -------------------------------------------------------
*.pkl
*.joblib
models/*.joblib
models/metadata.json

# Python bytecode caches
__pycache__/
*.py[cod]

# Jupyter notebook checkpoints
.ipynb_checkpoints/

# Virtual environment directories
venv/
.venv/
env/
```

**Notes:**
- `*.pkl` is listed per the project brief even though the canonical runtime extension is `.joblib`, guarding against accidental pickle serialization.
- `models/` directory itself is NOT ignored. The directory is tracked via `models/.gitkeep` so contributors see it in a fresh clone.
- `models/metadata.json` is explicitly ignored per US-01 AC. The sidecar contains runtime-specific timestamps and environment versions that must not be committed.

---

### 2.5 `requirements.txt` (Dev3)

```text
# Core ML / serialization dependencies (exact versions required)
scikit-learn==1.4.2     # MUST be pinned — sklearn estimator format is version-sensitive
joblib==1.3.2           # direct dependency of model_utils.py; also shipped by sklearn
numpy==1.26.4           # required transitive dep; pin to match sklearn build

# Notebook and visualization dependencies (for the existing .ipynb notebook)
matplotlib==3.8.4
seaborn==0.13.2
pandas==2.2.2
```

**Notes:**
- Exact version numbers above are representative placeholders. Dev3 must replace them with the output of `pip freeze | grep -E "scikit-learn|joblib|numpy|matplotlib|seaborn|pandas"` taken from the active development environment used to produce the 96% baseline.
- `joblib` must be listed explicitly even though sklearn pulls it in transitively, because `model_utils.py` imports it directly.
- Notebook visualization libraries are included so that a contributor who runs `pip install -r requirements.txt` can execute all existing notebook cells without a separate install step.

---

## 3. Dev Task Assignments

### Summary Table

| Dev | Tasks from STORIES.md | Deliverables |
|-----|----------------------|--------------|
| **Dev1** | TASK-01, TASK-02, TASK-03 | `src/model_utils.py`, `src/train.py`, `models/.gitkeep` |
| **Dev2** | TASK-04, TASK-05, TASK-06 | `src/predict.py` (including `validate_input`) |
| **Dev3** | TASK-07, TASK-08, TASK-09 | `requirements.txt`, updated `.gitignore`, updated `README.md`, smoke test sign-off |

---

### Dev1: `src/model_utils.py` + `src/train.py`

Covers TASK-01, TASK-02, TASK-03 (US-01).

| Task | Deliverable | Acceptance Gate |
|------|-------------|-----------------|
| TASK-01 | Create `models/` dir with `.gitkeep`; coordinate `.gitignore` additions with Dev3 | `git ls-files models/` shows `.gitkeep`; `models/*.joblib` is listed in `.gitignore` |
| TASK-02 | `src/model_utils.py` with `save_model` / `load_model`; `src/train.py` with `train_and_save` and `main` | `python src/train.py` produces `models/digits_model.joblib`; metadata shows `test_accuracy >= 0.96` |
| TASK-03 | `write_metadata` logic inside `train_and_save`; sidecar written to `models/metadata.json` | `metadata.json` contains `sklearn_version`, `trained_at`, `test_accuracy` after every run |

**Execution order:**
1. Day 1 — `src/model_utils.py` (`save_model`, `load_model`) — committed first so Dev2 can import it.
2. Day 1-2 — `src/train.py` skeleton, `train_and_save()` body without metadata.
3. Day 3 — Metadata sidecar block inside `train_and_save()`.

**Dependency note:** Dev2 imports `load_model` from `model_utils`. Dev1 should commit at minimum a stub version of `load_model` by end of Day 1 so Dev2 is not blocked. The stub can raise `NotImplementedError` and be replaced on Day 3.

---

### Dev2: `src/predict.py` + Input Validation

Covers TASK-04, TASK-05, TASK-06 (US-02, US-03).

| Task | Deliverable | Acceptance Gate |
|------|-------------|-----------------|
| TASK-04 | `src/predict.py` with `argparse`, `predict_digit()`, `main()` | `python src/predict.py --input "<64 values>"` prints a single digit (0-9) |
| TASK-05 | Model-not-found guard in `main()` | Script prints `"Model not found. Run train.py first."` and exits code 1 when `.joblib` is absent |
| TASK-06 | `validate_input()` with wrong-count and non-numeric error paths; `try/except` coverage | Wrong count → correct message + exit 1; non-numeric → correct message + exit 1; no raw traceback under any input |

**Execution order:**
1. Day 1 — `validate_input()` + `predict_digit()` signatures; argparse wiring in `main()`.
2. Day 2 — Model-not-found guard (TASK-05); local integration with stub `load_model`.
3. Day 3-4 — Full `try/except` coverage; integration with real `model_utils.load_model` once Dev1 commits it.

**Local development tip:** Dev2 can develop and test `validate_input()` without a real model by using a mock: `model = type("M", (), {"predict": lambda self, x: [5]})()`.

---

### Dev3: `requirements.txt` + `.gitignore` + README Update

Covers TASK-07, TASK-08, TASK-09 (US-04 + integration gate).

| Task | Deliverable | Acceptance Gate |
|------|-------------|-----------------|
| TASK-07 | `requirements.txt` with all direct deps pinned to exact environment versions | `pip install -r requirements.txt` succeeds on a clean virtualenv |
| TASK-08 | Updated `README.md` with Setup, Train, Predict, and Error sections; concrete 64-value example command | New contributor can follow README verbatim on a clean clone with no prior knowledge of the codebase |
| TASK-09 | End-to-end smoke test on fresh clone; all Definition of Done checklist items signed off | All 8 smoke-test steps below pass; Dev3 documents results and signs off |

**TASK-09 smoke test checklist:**
1. Clone repo to a clean directory; create a fresh virtualenv.
2. `pip install -r requirements.txt` — must complete with no errors.
3. `python src/train.py` — verify `models/digits_model.joblib` and `models/metadata.json` produced; confirm `test_accuracy >= 0.96` in `metadata.json`.
4. `python src/predict.py --input "<64 valid values>"` — verify a single digit (0-9) is printed; measure wall-clock time is under 1 second.
5. `python src/predict.py --input "<63 values>"` — verify output is exactly `"Error: Expected 64 feature values, got 63."` and exit code is 1.
6. `python src/predict.py --input "0,0,abc,0,..."` — verify output is exactly `"Error: All feature values must be numeric."` and exit code is 1.
7. Delete `models/digits_model.joblib`; run `python src/predict.py --input "<64 values>"` — verify output is `"Model not found. Run train.py first."` and exit code is 1.
8. Run `git status` — verify `models/digits_model.joblib` and `models/metadata.json` appear as untracked/ignored and are not staged.

---

## 4. Integration Points

```
                        +---------------------------+
                        |     src/model_utils.py    |
                        |                           |
                        |  save_model(model,        |
                        |             path: str)    |
                        |       -> None             |
                        |                           |
                        |  load_model(path: str)    |
                        |       -> sklearn estimator|
                        +-------------+-------------+
                                      |
                          imported by both scripts
                                      |
              +-----------------------+---------------------+
              |                                             |
+-------------+-------------+             +----------------+---------------+
|        src/train.py       |             |         src/predict.py         |
|                           |             |                                |
|  train_and_save(          |             |  validate_input(               |
|    model_path: str        |             |    raw_input: str              |
|  ) -> dict                |             |  ) -> list                     |
|                           |             |                                |
|  main() -> None           |             |  predict_digit(                |
|                           |             |    pixel_values: list,         |
|  Writes:                  |             |    model_path: str             |
|  - models/digits_model    |             |  ) -> int                      |
|    .joblib                |             |                                |
|  - models/metadata.json   |             |  main() -> None                |
+-------------+-------------+             +----------------+---------------+
              |                                             |
              | writes                                      | reads
              |                                            |
              +-------------------+-------------------------+
                                  |
                          +-------+--------+
                          |    models/     |
                          |                |
                          | digits_model   |
                          | .joblib        |
                          |                |
                          | metadata.json  |
                          +----------------+
                          (git-ignored, runtime only)
```

**Detailed connection map:**

| Consumer | Provider | Interface | Contract |
|----------|----------|-----------|----------|
| `train.py` | `model_utils.save_model` | `save_model(model, path: str) -> None` | Parent directory of `path` must exist before call; `train.py` ensures this via `os.makedirs`. |
| `predict.py` | `model_utils.load_model` | `load_model(path: str) -> sklearn estimator` | Returns fitted estimator or raises `FileNotFoundError`; `predict.py` catches and exits 1 with user-friendly message. |
| `predict.py` | `models/digits_model.joblib` | Filesystem — file produced by `train.py` | If absent, `load_model` raises `FileNotFoundError`. User sees `"Model not found. Run train.py first."`. |
| `train.py` | `sklearn.datasets.load_digits` | sklearn public API | Dataset is 1797 samples, 64 features, 10 classes; loaded in-process with no network or file I/O. |
| `predict.py` | `validate_input` (internal) | `validate_input(raw_input: str) -> list` | Called before `predict_digit`; raises `ValueError` with exact user-facing strings on bad input. |

**Import path note:** Both `train.py` and `predict.py` use `from model_utils import ...`. Because all three files are in `src/`, this works when scripts are invoked from the repo root as `python src/train.py` (Python inserts the script's directory into `sys.path[0]`). No `PYTHONPATH` export or `sys.path` manipulation is required.

---

## 5. Technical Risks and Mitigations

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|------------|
| R-01 | **sklearn version skew between train and predict environments.** A model serialized with sklearn 1.4.x may fail to deserialize or silently produce different predictions under a different version. | Medium | High | Pin `scikit-learn` to an exact version in `requirements.txt`. Record `sklearn_version` in `metadata.json` at save time so environment mismatches are detectable. A version-check warning in `load_model` can be added in Sprint 2. |
| R-02 | **Non-deterministic accuracy below 96%.** Without a fixed `random_state`, `train_test_split` produces different splits on every run and accuracy can dip below the 96% acceptance criterion on unlucky splits. | Medium | Medium | Set `random_state=42` in `train_test_split`. This makes the split fully reproducible. The notebook's observed 0.96 score aligns with this fixed seed. |
| R-03 | **Model file accidentally committed to version control.** `.joblib` files are binary blobs; a committed model inflates repo history and permanently embeds a specific sklearn version. | Low | Low | Add `models/*.joblib` and `models/metadata.json` to `.gitignore` in TASK-01 (Day 1, first task). Document in README that the model must be generated locally by running `train.py`. |
| R-04 | **Raw Python traceback reaching the end user via `predict.py`.** Any unhandled exception surfaces a multi-line traceback instead of a clean error message, violating US-03 AC. | Medium | Low | Wrap the entire `main()` execution body in a catch-all `except Exception` after specific `FileNotFoundError` and `ValueError` catches. `validate_input()` is designed to only ever raise `ValueError`. No exception propagates to the top level unhandled. |
| R-05 | **Merge conflict on `model_utils.py`.** Both Dev1 (owns the file) and Dev2 (needs `load_model`) touch this file if Dev2 adds any logic there. | Low | Medium | Dev1 commits `model_utils.py` with complete signatures by end of Day 1. Dev2 imports it as a read-only dependency and never modifies it. All input validation lives exclusively in `predict.py` (`validate_input`), not in `model_utils`. |
| R-06 | **Scripts break when run from inside `src/` instead of the repo root.** The default model path `"models/digits_model.joblib"` resolves relative to the current working directory, not the script location. If a user runs `cd src && python train.py`, the path resolves to `src/models/...` which does not exist. | Low | Medium | Document clearly in README that all commands must be run from the repo root (`python src/train.py`). Sprint 2 hardening can resolve the path relative to `__file__` using `os.path.dirname(os.path.abspath(__file__))` to make scripts location-agnostic. |
| R-07 | **Pixel values outside the 0-16 range accepted silently.** The sklearn digits dataset uses 0-16 intensities. The current validation only checks count and numeric type; out-of-range values produce no error but may yield unreliable predictions. | Medium | Low | Document the 0-16 range constraint in the README and in `predict.py`'s `--help` text. Range validation (`0 <= v <= 16`) is out-of-scope for this sprint per PRD Section 5, but is captured as a Sprint 2 enhancement to `validate_input()`. |
| R-08 | **Prediction round-trip time exceeds 1 second.** If joblib deserialization is slow on a cold filesystem, the < 1 second CLI SLA (US-02 AC) may be missed. | Low | Low | A LogisticRegression on 64 features / 10 classes is a tiny model (< 100 KB serialized). Deserialization should complete in single-digit milliseconds. Dev3 must measure wall-clock time during TASK-09 and flag if > 500 ms is observed. |

---

## Appendix: Key Constants and Baseline Facts

| Constant | Value | Source |
|----------|-------|--------|
| Number of input features | 64 | sklearn digits dataset (8x8 pixel images, flattened) |
| Number of classes | 10 | Digits 0-9 |
| Pixel intensity range | 0-16 | sklearn digits dataset convention |
| Total dataset samples | 1797 | `load_digits().data.shape[0]` |
| Target test accuracy | >= 96% | Notebook baseline (observed: 0.96); PRD success metric |
| Canonical model path | `models/digits_model.joblib` | PRD in-scope spec; US-01 AC |
| Metadata sidecar path | `models/metadata.json` | US-01 AC |
| `LogisticRegression` `max_iter` | 1,000,000 | Matches notebook exactly; ensures convergence on this dataset |
| `train_test_split` test size | 0.25 (25%) | Matches notebook default |
| `train_test_split` `random_state` | 42 | Added for reproducibility (not in notebook; introduced here) |
