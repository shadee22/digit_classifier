# STORIES.md: Model Persistence + Prediction CLI

**Sprint:** 1
**Date Created:** 2026-03-03
**Author:** Business Analyst / Scrum Master

---

## User Stories

---

### US-01: Train and Persist the Model

**As a** data scientist,
**I want** a standalone training script that saves the trained model to disk,
**So that** I never have to retrain the model inside a notebook session again and can share a reproducible artifact with teammates.

**Acceptance Criteria:**
- Running `python train.py` completes without error.
- A file `models/digits_model.joblib` is produced after the script finishes.
- The saved model achieves >= 96% accuracy on the sklearn digits test split (consistent with the notebook baseline).
- The scikit-learn version used during training is recorded in a sidecar file `models/metadata.json` (fields: `sklearn_version`, `trained_at`, `test_accuracy`).
- `models/*.joblib` and `models/metadata.json` are listed in `.gitignore`.

**Story Points:** 3

---

### US-02: Predict a Digit via the Command Line

**As a** developer or analyst,
**I want** a CLI script that accepts a pixel feature vector and prints the predicted digit,
**So that** I can integrate digit prediction into scripts and automated workflows without opening a notebook.

**Acceptance Criteria:**
- Running `python predict.py --input "0,1,5,..."` (64 comma-separated floats) prints a single predicted digit (0-9) to stdout.
- The script loads the model from `models/digits_model.joblib` without retraining.
- End-to-end round-trip time (invocation to printed result, model already on disk) is under 1 second.
- If the model file is not found, the script exits with a clear error message: `"Model not found. Run train.py first."`.

**Story Points:** 3

---

### US-03: Validate CLI Input and Surface Friendly Errors

**As a** user of the prediction CLI,
**I want** the script to validate my input and return a clear error message when it is malformed,
**So that** I understand exactly what went wrong instead of seeing a Python traceback.

**Acceptance Criteria:**
- Providing fewer or more than 64 values prints: `"Error: Expected 64 feature values, got <N>."` and exits with code 1.
- Providing any non-numeric value prints: `"Error: All feature values must be numeric."` and exits with code 1.
- All input parsing is wrapped in explicit `try/except` blocks; no raw Python tracebacks are shown to the user under any invalid-input scenario.
- Valid input continues to produce a correct prediction (regression guard).

**Story Points:** 2

---

### US-04: Document Train and Predict Commands in the README

**As a** new contributor or end user,
**I want** the README to include clear setup and usage instructions for the training and prediction scripts,
**So that** I can install dependencies and run my first prediction with a single documented command, without reading source code.

**Acceptance Criteria:**
- README contains a "Setup" section listing the install command (`pip install -r requirements.txt`).
- README contains a "Train" section with the exact command to run `train.py` and a note on the output file location.
- README contains a "Predict" section with a concrete example command including a sample 64-value input vector and the expected output.
- README contains a note explaining that `models/digits_model.joblib` is not committed to the repo and must be generated locally.
- The scikit-learn version is pinned in `requirements.txt` as documented in the README.

**Story Points:** 2

---

## Sprint Plan

**Sprint Goal:** A user can train the model once, save it to disk, and make digit predictions from the command line with proper input validation and documentation — all without opening a Jupyter notebook.

**Sprint Duration:** 1 week
**Team:** Dev1, Dev2, Dev3

---

### Task Breakdown

| Task ID | Description | User Story | Assigned To | Depends On |
|---------|-------------|------------|-------------|------------|
| TASK-01 | Create `models/` directory, add `models/*.joblib` and `models/metadata.json` to `.gitignore` | US-01 | Dev1 | None |
| TASK-02 | Write `train.py`: load sklearn digits dataset, train `LogisticRegression`, evaluate accuracy, save model to `models/digits_model.joblib` using `joblib` | US-01 | Dev1 | TASK-01 |
| TASK-03 | Write metadata sidecar logic in `train.py`: capture `sklearn_version`, `trained_at` timestamp, and `test_accuracy`; write to `models/metadata.json` | US-01 | Dev1 | TASK-02 |
| TASK-04 | Write `predict.py` skeleton: argument parsing (`--input` flag via `argparse`), model loading from `models/digits_model.joblib`, and prediction output | US-02 | Dev2 | None |
| TASK-05 | Add model-not-found guard in `predict.py`: check for file existence before loading, print user-friendly error and exit with code 1 if missing | US-02 | Dev2 | TASK-04 |
| TASK-06 | Add input validation logic in `predict.py`: check for exactly 64 values, check all values are numeric, raise user-friendly errors with exit code 1; wrap all parsing in `try/except` | US-03 | Dev2 | TASK-04 |
| TASK-07 | Pin `scikit-learn` (and other direct dependencies) in `requirements.txt` | US-04 | Dev3 | None |
| TASK-08 | Update README: add Setup, Train, and Predict sections with example commands, sample input vector, expected output, and note about `.gitignore` for model files | US-04 | Dev3 | TASK-07 |
| TASK-09 | End-to-end manual smoke test: run `train.py`, verify `models/` output, run `predict.py` with a valid and invalid input, verify round-trip time < 1s, confirm README steps work on a clean clone | All | Dev3 | TASK-03, TASK-06, TASK-08 |

---

### Parallelization View by Developer

```
WEEK 1
------
Day 1-2:
  Dev1  --> TASK-01 --> TASK-02
  Dev2  --> TASK-04
  Dev3  --> TASK-07

Day 3-4:
  Dev1  --> TASK-03  (extends TASK-02)
  Dev2  --> TASK-05, TASK-06  (extend TASK-04)
  Dev3  --> TASK-08  (extends TASK-07)

Day 5:
  Dev3  --> TASK-09  (integration smoke test; all other tasks must be complete)
```

---

## Definition of Done

The following checklist must be fully satisfied before any user story is considered complete and before the sprint is closed.

### Code Quality
- [ ] All new Python files (`train.py`, `predict.py`) are committed to version control.
- [ ] Code follows consistent style (PEP 8); no unused imports or debug `print` statements left in place.
- [ ] All input parsing in `predict.py` is wrapped in `try/except`; no raw tracebacks are reachable via invalid user input.

### Functionality
- [ ] `python train.py` runs to completion and produces `models/digits_model.joblib` and `models/metadata.json`.
- [ ] Saved model achieves >= 96% accuracy on the sklearn digits test split (confirmed by `metadata.json` output).
- [ ] `python predict.py --input "<64 values>"` prints a single digit (0-9) in under 1 second.
- [ ] `predict.py` exits with code 1 and a human-readable message for all invalid input scenarios (wrong count, non-numeric values, missing model file).

### Repository Hygiene
- [ ] `models/*.joblib` and `models/metadata.json` are listed in `.gitignore` and are NOT committed to the repository.
- [ ] `requirements.txt` exists with the scikit-learn version pinned.

### Documentation
- [ ] README contains Setup, Train, and Predict sections with working example commands.
- [ ] README includes a concrete 64-value sample input and the expected predicted output.
- [ ] README notes that the model file must be generated locally by running `train.py`.

### Acceptance Sign-Off
- [ ] Product Owner has reviewed and approved all acceptance criteria for US-01, US-02, US-03, and US-04.
- [ ] TASK-09 smoke test has been completed and all steps passed on a clean environment.
