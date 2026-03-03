# PRD: Model Persistence + Prediction CLI

**Status:** Proposed
**Date:** 2026-03-03
**Author:** Product Owner

---

## 1. Feature Title

Model Persistence + Prediction CLI

---

## 2. Business Objective

Currently, the trained Logistic Regression model exists only within a Jupyter notebook session and is lost when the kernel restarts, forcing users to retrain on every use. This feature serializes the trained model to disk and exposes a command-line interface so that users can make digit predictions at any time without retraining. The result is a repeatable, scriptable prediction workflow that moves the project from an exploratory notebook toward a usable tool.

---

## 3. Success Metrics

| Metric | Target |
|---|---|
| Model saved to disk after training | Single `.joblib` file produced in `models/` directory |
| CLI prediction round-trip time | < 1 second from invocation to printed result (excluding training) |
| Prediction accuracy via CLI | Matches notebook baseline of >= 96% on the sklearn digits test split |
| CLI usability | A new user can install dependencies and run a prediction with a single command documented in the README |

---

## 4. In-Scope

- **Model serialization:** Save the trained `LogisticRegression` model to `models/digits_model.joblib` using `joblib`.
- **Model loading:** Load the saved model from disk at prediction time without retraining.
- **Training script:** A standalone `train.py` script that trains the model on the sklearn digits dataset and saves it to disk.
- **Prediction CLI:** A `predict.py` script (or CLI entry point) that accepts a pixel feature vector as input (16 comma-separated floats matching the 8x8 flattened image) and prints the predicted digit class (0-9).
- **Basic input validation:** Reject inputs that are the wrong length or contain non-numeric values with a clear error message.
- **README update:** Document the train and predict commands with examples.

---

## 5. Out-of-Scope

- Web API or HTTP serving (REST/FastAPI endpoints).
- Image file input (PNG/JPEG); only raw feature vectors are supported.
- Model versioning or experiment tracking (e.g., MLflow, DVC).
- Automated retraining or scheduled jobs.
- A graphical or web-based UI.
- Support for models other than the existing Logistic Regression baseline.
- Unit or integration tests (tracked separately as a future feature).
- Containerization (Docker).

---

## 6. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Saved model becomes stale if sklearn version changes between train and predict environments | Medium | High | Pin `scikit-learn` version in `requirements.txt`; log the version into a sidecar metadata file at save time |
| Feature vector format is unclear to end users, leading to bad inputs | High | Medium | Document the exact 64-feature format (8x8 pixel values, range 0-16) in the README with a concrete example |
| Model file checked into version control inflates repo size over time | Low | Low | Add `models/*.joblib` to `.gitignore`; document that the model must be generated locally by running `train.py` |
| CLI input parsing errors surface confusing tracebacks to users | Medium | Low | Wrap input parsing in explicit try/except blocks and print user-friendly error messages |
