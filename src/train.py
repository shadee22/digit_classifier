"""Train a LogisticRegression digit classifier and persist it to disk.

Usage:
    python src/train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import model_utils

ACCURACY_THRESHOLD = 0.96
RANDOM_STATE       = 42


def load_data() -> tuple:
    digits = load_digits()
    return train_test_split(
        digits.data,
        digits.target,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=digits.target,
    )


def train_model(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(max_iter=1_000_000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> float:
    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    if accuracy < ACCURACY_THRESHOLD:
        print(
            f"WARNING: Accuracy {accuracy:.4f} is below target {ACCURACY_THRESHOLD}. "
            "Model saved anyway for inspection."
        )
    return accuracy


def main() -> None:
    print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_data()

    print("Training LogisticRegression (max_iter=1_000_000)...")
    model = train_model(X_train, y_train)

    accuracy = evaluate_model(model, X_test, y_test)

    print(f"Saving model to {model_utils.MODEL_PATH} ...")
    model_utils.save_model(model)

    print(f"Writing metadata to {model_utils.META_PATH} ...")
    model_utils.write_metadata(accuracy)

    print("Done. Model and metadata written successfully.")


if __name__ == "__main__":
    main()
