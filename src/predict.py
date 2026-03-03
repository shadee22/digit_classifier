"""Predict a digit class from a 64-value feature vector.

Usage:
    python src/predict.py --input "0,0,5,13,9,1,0,0,..."

Output:
    A single integer in [0, 9].

Exit codes:
    0  -- successful prediction
    1  -- any error (model missing, wrong input length, non-numeric input)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_utils


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict a handwritten digit (0-9) from a 64-value pixel feature vector."
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="FEATURES",
        help=(
            "Comma-separated list of exactly 64 pixel values (floats, range 0-16) "
            "representing the flattened 8x8 digit image."
        ),
    )
    return parser


def predict(raw_input: str) -> int:
    features = model_utils.parse_and_validate_input(raw_input)
    model    = model_utils.load_model()
    result   = model.predict(features)
    return int(result[0])


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    try:
        digit = predict(args.input)
        print(digit)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)
    except ValueError as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
