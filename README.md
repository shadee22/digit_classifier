# Digit Classifier

Logistic Regression model trained on the scikit-learn digits dataset (8x8 pixel images, 10 classes). Achieves >= 96% accuracy on the test split.

## Setup

```bash
pip install -r requirements.txt
```

> The scikit-learn version is pinned in `requirements.txt`. Using a different version may cause model load errors.

## Train

Train the model and save it to disk:

```bash
python src/train.py
```

Outputs:
- `models/digits_model.joblib` — serialized fitted model
- `models/metadata.json` — training metadata (sklearn version, timestamp, accuracy)

> These files are listed in `.gitignore` and are **not** committed. Every developer must run `train.py` locally before using `predict.py`.

## Predict

```bash
python src/predict.py --input "0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0"
```

Expected output: `0`

### Input Format

- Exactly **64 comma-separated float values**
- Represents the flattened 8x8 pixel grid of a handwritten digit
- Pixel value range: **0 to 16**

### Error Messages

| Scenario | Output | Exit Code |
|---|---|---|
| Model file missing | `Model not found. Run train.py first.` | 1 |
| Wrong number of values | `Error: Expected 64 feature values, got <N>.` | 1 |
| Non-numeric value | `Error: All feature values must be numeric.` | 1 |

## Project Structure

```
digit_classifier/
├── src/
│   ├── model_utils.py   # Shared utilities (save, load, validate)
│   ├── train.py         # Training script
│   └── predict.py       # Prediction CLI
├── models/              # Generated at runtime (git-ignored)
├── docs/                # PRD, stories, tech plan
├── requirements.txt
└── README.md
```
