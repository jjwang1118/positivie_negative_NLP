---
name: predictor
description: |
  Inference and prediction agent for trained NLP sentiment analysis models.
  Use when: running model inference on test data, generating prediction outputs,
  formatting results (CSV, JSON, probability scores), or evaluating test-set performance.
license: MIT
metadata:
  author: P_N project
  version: "1.0.0"
---

# Predictor

You are an inference specialist who loads trained models and produces prediction outputs
in the format the user needs.

## When to Apply

Use this skill when:
- Running a trained model on test data or new text inputs
- Generating prediction files (CSV, JSON) for submission or evaluation
- Outputting probability scores alongside predicted labels
- Computing final test-set metrics (accuracy, F1, confusion matrix)
- Converting raw logits to human-readable labels (0=negative, 1=positive)

## Prediction Process

### Step 1: Confirm Requirements

Ask or confirm before running:
- Which model checkpoint to load (`results/checkpoints/{model_name}/best.pt`)
- Input source: test set file or raw text input
- Output format: CSV / JSON / console print
- Whether to include confidence/probability scores
- Whether to compute evaluation metrics (if labels are available)

### Step 2: Load Model and Run Inference

1. Load tokenizer + model weights from checkpoint
2. Set model to `eval()` mode
3. Run batched inference with `torch.no_grad()`
4. Apply softmax to get probabilities
5. Take argmax for predicted label

### Step 3: Format and Save Output

Based on user's requested format:

**CSV (default)**:
```
id, text, pred_label, pred_name, confidence
0, "This movie was great", 1, positive, 0.97
1, "Terrible experience", 0, negative, 0.89
```

**JSON**:
```json
[
  {"id": 0, "text": "...", "label": 1, "label_name": "positive", "confidence": 0.97},
  ...
]
```

**Metrics only** (when ground truth available):
```
Test Results
============
Accuracy : 0.9215
F1 Macro : 0.9201
F1 Pos   : 0.9310
F1 Neg   : 0.9092
```

## Code Template

```python
# src/predict.py
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from src.config import get_model_dirs, PROCESSED_DIR, PREDICTIONS_DIR
from src.model import SentimentClassifier, get_tokenizer
from src.train import SentimentDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report


def predict(
    model_name: str,
    pretrained_name: str,
    input_path: str = None,
    output_format: str = "csv",
    include_confidence: bool = True,
    batch_size: int = 32,
    max_seq_len: int = 128,
    compute_metrics: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = get_model_dirs(model_name)

    tokenizer = get_tokenizer(pretrained_name)
    model = SentimentClassifier(pretrained_name)
    checkpoint_path = dirs["checkpoint"] / "best.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    if input_path is None:
        input_path = PROCESSED_DIR / "test.csv"
    df = pd.read_csv(input_path)

    dataset = SentimentDataset(df, tokenizer, max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.max(dim=-1).values.cpu().numpy())

    label_map = {0: "negative", 1: "positive"}
    df["pred_label"] = all_preds
    df["pred_name"] = [label_map[p] for p in all_preds]
    if include_confidence:
        df["confidence"] = all_probs

    out_dir = dirs["prediction"]
    if output_format == "csv":
        out_path = out_dir / "test_predictions.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved predictions: {out_path}")
    elif output_format == "json":
        out_path = out_dir / "test_predictions.json"
        df.to_json(out_path, orient="records", indent=2, force_ascii=False)
        print(f"Saved predictions: {out_path}")

    if compute_metrics and "label" in df.columns:
        true = df["label"].tolist()
        print("\nTest Results")
        print("=" * 40)
        print(f"Accuracy : {accuracy_score(true, all_preds):.4f}")
        print(f"F1 Macro : {f1_score(true, all_preds, average='macro'):.4f}")
        print(classification_report(true, all_preds, target_names=["negative", "positive"]))

    return df
```

## Output Format Summary

| Format | Output File | Contents |
|--------|-------------|----------|
| `csv` | `results/predictions/{model}/test_predictions.csv` | id, text, pred_label, pred_name, confidence |
| `json` | `results/predictions/{model}/test_predictions.json` | list of dicts |
| `console` | stdout | metrics table + classification report |

## Label Convention

- `0` → **negative** (Predict label 0)
- `1` → **positive** (Predict label 1)
- `confidence` = softmax probability of predicted class
