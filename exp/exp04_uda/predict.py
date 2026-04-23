"""
exp/exp04_uda/predict.py
========================
Ensemble prediction for the UDA experiment (exp04).

Loads the best.pt checkpoint from each of the 5 cross-validation folds,
runs inference on tests/test.csv, averages the softmax probabilities, and
writes predictions.csv to the run directory.

Output format matches exp01–03:
  row_id,predicted_label

Usage
-----
    python exp/exp04_uda/predict.py            # uses the latest run
    python exp/exp04_uda/predict.py --run 01   # uses a specific run
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR      = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_REGISTRY, RESULTS_DIR, TEST_FILE
from src.model import AttentionPoolingClassifier


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Run selection
# ---------------------------------------------------------------------------

def resolve_run_dir(exp_results_dir: Path, run_id: str | None) -> tuple[Path, str]:
    if run_id is not None:
        run_dir = exp_results_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir, run_id

    if not exp_results_dir.exists():
        raise FileNotFoundError(
            f"No results directory found: {exp_results_dir}\n"
            "Run train.py first to generate a run."
        )
    existing = sorted(
        [d.name for d in exp_results_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=int,
    )
    if not existing:
        raise FileNotFoundError(f"No run directories found under {exp_results_dir}")
    run_id  = existing[-1]
    run_dir = exp_results_dir / run_id
    return run_dir, run_id


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_test_csv(path: Path) -> tuple[list[str], list[str]]:
    row_ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_ids.append(row["row_id"].strip())
            texts.append(row["TEXT"].strip())
    return row_ids, texts


def tokenize_texts(texts: list[str], tokenizer, max_length: int) -> dict:
    return tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="UDA exp04 ensemble predictor")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run ID to load checkpoints from (e.g. '01'). Defaults to latest.",
    )
    args = parser.parse_args()

    cfg    = load_config(EXP_DIR / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mo = cfg["model"]
    tr = cfg["training"]

    exp_name        = cfg["experiment"]["name"]
    exp_results_dir = RESULTS_DIR / exp_name
    run_dir, run_id = resolve_run_dir(exp_results_dir, args.run)

    print(f"Experiment  : {exp_name}")
    print(f"Run         : {run_id}")
    print(f"Run dir     : {run_dir}")
    print(f"Device      : {device}")

    pretrained = MODEL_REGISTRY[mo["name"]]
    tokenizer  = AutoTokenizer.from_pretrained(pretrained)

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")
    row_ids, test_texts = load_test_csv(TEST_FILE)
    print(f"\nTest samples: {len(test_texts)}")

    # Tokenize all test texts once
    enc            = tokenize_texts(test_texts, tokenizer, tr["max_length"])
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    n_folds   = tr["n_folds"]
    all_probs: list[np.ndarray] = []

    print(f"\nRunning ensemble over {n_folds} folds...")
    for fold in range(n_folds):
        ckpt_path = run_dir / f"fold{fold}" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for fold {fold}: {ckpt_path}\n"
                "Run train.py first."
            )

        model = AttentionPoolingClassifier(
            pretrained_name=pretrained,
            hidden_dim=mo["hidden_dim"],
            mlp_hidden=mo["mlp_hidden"],
            dropout=mo["dropout"],
            num_labels=mo["num_labels"],
            freeze_encoder=True,
        ).to(device)
        model.load_state_dict(
            torch.load(str(ckpt_path), map_location=device, weights_only=True)
        )
        model.eval()
        print(f"  Loaded fold {fold}: {ckpt_path}")

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()  # (N, K)
        all_probs.append(probs)

    # Ensemble: average softmax probabilities, then argmax
    ensemble_probs = np.mean(all_probs, axis=0)   # (N, K)
    predictions    = ensemble_probs.argmax(axis=1).tolist()

    # Write predictions
    out_path = run_dir / "predictions.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "predicted_label"])
        for rid, pred in zip(row_ids, predictions):
            writer.writerow([rid, pred])

    pos = sum(predictions)
    neg = len(predictions) - pos
    print(f"\nPredictions saved: {out_path}")
    print(f"  Total     : {len(predictions)}")
    print(f"  Positive  : {pos}  ({pos / len(predictions):.1%})")
    print(f"  Negative  : {neg}  ({neg / len(predictions):.1%})")


if __name__ == "__main__":
    main()
