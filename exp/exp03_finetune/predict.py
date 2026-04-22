"""
exp/exp03_finetune/predict.py
==============================
Ensemble prediction using the 3-fold RoBERTa models trained in exp03.

Ensemble strategy: average softmax probabilities across all fold models,
then argmax to obtain the final label.

Output: results/exp03/{run_id}/predictions.csv
Format: row_id,LABEL

Usage
-----
    python exp/exp03_finetune/predict.py            # latest run
    python exp/exp03_finetune/predict.py --run 01
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
from src.model import SequenceClassifier


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_config(cfg: dict) -> dict:
    flat = {}
    for section in cfg.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat


# ---------------------------------------------------------------------------
# Run selection
# ---------------------------------------------------------------------------

def resolve_run_dir(exp_results_dir: Path, run_id: str | None) -> tuple[Path, str]:
    if run_id is not None:
        run_dir = exp_results_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir, run_id

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


def tokenise(texts: list[str], tokenizer, max_length: int) -> dict:
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None, help="Run ID (e.g. 01). Defaults to latest.")
    args = parser.parse_args()

    cfg    = load_config(EXP_DIR / "config.yaml")
    hp     = flatten_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name        = cfg["experiment"]["name"]
    exp_results_dir = RESULTS_DIR / exp_name
    run_dir, run_id = resolve_run_dir(exp_results_dir, args.run)

    print(f"Experiment : {exp_name}")
    print(f"Run        : {run_id}")
    print(f"Device     : {device}")

    pretrained = MODEL_REGISTRY[hp["name"]]
    tokenizer  = AutoTokenizer.from_pretrained(pretrained)

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")
    row_ids, test_texts = load_test_csv(TEST_FILE)
    print(f"Test samples: {len(test_texts)}")

    enc = tokenise(test_texts, tokenizer, hp["max_length"])
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    n_folds    = hp["n_folds"]
    all_probs: list[np.ndarray] = []

    for fold in range(n_folds):
        ckpt_path = run_dir / f"fold{fold}" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = SequenceClassifier(
            pretrained_name=pretrained,
            hidden_dim=hp["hidden_dim"],
            dropout=hp["dropout"],
            num_labels=hp["num_labels"],
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        print(f"  Loaded fold {fold}: {ckpt_path}")

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    ensemble_probs = np.mean(all_probs, axis=0)
    predictions    = ensemble_probs.argmax(axis=1).tolist()

    out_path = run_dir / "predictions.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "LABEL"])
        for rid, pred in zip(row_ids, predictions):
            writer.writerow([rid, pred])

    print(f"\nPredictions saved to: {out_path}")
    print(f"  Total predictions : {len(predictions)}")
    pos = sum(predictions)
    print(f"  Positive (1)      : {pos}  ({pos / len(predictions):.1%})")
    print(f"  Negative (0)      : {len(predictions) - pos}  ({(len(predictions) - pos) / len(predictions):.1%})")


if __name__ == "__main__":
    main()
