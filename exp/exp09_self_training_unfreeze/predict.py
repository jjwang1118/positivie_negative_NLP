"""
exp/exp09_self_training_unfreeze/predict.py
============================================
Ensemble prediction for exp09.

Loads best.pt (final stage) from each fold, averages softmax probabilities,
writes predictions.csv to the run directory.

Usage
-----
    python exp/exp09_self_training_unfreeze/predict.py            # latest run
    python exp/exp09_self_training_unfreeze/predict.py --run 01   # specific run
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR      = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_REGISTRY, RESULTS_DIR, TEST_FILE
from src.model import MeanPoolingClassifier


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_run_dir(exp_results_dir: Path, run_id: str | None) -> tuple[Path, str]:
    if run_id is not None:
        run_dir = exp_results_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir, run_id

    if not exp_results_dir.exists():
        raise FileNotFoundError(
            f"No results directory found: {exp_results_dir}\nRun train.py first."
        )
    existing = sorted(
        [d.name for d in exp_results_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=int,
    )
    if not existing:
        raise FileNotFoundError(f"No run directories found under {exp_results_dir}")
    run_id  = existing[-1]
    return exp_results_dir / run_id, run_id


def load_test_csv(path: Path) -> tuple[list[str], list[str]]:
    row_ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row_ids.append(row["row_id"].strip())
            texts.append(row["TEXT"].strip())
    return row_ids, texts


class InferenceDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_len: int):
        self.texts     = texts
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


@torch.no_grad()
def batched_predict(
    model: MeanPoolingClassifier,
    texts: list[str],
    tokenizer,
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    ds     = InferenceDataset(texts, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs_list = []
    model.eval()
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        probs_list.append(torch.softmax(model(ids, mask), dim=-1).cpu().numpy())
    return np.vstack(probs_list)


def main() -> None:
    parser = argparse.ArgumentParser(description="exp09 ensemble predictor")
    parser.add_argument("--run", type=str, default=None,
                        help="Run ID (e.g. '01'). Defaults to latest.")
    args = parser.parse_args()

    cfg    = load_config(EXP_DIR / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mo = cfg["model"]
    s1 = cfg["stage1"]

    exp_name        = cfg["experiment"]["name"]
    exp_results_dir = RESULTS_DIR / exp_name
    run_dir, run_id = resolve_run_dir(exp_results_dir, args.run)

    print(f"Experiment : {exp_name}")
    print(f"Run        : {run_id}")
    print(f"Device     : {device}")

    pretrained = MODEL_REGISTRY[mo["name"]]
    tokenizer  = AutoTokenizer.from_pretrained(pretrained)

    row_ids, test_texts = load_test_csv(TEST_FILE)
    print(f"\nTest samples: {len(test_texts)}")

    n_folds   = s1["n_folds"]
    all_probs: list[np.ndarray] = []

    print(f"\nRunning ensemble over {n_folds} folds...")
    for fold in range(n_folds):
        ckpt_path = run_dir / f"fold{fold}" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\nRun train.py first."
            )

        # freeze_encoder=False: encoder may have been partially fine-tuned;
        # requires_grad flags don't affect inference, but this reflects the
        # trained architecture accurately.
        model = MeanPoolingClassifier(
            pretrained_name=pretrained,
            hidden_dim=mo["hidden_dim"],
            mlp_hidden=mo["mlp_hidden"],
            dropout=mo["dropout"],
            num_labels=mo["num_labels"],
            freeze_encoder=False,
        ).to(device)
        model.load_state_dict(
            torch.load(str(ckpt_path), map_location=device, weights_only=True)
        )
        print(f"  Loaded fold {fold}: {ckpt_path}")

        probs = batched_predict(
            model, test_texts, tokenizer,
            max_len=s1["max_length"],
            batch_size=s1["batch_size"],
            device=device,
        )
        all_probs.append(probs)

    ensemble_probs = np.mean(all_probs, axis=0)
    predictions    = ensemble_probs.argmax(axis=1).tolist()

    out_path = run_dir / "predictions.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "LABEL"])
        for rid, pred in zip(row_ids, predictions):
            writer.writerow([rid, pred])

    pos = sum(predictions)
    neg = len(predictions) - pos
    print(f"\nPredictions saved: {out_path}")
    print(f"  Total    : {len(predictions)}")
    print(f"  Positive : {pos}  ({pos / len(predictions):.1%})")
    print(f"  Negative : {neg}  ({neg / len(predictions):.1%})")


if __name__ == "__main__":
    main()
