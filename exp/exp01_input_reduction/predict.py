"""
exp/exp01_input_reduction/predict.py
=====================================
Ensemble prediction on test set using all fold models from a specific run.

Loads checkpoints from:
  results/exp01/{run_id}/fold{k}/best.pt

Saves predictions to:
  results/exp01/{run_id}/predictions.csv

Output format:
  row_id,LABEL
  0,1
  1,0
  ...

Usage
-----
    python exp/exp01_input_reduction/predict.py           # latest run
    python exp/exp01_input_reduction/predict.py --run 02
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR      = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_REGISTRY, RESULTS_DIR, TEST_FILE
from src.model import AttentionPoolingClassifier


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
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, encodings: dict):
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        return {k: v[idx] for k, v in self.encodings.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_csv(path: Path) -> list[str]:
    """Load test.csv — reads TEXT column, ignores others."""
    texts = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["TEXT"].strip())
    return texts


@torch.no_grad()
def get_probs(
    model: AttentionPoolingClassifier,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    all_probs = []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits         = model(input_ids=input_ids, attention_mask=attention_mask)
        probs          = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


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

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")
    texts = load_test_csv(TEST_FILE)
    print(f"Test samples: {len(texts)}")

    pretrained = MODEL_REGISTRY[hp["name"]]
    tokenizer  = AutoTokenizer.from_pretrained(pretrained)
    encodings  = tokenizer(
        texts,
        max_length=hp["max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    loader = DataLoader(TextDataset(encodings), batch_size=hp["batch_size"], shuffle=False)

    n_folds   = hp["n_folds"]
    sum_probs = np.zeros((len(texts), hp["num_labels"]), dtype=np.float64)

    for fold in range(n_folds):
        ckpt_path = run_dir / f"fold{fold}" / "best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\nRun train.py first.")

        model = AttentionPoolingClassifier(
            pretrained_name=pretrained,
            hidden_dim=hp["hidden_dim"],
            mlp_hidden=hp["mlp_hidden"],
            dropout=hp["dropout"],
            num_labels=hp["num_labels"],
            freeze_encoder=True,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"  Fold {fold}: loaded {ckpt_path}")

        sum_probs += get_probs(model, loader, device)

    avg_probs = sum_probs / n_folds
    labels    = avg_probs.argmax(axis=1).tolist()

    out_path = run_dir / "predictions.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "LABEL"])
        for row_id, label in enumerate(labels):
            writer.writerow([row_id, label])

    pos = sum(labels)
    print(f"\nPredictions saved: {out_path}")
    print(f"  Total            : {len(labels)}")
    print(f"  Label 1 (pos)    : {pos}  ({pos/len(labels):.1%})")
    print(f"  Label 0 (neg)    : {len(labels)-pos}  ({(len(labels)-pos)/len(labels):.1%})")


if __name__ == "__main__":
    main()
