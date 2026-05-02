"""
exp/exp11_pseudo_single_stage/predict.py

Re-generates predictions.csv from saved best_model checkpoints without retraining.

Usage
-----
    python exp/exp11_pseudo_single_stage/predict.py            # latest run
    python exp/exp11_pseudo_single_stage/predict.py --run 01
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    import regex as re
except ImportError:
    import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR      = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_REGISTRY, RESULTS_DIR, TEST_FILE


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"(\w+)\s+t\b",  r"\1not",  text)
    text = re.sub(r"i\s+m\b",       "i am",    text)
    text = re.sub(r"it\s+s\b",      "it is",   text)
    text = re.sub(r"(\w+)\s+s\b",  r"\1s",     text)
    text = re.sub(r"(\w+)\s+re\b", r"\1 are",  text)
    text = re.sub(r"(\w+)\s+ve\b", r"\1 have", text)
    text = re.sub(r"(\w+)\s+ll\b", r"\1 will", text)
    text = re.sub(r"(.)\1{2,}",    r"\1\1",    text)
    text = text.replace("num_num", "number")
    return re.sub(r"\s+", " ", text).strip()


def read_test(path: Path) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(row["row_id"].strip())
            texts.append(clean_text(row["TEXT"].strip()))
    return ids, texts


def resolve_run_dir(exp_dir: Path, run_id: str | None) -> tuple[Path, str]:
    if run_id is not None:
        d = exp_dir / run_id
        if not d.exists():
            raise FileNotFoundError(f"Run not found: {d}")
        return d, run_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"No results at {exp_dir} — run train.py first.")
    ids = sorted([d.name for d in exp_dir.iterdir() if d.is_dir() and d.name.isdigit()], key=int)
    if not ids:
        raise FileNotFoundError(f"No runs under {exp_dir}")
    return exp_dir / ids[-1], ids[-1]


@torch.no_grad()
def infer(
    model: AutoModelForSequenceClassification,
    texts: list[str],
    tokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        enc    = tokenizer(texts[i:i + batch_size], truncation=True,
                           max_length=max_length, padding=True, return_tensors="pt")
        logits = model(input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device)).logits
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(probs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    args = parser.parse_args()

    cfg    = load_config(EXP_DIR / "config.yaml")
    tr     = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_results_dir = RESULTS_DIR / cfg["experiment"]["name"]
    run_dir, run_id = resolve_run_dir(exp_results_dir, args.run)

    print(f"Experiment : {cfg['experiment']['name']}  run={run_id}")
    print(f"Device     : {device}")

    pretrained       = MODEL_REGISTRY[cfg["model"]["name"]]
    tokenizer        = AutoTokenizer.from_pretrained(pretrained)
    test_ids, texts  = read_test(TEST_FILE)
    batch_size       = tr["batch_size"] * 2

    print(f"Test samples : {len(texts)}")

    all_probs: list[np.ndarray] = []
    for fold in range(1, tr["n_folds"] + 1):
        model_dir = run_dir / f"fold{fold}" / "best_model"
        if not model_dir.exists():
            raise FileNotFoundError(f"{model_dir} not found — run train.py first.")
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
        all_probs.append(infer(model, texts, tokenizer, tr["max_length"], batch_size, device))
        print(f"  fold {fold} done")
        del model
        torch.cuda.empty_cache()

    preds     = np.mean(all_probs, axis=0).argmax(axis=1).tolist()
    out_path  = run_dir / "predictions.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "LABEL"])
        w.writerows(zip(test_ids, preds))

    pos = sum(preds)
    print(f"\nSaved: {out_path}")
    print(f"  pos={pos} ({pos/len(preds):.1%})  neg={len(preds)-pos} ({(len(preds)-pos)/len(preds):.1%})")


if __name__ == "__main__":
    main()
