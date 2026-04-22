"""
exp/exp03_finetune/train.py
============================
Full fine-tune of RoBERTa-base for maximum sentiment classification accuracy.

Architecture: roberta-base → [CLS] pooling → Dropout → Linear(768→2)
All parameters (encoder + head) are trained end-to-end.
A linear warmup scheduler is applied for stable training.

Parameters are loaded from config.yaml in the same directory.
Each execution creates a new numbered run directory:
  results/exp03/01/  results/exp03/02/  ...

Usage
-----
    python exp/exp03_finetune/train.py
"""

import csv
import json
import math
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR      = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_REGISTRY, RESULTS_DIR, TRAIN_FILE
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
# Run numbering
# ---------------------------------------------------------------------------

def next_run_dir(exp_results_dir: Path) -> tuple[Path, str]:
    exp_results_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        [d.name for d in exp_results_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=int,
    )
    run_id  = f"{int(existing[-1]) + 1:02d}" if existing else "01"
    run_dir = exp_results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id


# ---------------------------------------------------------------------------
# Experiment log
# ---------------------------------------------------------------------------

def append_experiment_log(log_path: Path, entry: dict) -> None:
    log = []
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
    log.append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def append_experiment_csv(csv_path: Path, entry: dict) -> None:
    cfg = entry["config"]
    tr  = cfg.get("training", {})
    mo  = cfg.get("model", {})

    row = {
        "experiment":    entry.get("experiment"),
        "run_id":        entry["run_id"],
        "timestamp":     entry["timestamp"],
        "mean_val_acc":  entry["mean_val_acc"],
        "std_val_acc":   entry["std_val_acc"],
    }
    for fold_res in entry["fold_results"]:
        k = fold_res["fold"]
        row[f"fold{k}_acc"]  = fold_res["best_val_acc"]
        row[f"fold{k}_loss"] = fold_res["best_val_loss"]

    row.update({
        "learning_rate": tr.get("learning_rate"),
        "batch_size":    tr.get("batch_size"),
        "epochs":        tr.get("epochs"),
        "max_length":    tr.get("max_length"),
        "seed":          tr.get("seed"),
        "n_folds":       tr.get("n_folds"),
        "warmup_ratio":  tr.get("warmup_ratio"),
        "model_name":    mo.get("name"),
        "hidden_dim":    mo.get("hidden_dim"),
        "dropout":       mo.get("dropout"),
    })

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SentimentDataset(Dataset):
    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["TEXT"].strip())
            labels.append(int(row["LABEL"].strip()))
    return texts, labels


def tokenise(texts: list[str], tokenizer, max_length: int) -> dict:
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def subset_encodings(encodings: dict, indices: list[int]) -> dict:
    return {k: v[indices] for k, v in encodings.items()}


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        optimizer.zero_grad()
        loss = criterion(model(input_ids=input_ids, attention_mask=attention_mask), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        logits         = model(input_ids=input_ids, attention_mask=attention_mask)
        total_loss    += criterion(logits, labels).item()
        correct       += (logits.argmax(dim=-1) == labels).sum().item()
        total         += labels.size(0)
    return total_loss / len(loader), correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg    = load_config(EXP_DIR / "config.yaml")
    hp     = flatten_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(hp["seed"])

    exp_name        = cfg["experiment"]["name"]
    exp_results_dir = RESULTS_DIR / exp_name
    run_dir, run_id = next_run_dir(exp_results_dir)

    print(f"Experiment : {exp_name}")
    print(f"Run        : {run_id}")
    print(f"Run dir    : {run_dir}")
    print(f"Device     : {device}")

    shutil.copy(EXP_DIR / "config.yaml", run_dir / "config.yaml")

    # --- Load data ---
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
    all_texts, all_labels = load_csv(TRAIN_FILE)
    label_arr = np.array(all_labels)
    print(f"Total samples: {len(all_texts)}")

    # --- Tokenise once ---
    pretrained = MODEL_REGISTRY[hp["name"]]
    tokenizer  = AutoTokenizer.from_pretrained(pretrained)
    all_enc    = tokenise(all_texts, tokenizer, hp["max_length"])

    # --- Cross-validation ---
    n_folds      = hp["n_folds"]
    warmup_ratio = hp.get("warmup_ratio", 0.06)
    skf          = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=hp["seed"])
    fold_summaries: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_texts, label_arr)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_folds}  |  train={len(train_idx)}  val={len(val_idx)}")

        fold_dir = run_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_ds = SentimentDataset(
            subset_encodings(all_enc, train_idx.tolist()),
            label_arr[train_idx].tolist(),
        )
        val_ds = SentimentDataset(
            subset_encodings(all_enc, val_idx.tolist()),
            label_arr[val_idx].tolist(),
        )
        train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=hp["batch_size"], shuffle=False)

        model = SequenceClassifier(
            pretrained_name=pretrained,
            hidden_dim=hp["hidden_dim"],
            dropout=hp["dropout"],
            num_labels=hp["num_labels"],
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=0.01)

        total_steps  = hp["epochs"] * math.ceil(len(train_ds) / hp["batch_size"])
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler    = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        criterion = nn.CrossEntropyLoss()

        best_val_acc  = 0.0
        best_val_loss = float("inf")
        log_rows: list[dict] = []

        print(f"  Warmup steps: {warmup_steps} / {total_steps} total")

        for epoch in range(1, hp["epochs"] + 1):
            train_loss        = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            log_rows.append({
                "epoch":      epoch,
                "train_loss": f"{train_loss:.4f}",
                "val_loss":   f"{val_loss:.4f}",
                "val_acc":    f"{val_acc:.4f}",
            })
            print(
                f"  Epoch {epoch:02d}/{hp['epochs']}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc  = val_acc
                best_val_loss = val_loss
                torch.save(model.state_dict(), fold_dir / "best.pt")
                print(f"    -> saved best (val_acc={best_val_acc:.4f})")

        torch.save(model.state_dict(), fold_dir / "last.pt")

        with open(fold_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc"])
            writer.writeheader()
            writer.writerows(log_rows)

        fold_summaries.append({
            "fold":          fold,
            "best_val_acc":  round(best_val_acc, 4),
            "best_val_loss": round(best_val_loss, 4),
        })

    # --- CV summary ---
    accs = [s["best_val_acc"] for s in fold_summaries]
    cv_summary = {
        "experiment":   exp_name,
        "run_id":       run_id,
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "config":       cfg,
        "fold_results": fold_summaries,
        "mean_val_acc": round(float(np.mean(accs)), 4),
        "std_val_acc":  round(float(np.std(accs)),  4),
    }

    with open(run_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, ensure_ascii=False, indent=2)

    append_experiment_log(exp_results_dir / "experiment_log.json", cv_summary)
    append_experiment_csv(RESULTS_DIR / "experiment_log.csv",      cv_summary)

    print(f"\n{'='*50}")
    print(f"Cross-Validation Summary  (run {run_id})")
    for s in fold_summaries:
        print(f"  Fold {s['fold']}  best_val_acc={s['best_val_acc']:.4f}  best_val_loss={s['best_val_loss']:.4f}")
    print(f"  Mean val_acc : {cv_summary['mean_val_acc']:.4f}")
    print(f"  Std  val_acc : {cv_summary['std_val_acc']:.4f}")
    print(f"  Run dir      : {run_dir}")


if __name__ == "__main__":
    main()
