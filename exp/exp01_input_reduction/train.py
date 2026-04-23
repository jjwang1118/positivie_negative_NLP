"""
exp/exp01_input_reduction/train.py
===================================
3-fold stratified cross-validation training for Exp01.

Parameters are loaded from config.yaml in the same directory.
Each execution creates a new numbered run directory:
  results/exp01/01/  results/exp01/02/  ...

Each run saves:
  {run_dir}/config.yaml          snapshot of config used
  {run_dir}/fold{k}/best.pt
  {run_dir}/fold{k}/last.pt
  {run_dir}/fold{k}/train_log.csv
  {run_dir}/cv_summary.json
  results/exp01/experiment_log.json  (appended)

Usage
-----
    python exp/exp01_input_reduction/train.py
"""

import csv
import json
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
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR      = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_REGISTRY, RESULTS_DIR, TRAIN_FILE
from src.model import AttentionPoolingClassifier


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_config(cfg: dict) -> dict:
    """Merge nested config sections into a flat dict for convenience."""
    flat = {}
    for section in cfg.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat


# ---------------------------------------------------------------------------
# Run numbering
# ---------------------------------------------------------------------------

def next_run_dir(exp_results_dir: Path) -> tuple[Path, str]:
    """Return (run_path, run_id) for the next numbered run (01, 02, ...)."""
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
    """Append one run entry to experiment_log.json (cumulative)."""
    log = []
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
    log.append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


FIELDNAMES = [
    "exp", "run_id", "timestamp", "mean_val_acc", "std_val_acc",
    "fold0_acc", "fold0_loss", "fold1_acc", "fold1_loss",
    "fold2_acc", "fold2_loss", "fold3_acc", "fold3_loss",
    "fold4_acc", "fold4_loss",
    "learning_rate", "batch_size", "epochs", "max_length", "seed", "n_folds",
    "model_name", "hidden_dim", "mlp_hidden", "dropout",
    "shap_max_evals",
    "lambda_u", "confidence_threshold", "tsa_schedule", "augment_prob",
]


def append_experiment_csv(csv_path: Path, entry: dict) -> None:
    """Append one row per run to experiment_log.csv (one row = one run summary)."""
    cfg = entry["config"]
    tr  = cfg.get("training", {})
    mo  = cfg.get("model", {})
    ex  = cfg.get("extraction", {})

    fold_data = {r["fold"]: r for r in entry["fold_results"]}
    row = {
        "exp":                  entry.get("experiment"),
        "run_id":               entry["run_id"],
        "timestamp":            entry["timestamp"],
        "mean_val_acc":         entry["mean_val_acc"],
        "std_val_acc":          entry["std_val_acc"],
        "fold0_acc":            fold_data.get(0, {}).get("best_val_acc"),
        "fold0_loss":           fold_data.get(0, {}).get("best_val_loss"),
        "fold1_acc":            fold_data.get(1, {}).get("best_val_acc"),
        "fold1_loss":           fold_data.get(1, {}).get("best_val_loss"),
        "fold2_acc":            fold_data.get(2, {}).get("best_val_acc"),
        "fold2_loss":           fold_data.get(2, {}).get("best_val_loss"),
        "fold3_acc":            fold_data.get(3, {}).get("best_val_acc"),
        "fold3_loss":           fold_data.get(3, {}).get("best_val_loss"),
        "fold4_acc":            fold_data.get(4, {}).get("best_val_acc"),
        "fold4_loss":           fold_data.get(4, {}).get("best_val_loss"),
        "learning_rate":        tr.get("learning_rate"),
        "batch_size":           tr.get("batch_size"),
        "epochs":               tr.get("epochs"),
        "max_length":           tr.get("max_length"),
        "seed":                 tr.get("seed"),
        "n_folds":              tr.get("n_folds"),
        "model_name":           mo.get("name"),
        "hidden_dim":           mo.get("hidden_dim"),
        "mlp_hidden":           mo.get("mlp_hidden"),
        "dropout":              mo.get("dropout"),
        "shap_max_evals":       ex.get("shap_max_evals"),
        "lambda_u":             "",
        "confidence_threshold": "",
        "tsa_schedule":         "",
        "augment_prob":         "",
    }

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


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
    """Load train_2022.csv with columns: row_id, TEXT, LABEL."""
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

def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        optimizer.zero_grad()
        loss = criterion(model(input_ids=input_ids, attention_mask=attention_mask), labels)
        loss.backward()
        optimizer.step()
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
    n_folds = hp["n_folds"]
    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=hp["seed"])
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

        model = AttentionPoolingClassifier(
            pretrained_name=pretrained,
            hidden_dim=hp["hidden_dim"],
            mlp_hidden=hp["mlp_hidden"],
            dropout=hp["dropout"],
            num_labels=hp["num_labels"],
            freeze_encoder=True,
        ).to(device)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=hp["learning_rate"],
        )
        criterion = nn.CrossEntropyLoss()

        best_val_acc  = 0.0
        best_val_loss = float("inf")
        log_rows: list[dict] = []

        for epoch in range(1, hp["epochs"] + 1):
            train_loss        = train_epoch(model, train_loader, optimizer, criterion, device)
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
        "experiment":  exp_name,
        "run_id":      run_id,
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "config":      cfg,
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
    print(f"  Exp log      : {exp_results_dir / 'experiment_log.json'}")


if __name__ == "__main__":
    main()
