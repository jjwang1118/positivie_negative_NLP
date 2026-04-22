"""
exp/exp03_shap_consistent/train.py
====================================
Improvement over exp02: warm-start Phase 3 from Phase 1 checkpoint +
CosineAnnealingLR for smoother convergence.

Phase 1 – Initial Training:
  Train AttentionPoolingClassifier on original texts using stratified 5-fold CV.
  Checkpoints saved to run_dir/phase1/fold{k}/best.pt.

Phase 2 – SHAP Extraction:
  For each fold's val set, load the Phase-1 checkpoint and run
  shap.PartitionExplainer to compute per-word Shapley values.
  Key string = words with positive SHAP value for the predicted class,
  in original sentence order.  Aggregated to run_dir/key_strings.json.

Phase 3 – Retrain on Filtered Texts (improved):
  - Warm-start: load Phase 1 best.pt of the same fold so attention/MLP
    weights are already primed instead of random-initialised.
  - CosineAnnealingLR: decays LR from initial value to 1% over all epochs,
    avoiding late-epoch oscillation.
  Checkpoints saved to run_dir/fold{k}/.

Parameters are loaded from config.yaml in the same directory.
Each execution creates a new numbered run directory:
  results/exp03/01/  results/exp03/02/  ...

Usage
-----
    python exp/exp03_shap_consistent/train.py
"""

import csv
import json
import random
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import shap
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


def merge(cfg: dict, *sections: str) -> dict:
    """Merge listed config sections into one flat dict."""
    out = {}
    for s in sections:
        if s in cfg and isinstance(cfg[s], dict):
            out.update(cfg[s])
    return out


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


FIELDNAMES = [
    "exp", "run_id", "timestamp", "mean_val_acc", "std_val_acc",
    "fold0_acc", "fold0_loss", "fold1_acc", "fold1_loss", "fold2_acc", "fold2_loss",
    "learning_rate", "batch_size", "epochs", "max_length", "seed", "n_folds",
    "model_name", "hidden_dim", "mlp_hidden", "dropout", "shap_max_evals",
]


def append_experiment_csv(csv_path: Path, entry: dict) -> None:
    cfg  = entry["config"]
    p2   = cfg.get("phase2", {})
    mo   = cfg.get("model", {})
    ext  = cfg.get("extraction", {})

    fold_data = {r["fold"]: r for r in entry["fold_results"]}
    exp_name  = entry.get("experiment", "")
    exp_short = exp_name.replace("exp", "") if exp_name.startswith("exp") else exp_name
    row = {
        "exp":           exp_short,
        "run_id":        entry["run_id"],
        "timestamp":     entry["timestamp"],
        "mean_val_acc":  entry["mean_val_acc"],
        "std_val_acc":   entry["std_val_acc"],
        "fold0_acc":     fold_data.get(0, {}).get("best_val_acc"),
        "fold0_loss":    fold_data.get(0, {}).get("best_val_loss"),
        "fold1_acc":     fold_data.get(1, {}).get("best_val_acc"),
        "fold1_loss":    fold_data.get(1, {}).get("best_val_loss"),
        "fold2_acc":     fold_data.get(2, {}).get("best_val_acc"),
        "fold2_loss":    fold_data.get(2, {}).get("best_val_loss"),
        "learning_rate": p2.get("learning_rate"),
        "batch_size":    p2.get("batch_size"),
        "epochs":        p2.get("epochs"),
        "max_length":    p2.get("max_length"),
        "seed":          p2.get("seed"),
        "n_folds":       p2.get("n_folds"),
        "model_name":    mo.get("name"),
        "hidden_dim":    mo.get("hidden_dim"),
        "mlp_hidden":    mo.get("mlp_hidden"),
        "dropout":       mo.get("dropout"),
        "shap_max_evals": ext.get("shap_max_evals"),
    }

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
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
# SHAP helpers
# ---------------------------------------------------------------------------

def make_predict_fn(model: AttentionPoolingClassifier, tokenizer, max_length: int, device: torch.device):
    @torch.no_grad()
    def predict(texts):
        enc = tokenizer(
            list(texts),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        )
        return torch.softmax(logits, dim=-1).cpu().numpy()
    return predict


def extract_shap_key_strings(
    fold: int,
    val_indices: list[int],
    all_texts: list[str],
    all_labels: list[int],
    model: AttentionPoolingClassifier,
    tokenizer,
    max_length: int,
    shap_max_evals: int,
    shap_label_idx: int | None,
    device: torch.device,
) -> list[dict]:
    """Return key_string records for the given val indices."""
    predict_fn = make_predict_fn(model, tokenizer, max_length, device)
    masker     = shap.maskers.Text(r"\W+")
    explainer  = shap.Explainer(predict_fn, masker, output_names=["negative", "positive"])

    records: list[dict] = []
    for i, global_idx in enumerate(val_indices):
        text  = all_texts[global_idx]
        label = all_labels[global_idx]

        stripped_words = [w for w in re.split(r"\W+", text) if w]
        if len(stripped_words) < 2:
            records.append({"fold": fold, "index": global_idx, "label": label, "key_string": " ".join(stripped_words)})
            continue

        sv        = explainer([text], max_evals=shap_max_evals)
        words     = list(sv.data[0])
        values_2d = sv.values[0]          # (n_words, 2)

        if shap_label_idx is not None:
            cls = shap_label_idx
        else:
            probs = predict_fn([text])[0]
            cls   = int(probs.argmax())

        word_shap  = values_2d[:, cls]
        key_set    = {idx for idx, v in enumerate(word_shap) if v > 0}
        key_string = " ".join(words[idx] for idx in sorted(key_set))

        records.append({
            "fold":       fold,
            "index":      global_idx,
            "label":      label,
            "key_string": key_string,
        })

        if (i + 1) % 10 == 0:
            print(f"    SHAP processed {i + 1}/{len(val_indices)}")

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg    = load_config(EXP_DIR / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["phase1"]["seed"])

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

    pretrained = MODEL_REGISTRY[cfg["model"]["name"]]
    tokenizer  = AutoTokenizer.from_pretrained(pretrained)

    hp1     = merge(cfg, "model", "phase1")
    hp_ext  = cfg.get("extraction", {})
    shap_max_evals = hp_ext.get("shap_max_evals", 500)
    shap_label_idx = hp_ext.get("shap_label_idx", None)

    n_folds = hp1["n_folds"]
    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=hp1["seed"])
    splits  = list(skf.split(all_texts, label_arr))

    # -----------------------------------------------------------------------
    # PHASE 1 — Initial training on original texts
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 1 — Initial training on original texts")
    print(f"{'='*60}")

    phase1_dir = run_dir / "phase1"
    phase1_dir.mkdir(parents=True, exist_ok=True)
    all_enc1 = tokenise(all_texts, tokenizer, hp1["max_length"])

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n  Fold {fold + 1}/{n_folds}  train={len(train_idx)}  val={len(val_idx)}")
        fold_dir = phase1_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_ds = SentimentDataset(
            subset_encodings(all_enc1, train_idx.tolist()),
            label_arr[train_idx].tolist(),
        )
        val_ds = SentimentDataset(
            subset_encodings(all_enc1, val_idx.tolist()),
            label_arr[val_idx].tolist(),
        )
        train_loader = DataLoader(train_ds, batch_size=hp1["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=hp1["batch_size"], shuffle=False)

        model = AttentionPoolingClassifier(
            pretrained_name=pretrained,
            hidden_dim=hp1["hidden_dim"],
            mlp_hidden=hp1["mlp_hidden"],
            dropout=hp1["dropout"],
            num_labels=hp1["num_labels"],
            freeze_encoder=True,
        ).to(device)

        optimizer    = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hp1["learning_rate"])
        criterion    = nn.CrossEntropyLoss()
        best_val_acc = 0.0

        for epoch in range(1, hp1["epochs"] + 1):
            train_loss        = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(
                f"    Epoch {epoch:02d}/{hp1['epochs']}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), fold_dir / "best.pt")

        torch.save(model.state_dict(), fold_dir / "last.pt")
        print(f"    Phase-1 fold {fold} best val_acc: {best_val_acc:.4f}")

    # -----------------------------------------------------------------------
    # PHASE 2 — SHAP extraction
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 2 — SHAP key string extraction")
    print(f"{'='*60}")
    print(f"  shap_max_evals={shap_max_evals}  shap_label_idx={shap_label_idx}")

    all_ks_records: list[dict] = []

    for fold, (_, val_idx) in enumerate(splits):
        print(f"\n  Fold {fold + 1}/{n_folds}  val size={len(val_idx)}")
        ckpt_path = phase1_dir / f"fold{fold}" / "best.pt"

        model = AttentionPoolingClassifier(
            pretrained_name=pretrained,
            hidden_dim=hp1["hidden_dim"],
            mlp_hidden=hp1["mlp_hidden"],
            dropout=hp1["dropout"],
            num_labels=hp1["num_labels"],
            freeze_encoder=True,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        records = extract_shap_key_strings(
            fold, val_idx.tolist(), all_texts, all_labels,
            model, tokenizer, hp1["max_length"],
            shap_max_evals, shap_label_idx, device,
        )
        all_ks_records.extend(records)
        print(f"    {len(records)} key strings extracted for fold {fold}")

    ks_path = run_dir / "key_strings.json"
    with open(ks_path, "w", encoding="utf-8") as f:
        json.dump(all_ks_records, f, ensure_ascii=False, indent=2)
    print(f"\n  Key strings saved: {ks_path}")

    ks_map = {r["index"]: r["key_string"].strip() for r in all_ks_records}

    # -----------------------------------------------------------------------
    # PHASE 3 — Retrain on SHAP-filtered texts (warm-start + CosineAnnealingLR)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 3 — Retrain on SHAP-filtered texts (warm-start + CosineAnnealingLR)")
    print(f"{'='*60}")

    hp2 = merge(cfg, "model", "phase2")
    filtered_texts = [ks_map.get(i) or all_texts[i] for i in range(len(all_texts))]
    n_replaced     = sum(1 for i in range(len(all_texts)) if ks_map.get(i))
    print(f"  Replaced texts: {n_replaced} / {len(all_texts)}")

    all_enc2 = tokenise(filtered_texts, tokenizer, hp2["max_length"])

    skf2           = StratifiedKFold(n_splits=hp2["n_folds"], shuffle=True, random_state=hp2["seed"])
    fold_summaries: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(skf2.split(filtered_texts, label_arr)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{hp2['n_folds']}  |  train={len(train_idx)}  val={len(val_idx)}")

        fold_dir = run_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_ds = SentimentDataset(
            subset_encodings(all_enc2, train_idx.tolist()),
            label_arr[train_idx].tolist(),
        )
        val_ds = SentimentDataset(
            subset_encodings(all_enc2, val_idx.tolist()),
            label_arr[val_idx].tolist(),
        )
        train_loader = DataLoader(train_ds, batch_size=hp2["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=hp2["batch_size"], shuffle=False)

        model = AttentionPoolingClassifier(
            pretrained_name=pretrained,
            hidden_dim=hp2["hidden_dim"],
            mlp_hidden=hp2["mlp_hidden"],
            dropout=hp2["dropout"],
            num_labels=hp2["num_labels"],
            freeze_encoder=True,
        ).to(device)

        # Warm-start: load Phase 1 checkpoint so attention/MLP start from a
        # converged state rather than random initialisation.
        phase1_ckpt = phase1_dir / f"fold{fold}" / "best.pt"
        if phase1_ckpt.exists():
            model.load_state_dict(torch.load(phase1_ckpt, map_location=device, weights_only=True))
            print(f"  Warm-start from Phase 1 fold {fold} checkpoint")

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=hp2["learning_rate"],
        )
        # CosineAnnealingLR decays LR smoothly from lr → lr*0.01 over all epochs.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=hp2["epochs"], eta_min=hp2["learning_rate"] * 0.01
        )
        criterion = nn.CrossEntropyLoss()

        best_val_acc  = 0.0
        best_val_loss = float("inf")
        log_rows: list[dict] = []

        for epoch in range(1, hp2["epochs"] + 1):
            train_loss        = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            log_rows.append({
                "epoch":      epoch,
                "train_loss": f"{train_loss:.4f}",
                "val_loss":   f"{val_loss:.4f}",
                "val_acc":    f"{val_acc:.4f}",
            })
            print(
                f"  Epoch {epoch:02d}/{hp2['epochs']}  "
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
