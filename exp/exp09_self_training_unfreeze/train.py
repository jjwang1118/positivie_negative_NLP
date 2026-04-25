"""
exp/exp09_self_training_unfreeze/train.py
==========================================
Direction A (partial encoder unfreeze) + Direction B (self-training pseudo-labels).

Per-fold pipeline
-----------------
Stage 1  Frozen encoder, labeled data only (1,600 samples).
         → model confidence on unlabeled test → pseudo_pool_S1 (threshold 0.95)

Stage 2  Unfreeze last N RoBERTa layers (layerwise LR).
         Warm-start from Stage 1 head weights.
         Train on labeled + pseudo_pool_S1.
         → model confidence → pseudo_pool_S2 (threshold 0.90)

Stage 3  (optional) Continue unfreezing, train on labeled + pseudo_pool_S2.
         LR further reduced to prevent drift.

best.pt per fold = final stage's best checkpoint (by val_acc).
Prediction uses 5-fold ensemble of best.pt.

Usage
-----
    python exp/exp09_self_training_unfreeze/train.py
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR      = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_REGISTRY, RESULTS_DIR, TEST_FILE, TRAIN_FILE
from src.model import MeanPoolingClassifier


# ---------------------------------------------------------------------------
# Config / run helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


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
# Experiment log helpers
# ---------------------------------------------------------------------------

def append_experiment_log(log_path: Path, entry: dict) -> None:
    log = []
    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
    log.append(entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


# Reuse exp08's FIELDNAMES schema for global CSV compatibility.
# Exp09-specific fields are mapped as follows:
#   learning_rate      → stage2 head_lr
#   n_clusters         → unfreeze_last_n
#   purity_threshold   → stage1 confidence_threshold
#   min_labeled…       → stage2 confidence_threshold
#   mean_pseudo_labels → mean of (mean_pseudo_s1, mean_pseudo_s2)
FIELDNAMES = [
    "exp", "run_id", "timestamp", "mean_val_acc", "std_val_acc",
    "fold0_acc", "fold0_loss", "fold1_acc", "fold1_loss",
    "fold2_acc", "fold2_loss", "fold3_acc", "fold3_loss",
    "fold4_acc", "fold4_loss",
    "learning_rate", "batch_size", "epochs", "max_length", "seed", "n_folds",
    "model_name", "hidden_dim", "mlp_hidden", "dropout",
    "n_clusters", "purity_threshold", "min_labeled_per_cluster",
    "mean_pseudo_labels", "label_smoothing", "warmup_ratio",
]


def append_experiment_csv(csv_path: Path, entry: dict) -> None:
    cfg  = entry["config"]
    s1   = cfg["stage1"]
    s2   = cfg["stage2"]
    mo   = cfg["model"]

    fold_data = {r["fold"]: r for r in entry["fold_results"]}
    mean_pseudo = round(
        float(np.mean([entry.get("mean_pseudo_s1", 0), entry.get("mean_pseudo_s2", 0)])), 1
    )

    row = {
        "exp":                     entry.get("experiment", ""),
        "run_id":                  entry["run_id"],
        "timestamp":               entry["timestamp"],
        "mean_val_acc":            entry["mean_val_acc"],
        "std_val_acc":             entry["std_val_acc"],
        "fold0_acc":               fold_data.get(0, {}).get("best_val_acc"),
        "fold0_loss":              fold_data.get(0, {}).get("best_val_loss"),
        "fold1_acc":               fold_data.get(1, {}).get("best_val_acc"),
        "fold1_loss":              fold_data.get(1, {}).get("best_val_loss"),
        "fold2_acc":               fold_data.get(2, {}).get("best_val_acc"),
        "fold2_loss":              fold_data.get(2, {}).get("best_val_loss"),
        "fold3_acc":               fold_data.get(3, {}).get("best_val_acc"),
        "fold3_loss":              fold_data.get(3, {}).get("best_val_loss"),
        "fold4_acc":               fold_data.get(4, {}).get("best_val_acc"),
        "fold4_loss":              fold_data.get(4, {}).get("best_val_loss"),
        "learning_rate":           s2["head_lr"],
        "batch_size":              s2["batch_size"],
        "epochs":                  s1["epochs"] + s2["epochs"],
        "max_length":              s1["max_length"],
        "seed":                    s1["seed"],
        "n_folds":                 s1["n_folds"],
        "model_name":              mo["name"],
        "hidden_dim":              mo["hidden_dim"],
        "mlp_hidden":              mo["mlp_hidden"],
        "dropout":                 mo["dropout"],
        "n_clusters":              s2["unfreeze_last_n"],
        "purity_threshold":        s1["confidence_threshold"],
        "min_labeled_per_cluster": s2["confidence_threshold"],
        "mean_pseudo_labels":      mean_pseudo,
        "label_smoothing":         s2["label_smoothing"],
        "warmup_ratio":            s2["warmup_ratio"],
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
# Data loading
# ---------------------------------------------------------------------------

def load_labeled_csv(path: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["TEXT"].strip())
            labels.append(int(row["LABEL"].strip()))
    return texts, labels


def load_unlabeled_csv(path: Path) -> tuple[list[str], list[str]]:
    row_ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row_ids.append(row["row_id"].strip())
            texts.append(row["TEXT"].strip())
    return row_ids, texts


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class LabeledDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.texts     = texts
        self.labels    = labels

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
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


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


# ---------------------------------------------------------------------------
# Encoder unfreeze helpers
# ---------------------------------------------------------------------------

def unfreeze_last_n_layers(encoder: nn.Module, n: int) -> None:
    """Unfreeze the last N transformer layers of a RoBERTa-family encoder.

    Assumes encoder.encoder.layer is a ModuleList of transformer blocks
    (standard HuggingFace RoBERTa architecture).
    """
    transformer_layers = encoder.encoder.layer   # ModuleList[0..23] for RoBERTa-large
    for layer in transformer_layers[-n:]:
        for p in layer.parameters():
            p.requires_grad = True


def build_layerwise_optimizer(
    model: MeanPoolingClassifier,
    encoder_lr: float,
    head_lr: float,
) -> AdamW:
    """Separate LR for (unfrozen) encoder params vs classifier head."""
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params    = list(model.classifier.parameters())
    param_groups   = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr})
    param_groups.append({"params": head_params, "lr": head_lr})
    return AdamW(param_groups)


# ---------------------------------------------------------------------------
# Confidence-based pseudo-label generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_confident_pseudolabels(
    model: nn.Module,
    texts: list[str],
    tokenizer,
    max_len: int,
    batch_size: int,
    device: torch.device,
    threshold: float,
) -> tuple[list[str], list[int]]:
    """Return (texts, labels) for unlabeled samples where max softmax >= threshold."""
    model.eval()
    ds     = InferenceDataset(texts, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_probs: list[torch.Tensor] = []
    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        all_probs.append(torch.softmax(model(ids, mask), dim=-1).cpu())

    probs      = torch.cat(all_probs, dim=0)   # [N, 2]
    confidence, pred_labels = probs.max(dim=1)
    indices    = (confidence >= threshold).nonzero(as_tuple=True)[0].tolist()

    return [texts[i] for i in indices], pred_labels[indices].tolist()


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    scheduler=None,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(ids, mask)
        total_loss += criterion(logits, labels).item()
        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    val_loss = total_loss / len(loader)
    val_acc  = accuracy_score(all_labels, all_preds)
    val_f1   = f1_score(all_labels, all_preds, average="macro")
    return val_loss, val_acc, val_f1


def train_stage(
    model: nn.Module,
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    tokenizer,
    device: torch.device,
    optimizer: AdamW,
    criterion: nn.CrossEntropyLoss,
    epochs: int,
    batch_size: int,
    max_length: int,
    ckpt_path: Path,
    log_path: Path,
    stage_name: str = "",
) -> tuple[float, float]:
    """Train for N epochs, save best checkpoint, write per-epoch log."""
    train_ds = LabeledDataset(train_texts, train_labels, tokenizer, max_length)
    val_ds   = LabeledDataset(val_texts, val_labels, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    total_steps  = len(train_loader) * epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_acc  = 0.0
    best_val_loss = float("inf")
    log_rows: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        log_rows.append({
            "epoch":      epoch,
            "train_loss": f"{train_loss:.4f}",
            "val_loss":   f"{val_loss:.4f}",
            "val_acc":    f"{val_acc:.4f}",
            "val_f1":     f"{val_f1:.4f}",
            "lr":         f"{current_lr:.2e}",
        })
        print(
            f"    [{stage_name}] ep {epoch:02d}/{epochs}  "
            f"tr={train_loss:.4f}  vl={val_loss:.4f}  acc={val_acc:.4f}  "
            f"f1={val_f1:.4f}  lr={current_lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"      → best saved (acc={best_val_acc:.4f})")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "val_f1", "lr"]
        )
        writer.writeheader()
        writer.writerows(log_rows)

    return best_val_acc, best_val_loss


# ---------------------------------------------------------------------------
# Per-fold 3-stage pipeline
# ---------------------------------------------------------------------------

def run_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    labeled_texts: list[str],
    labeled_labels: list[int],
    unlabeled_texts: list[str],
    pretrained: str,
    tokenizer,
    device: torch.device,
    cfg: dict,
    run_dir: Path,
) -> dict:
    mo   = cfg["model"]
    s1   = cfg["stage1"]
    s2   = cfg["stage2"]
    s3   = cfg.get("stage3", {"enabled": False})

    fold_dir = run_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_texts_l  = [labeled_texts[i]  for i in train_idx]
    train_labels_l = [labeled_labels[i] for i in train_idx]
    val_texts      = [labeled_texts[i]  for i in val_idx]
    val_labels     = [labeled_labels[i] for i in val_idx]

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"Fold {fold + 1}  |  labeled train={len(train_texts_l)}  val={len(val_texts)}")
    print(sep)

    # ── Stage 1: Frozen encoder, labeled data only ────────────────────────
    print(f"\n  [Stage 1] frozen encoder | {len(train_texts_l)} samples")

    model_s1 = MeanPoolingClassifier(
        pretrained_name=pretrained,
        hidden_dim=mo["hidden_dim"],
        mlp_hidden=mo["mlp_hidden"],
        dropout=mo["dropout"],
        num_labels=mo["num_labels"],
        freeze_encoder=True,
    ).to(device)

    optimizer_s1 = AdamW(
        filter(lambda p: p.requires_grad, model_s1.parameters()),
        lr=s1["learning_rate"],
    )
    criterion_s1 = nn.CrossEntropyLoss(label_smoothing=s1["label_smoothing"])

    best_s1_acc, best_s1_loss = train_stage(
        model=model_s1,
        train_texts=train_texts_l, train_labels=train_labels_l,
        val_texts=val_texts, val_labels=val_labels,
        tokenizer=tokenizer, device=device,
        optimizer=optimizer_s1, criterion=criterion_s1,
        epochs=s1["epochs"], batch_size=s1["batch_size"], max_length=s1["max_length"],
        ckpt_path=fold_dir / "stage1_best.pt",
        log_path=fold_dir / "stage1_log.csv",
        stage_name="S1",
    )

    # Generate pseudo-labels via model confidence (threshold S1)
    model_s1.load_state_dict(
        torch.load(fold_dir / "stage1_best.pt", map_location=device, weights_only=True)
    )
    pseudo_texts_s1, pseudo_labels_s1 = get_confident_pseudolabels(
        model_s1, unlabeled_texts, tokenizer,
        s1["max_length"], s1["batch_size"], device,
        threshold=s1["confidence_threshold"],
    )
    n_pseudo_s1 = len(pseudo_texts_s1)
    print(
        f"\n  Stage 1 done  best_val_acc={best_s1_acc:.4f}  "
        f"pseudo_S1={n_pseudo_s1}/{len(unlabeled_texts)} "
        f"(threshold={s1['confidence_threshold']})"
    )

    # Free Stage 1 model from GPU before building Stage 2
    del model_s1, optimizer_s1, criterion_s1
    torch.cuda.empty_cache()

    # ── Stage 2: Partial unfreeze + S1 pseudo-labels ──────────────────────
    train_texts_s2  = train_texts_l  + pseudo_texts_s1
    train_labels_s2 = train_labels_l + pseudo_labels_s1
    n_unfreeze = s2["unfreeze_last_n"]
    print(
        f"\n  [Stage 2] unfreeze last {n_unfreeze} layers | "
        f"{len(train_texts_s2)} samples  "
        f"(encoder_lr={s2['encoder_lr']:.1e}  head_lr={s2['head_lr']:.1e})"
    )

    model_s2 = MeanPoolingClassifier(
        pretrained_name=pretrained,
        hidden_dim=mo["hidden_dim"],
        mlp_hidden=mo["mlp_hidden"],
        dropout=mo["dropout"],
        num_labels=mo["num_labels"],
        freeze_encoder=True,       # start frozen, then selectively unfreeze
    ).to(device)

    # Warm-start head weights from Stage 1; encoder last N layers become trainable
    model_s2.load_state_dict(
        torch.load(fold_dir / "stage1_best.pt", map_location=device, weights_only=True)
    )
    unfreeze_last_n_layers(model_s2.encoder, n_unfreeze)

    optimizer_s2 = build_layerwise_optimizer(model_s2, s2["encoder_lr"], s2["head_lr"])
    criterion_s2 = nn.CrossEntropyLoss(label_smoothing=s2["label_smoothing"])

    best_s2_acc, best_s2_loss = train_stage(
        model=model_s2,
        train_texts=train_texts_s2, train_labels=train_labels_s2,
        val_texts=val_texts, val_labels=val_labels,
        tokenizer=tokenizer, device=device,
        optimizer=optimizer_s2, criterion=criterion_s2,
        epochs=s2["epochs"], batch_size=s2["batch_size"], max_length=s2["max_length"],
        ckpt_path=fold_dir / "stage2_best.pt",
        log_path=fold_dir / "stage2_log.csv",
        stage_name="S2",
    )

    final_best_acc  = best_s2_acc
    final_best_loss = best_s2_loss
    n_pseudo_s2     = 0

    # ── Stage 3 (optional): S2 model → refine with S2 pseudo-labels ───────
    if s3.get("enabled", False):
        model_s2.load_state_dict(
            torch.load(fold_dir / "stage2_best.pt", map_location=device, weights_only=True)
        )
        pseudo_texts_s2, pseudo_labels_s2 = get_confident_pseudolabels(
            model_s2, unlabeled_texts, tokenizer,
            s2["max_length"], s2["batch_size"], device,
            threshold=s2["confidence_threshold"],
        )
        n_pseudo_s2 = len(pseudo_texts_s2)
        print(
            f"\n  Stage 2 done  best_val_acc={best_s2_acc:.4f}  "
            f"pseudo_S2={n_pseudo_s2}/{len(unlabeled_texts)} "
            f"(threshold={s2['confidence_threshold']})"
        )

        # Free Stage 2 model from GPU before building Stage 3
        del model_s2, optimizer_s2, criterion_s2
        torch.cuda.empty_cache()

        train_texts_s3  = train_texts_l  + pseudo_texts_s2
        train_labels_s3 = train_labels_l + pseudo_labels_s2
        print(
            f"\n  [Stage 3] | {len(train_texts_s3)} samples  "
            f"(encoder_lr={s3['encoder_lr']:.1e}  head_lr={s3['head_lr']:.1e})"
        )

        model_s3 = MeanPoolingClassifier(
            pretrained_name=pretrained,
            hidden_dim=mo["hidden_dim"],
            mlp_hidden=mo["mlp_hidden"],
            dropout=mo["dropout"],
            num_labels=mo["num_labels"],
            freeze_encoder=True,
        ).to(device)
        model_s3.load_state_dict(
            torch.load(fold_dir / "stage2_best.pt", map_location=device, weights_only=True)
        )
        unfreeze_last_n_layers(model_s3.encoder, n_unfreeze)

        optimizer_s3 = build_layerwise_optimizer(model_s3, s3["encoder_lr"], s3["head_lr"])
        criterion_s3 = nn.CrossEntropyLoss(label_smoothing=s3["label_smoothing"])

        best_s3_acc, best_s3_loss = train_stage(
            model=model_s3,
            train_texts=train_texts_s3, train_labels=train_labels_s3,
            val_texts=val_texts, val_labels=val_labels,
            tokenizer=tokenizer, device=device,
            optimizer=optimizer_s3, criterion=criterion_s3,
            epochs=s3["epochs"], batch_size=s3["batch_size"], max_length=s3["max_length"],
            ckpt_path=fold_dir / "stage3_best.pt",
            log_path=fold_dir / "stage3_log.csv",
            stage_name="S3",
        )

        final_best_acc  = best_s3_acc
        final_best_loss = best_s3_loss
        shutil.copy(fold_dir / "stage3_best.pt", fold_dir / "best.pt")
        print(f"\n  Stage 3 done  best_val_acc={best_s3_acc:.4f}")
    else:
        print(f"\n  Stage 2 done  best_val_acc={best_s2_acc:.4f}  (Stage 3 disabled)")
        shutil.copy(fold_dir / "stage2_best.pt", fold_dir / "best.pt")

    return {
        "fold":          fold,
        "best_val_acc":  round(final_best_acc, 4),
        "best_val_loss": round(final_best_loss, 4),
        "stage1_acc":    round(best_s1_acc, 4),
        "stage2_acc":    round(best_s2_acc, 4),
        "n_pseudo_s1":   n_pseudo_s1,
        "n_pseudo_s2":   n_pseudo_s2,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg    = load_config(EXP_DIR / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    s1 = cfg["stage1"]
    mo = cfg["model"]
    s3_enabled = cfg.get("stage3", {}).get("enabled", False)

    set_seed(s1["seed"])

    exp_name        = cfg["experiment"]["name"]
    exp_results_dir = RESULTS_DIR / exp_name
    run_dir, run_id = next_run_dir(exp_results_dir)

    print(f"Experiment    : {exp_name}")
    print(f"Run           : {run_id}")
    print(f"Device        : {device}")
    print(f"Unfreeze last : {cfg['stage2']['unfreeze_last_n']} layers")
    print(f"Stage 3       : {'enabled' if s3_enabled else 'disabled'}")

    shutil.copy(EXP_DIR / "config.yaml", run_dir / "config.yaml")

    labeled_texts, labeled_labels = load_labeled_csv(TRAIN_FILE)
    _, unlabeled_texts = load_unlabeled_csv(TEST_FILE)
    label_arr = np.array(labeled_labels)

    print(f"\nLabeled   : {len(labeled_texts)}")
    print(f"Unlabeled : {len(unlabeled_texts)}")

    pretrained = MODEL_REGISTRY[mo["name"]]
    tokenizer  = AutoTokenizer.from_pretrained(pretrained)

    n_folds = s1["n_folds"]
    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=s1["seed"])
    splits  = list(skf.split(labeled_texts, label_arr))

    fold_results:    list[dict] = []
    pseudo_s1_counts: list[int] = []
    pseudo_s2_counts: list[int] = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        result = run_fold(
            fold=fold,
            train_idx=train_idx, val_idx=val_idx,
            labeled_texts=labeled_texts, labeled_labels=labeled_labels,
            unlabeled_texts=unlabeled_texts,
            pretrained=pretrained, tokenizer=tokenizer,
            device=device, cfg=cfg, run_dir=run_dir,
        )
        fold_results.append(result)
        pseudo_s1_counts.append(result["n_pseudo_s1"])
        pseudo_s2_counts.append(result["n_pseudo_s2"])

    # ── Summary ──────────────────────────────────────────────────────────
    accs = [r["best_val_acc"] for r in fold_results]
    mean_pseudo_s1 = round(float(np.mean(pseudo_s1_counts)), 1)
    mean_pseudo_s2 = round(float(np.mean(pseudo_s2_counts)), 1) if s3_enabled else 0.0

    cv_summary = {
        "experiment":    exp_name,
        "run_id":        run_id,
        "timestamp":     datetime.now().isoformat(timespec="seconds"),
        "config":        cfg,
        "fold_results":  fold_results,
        "mean_val_acc":  round(float(np.mean(accs)), 4),
        "std_val_acc":   round(float(np.std(accs)), 4),
        "mean_pseudo_s1": mean_pseudo_s1,
        "mean_pseudo_s2": mean_pseudo_s2,
    }

    with open(run_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, ensure_ascii=False, indent=2)

    append_experiment_log(exp_results_dir / "experiment_log.json", cv_summary)
    append_experiment_csv(RESULTS_DIR / "experiment_log.csv", cv_summary)

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"CV Summary  (run {run_id})")
    print(sep)
    for r in fold_results:
        print(
            f"  Fold {r['fold']}  "
            f"S1={r['stage1_acc']:.4f}  S2={r['stage2_acc']:.4f}  "
            f"final={r['best_val_acc']:.4f}  "
            f"pseudo_S1={r['n_pseudo_s1']}  pseudo_S2={r['n_pseudo_s2']}"
        )
    print(f"  Mean val_acc  : {cv_summary['mean_val_acc']:.4f}")
    print(f"  Std  val_acc  : {cv_summary['std_val_acc']:.4f}")
    print(f"  Mean pseudo S1: {mean_pseudo_s1:.0f}")
    print(f"  Mean pseudo S2: {mean_pseudo_s2:.0f}")
    print(f"  Run dir       : {run_dir}")


if __name__ == "__main__":
    main()
