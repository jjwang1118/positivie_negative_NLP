"""
exp/exp07_cluster_pseudo_siebert/train.py
==========================================
Cluster-based pseudo-labeling + supervised fine-tuning.
Encoder: siebert/sentiment-roberta-large-english (1024-dim, sentiment fine-tuned)

Pipeline
--------
1. Encode all 13k texts with frozen encoder → [13k × 1024] embeddings.
2. Run K-means once on all embeddings (unsupervised, no label leakage).
3. For each CV fold:
   a. Vote cluster majority label using ONLY the fold's train split.
   b. Assign pseudo-labels to unlabeled texts whose cluster passes the
      purity threshold and has enough labeled anchors.
   c. Train MeanPoolingClassifier on labeled train + pseudo-labeled data.
   d. Evaluate on held-out val split (labeled only, no pseudo-labels).
4. Aggregate 5-fold results and write experiment logs.

Usage
-----
    python exp/exp07_cluster_pseudo_siebert/train.py
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

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


FIELDNAMES = [
    "exp", "run_id", "timestamp", "mean_val_acc", "std_val_acc",
    "fold0_acc", "fold0_loss", "fold1_acc", "fold1_loss",
    "fold2_acc", "fold2_loss", "fold3_acc", "fold3_loss",
    "fold4_acc", "fold4_loss",
    "learning_rate", "batch_size", "epochs", "max_length", "seed", "n_folds",
    "model_name", "hidden_dim", "mlp_hidden", "dropout",
    "n_clusters", "purity_threshold", "min_labeled_per_cluster",
    "mean_pseudo_labels",
]


def append_experiment_csv(csv_path: Path, entry: dict) -> None:
    cfg = entry["config"]
    tr  = cfg.get("training", {})
    mo  = cfg.get("model", {})
    cl  = cfg.get("clustering", {})

    fold_data = {r["fold"]: r for r in entry["fold_results"]}
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
        "learning_rate":           tr.get("learning_rate"),
        "batch_size":              tr.get("batch_size"),
        "epochs":                  tr.get("epochs"),
        "max_length":              tr.get("max_length"),
        "seed":                    tr.get("seed"),
        "n_folds":                 tr.get("n_folds"),
        "model_name":              mo.get("name"),
        "hidden_dim":              mo.get("hidden_dim"),
        "mlp_hidden":              mo.get("mlp_hidden"),
        "dropout":                 mo.get("dropout"),
        "n_clusters":              cl.get("n_clusters"),
        "purity_threshold":        cl.get("purity_threshold"),
        "min_labeled_per_cluster": cl.get("min_labeled_per_cluster"),
        "mean_pseudo_labels":      entry.get("mean_pseudo_labels"),
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
# Encoding (mean pooling, frozen)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_texts(
    texts: list[str],
    encoder,
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 128,
) -> np.ndarray:
    """Return mask-aware mean-pooled embeddings as numpy [N, hidden_dim]."""
    encoder.eval()
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc  = tokenizer(batch, max_length=max_length, padding="max_length",
                         truncation=True, return_tensors="pt")
        ids  = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        tok  = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        m    = mask.unsqueeze(-1).float()
        vec  = (tok * m).sum(1) / m.sum(1)
        vecs.append(vec.cpu().numpy())
        if (i // batch_size + 1) % 20 == 0 or i + batch_size >= len(texts):
            print(f"  Encoded [{min(i + batch_size, len(texts))}/{len(texts)}]")
    return np.vstack(vecs)


# ---------------------------------------------------------------------------
# Pseudo-labeling
# ---------------------------------------------------------------------------

def build_pseudo_labels(
    train_idx: np.ndarray,
    all_labeled_labels: list[int],
    labeled_cluster_ids: np.ndarray,
    unlabeled_cluster_ids: np.ndarray,
    unlabeled_texts: list[str],
    n_clusters: int,
    purity_threshold: float,
    min_labeled: int,
) -> tuple[list[str], list[int]]:
    labels_arr = np.array(all_labeled_labels)

    cluster_label: dict[int, int] = {}
    for c in range(n_clusters):
        in_c = np.where(labeled_cluster_ids[train_idx] == c)[0]
        n    = len(in_c)
        if n < min_labeled:
            continue
        orig_idx = train_idx[in_c]
        pos      = int(labels_arr[orig_idx].sum())
        purity   = max(pos, n - pos) / n
        if purity >= purity_threshold:
            cluster_label[c] = 1 if pos >= n - pos else 0

    pseudo_texts, pseudo_labels = [], []
    for j, c in enumerate(unlabeled_cluster_ids):
        if c in cluster_label:
            pseudo_texts.append(unlabeled_texts[j])
            pseudo_labels.append(cluster_label[c])

    return pseudo_texts, pseudo_labels


# ---------------------------------------------------------------------------
# Dataset
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


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(
    model: MeanPoolingClassifier,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(ids, mask)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: MeanPoolingClassifier,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg    = load_config(EXP_DIR / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = cfg["training"]
    mo = cfg["model"]
    cl = cfg["clustering"]

    set_seed(tr["seed"])

    exp_name        = cfg["experiment"]["name"]
    exp_results_dir = RESULTS_DIR / exp_name
    run_dir, run_id = next_run_dir(exp_results_dir)

    print(f"Experiment  : {exp_name}")
    print(f"Run         : {run_id}")
    print(f"Run dir     : {run_dir}")
    print(f"Device      : {device}")
    print(f"K-means K   : {cl['n_clusters']}")
    print(f"Purity thr. : {cl['purity_threshold']}")
    print(f"Min labeled : {cl['min_labeled_per_cluster']}")

    shutil.copy(EXP_DIR / "config.yaml", run_dir / "config.yaml")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
    labeled_texts, labeled_labels = load_labeled_csv(TRAIN_FILE)
    label_arr = np.array(labeled_labels)

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")
    _, unlabeled_texts = load_unlabeled_csv(TEST_FILE)

    all_texts = labeled_texts + unlabeled_texts
    n_labeled   = len(labeled_texts)
    n_unlabeled = len(unlabeled_texts)
    print(f"\nLabeled samples  : {n_labeled}")
    print(f"Unlabeled samples: {n_unlabeled}")

    # -----------------------------------------------------------------------
    # Encode all texts with frozen encoder (once, shared across folds)
    # -----------------------------------------------------------------------
    pretrained = MODEL_REGISTRY[mo["name"]]
    tokenizer  = AutoTokenizer.from_pretrained(pretrained)
    encoder    = AutoModel.from_pretrained(pretrained).to(device)
    for p in encoder.parameters():
        p.requires_grad = False

    print(f"\nEncoding all texts with frozen {pretrained}...")
    embeddings = encode_texts(all_texts, encoder, tokenizer, device,
                              max_length=tr["max_length"])
    print(f"Embeddings shape: {embeddings.shape}")

    labeled_cluster_ids_placeholder = embeddings[:n_labeled]

    # -----------------------------------------------------------------------
    # K-means (once on all embeddings — unsupervised, no leakage)
    # -----------------------------------------------------------------------
    K = cl["n_clusters"]
    print(f"\nRunning K-means (K={K})...")
    km = MiniBatchKMeans(n_clusters=K, random_state=cl["random_state"], n_init=5)
    all_cluster_ids       = km.fit_predict(embeddings)
    labeled_cluster_ids   = all_cluster_ids[:n_labeled]
    unlabeled_cluster_ids = all_cluster_ids[n_labeled:]

    # -----------------------------------------------------------------------
    # 5-fold CV
    # -----------------------------------------------------------------------
    n_folds = tr["n_folds"]
    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=tr["seed"])
    splits  = list(skf.split(labeled_texts, label_arr))

    fold_summaries: list[dict] = []
    pseudo_label_counts: list[int] = []
    criterion = nn.CrossEntropyLoss()

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{n_folds}  |  "
              f"train={len(train_idx)}  val={len(val_idx)}")
        print(f"{'='*60}")

        fold_dir = run_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # --- Pseudo-labeling (using train split labels only) ---------------
        pseudo_texts, pseudo_label_list = build_pseudo_labels(
            train_idx             = train_idx,
            all_labeled_labels    = labeled_labels,
            labeled_cluster_ids   = labeled_cluster_ids,
            unlabeled_cluster_ids = unlabeled_cluster_ids,
            unlabeled_texts       = unlabeled_texts,
            n_clusters            = K,
            purity_threshold      = cl["purity_threshold"],
            min_labeled           = cl["min_labeled_per_cluster"],
        )
        pseudo_label_counts.append(len(pseudo_label_list))
        print(f"  Pseudo-labels : {len(pseudo_label_list)} / {n_unlabeled} "
              f"({len(pseudo_label_list)/n_unlabeled:.1%})")

        # --- Build combined train set ---------------------------------------
        train_texts_fold  = [labeled_texts[i]  for i in train_idx] + pseudo_texts
        train_labels_fold = [labeled_labels[i] for i in train_idx] + pseudo_label_list
        val_texts_fold    = [labeled_texts[i]  for i in val_idx]
        val_labels_fold   = [labeled_labels[i] for i in val_idx]

        print(f"  Train size (labeled + pseudo): {len(train_texts_fold)}")

        train_ds = LabeledDataset(train_texts_fold, train_labels_fold,
                                  tokenizer, tr["max_length"])
        val_ds   = LabeledDataset(val_texts_fold,   val_labels_fold,
                                  tokenizer, tr["max_length"])

        train_loader = DataLoader(train_ds, batch_size=tr["batch_size"],
                                  shuffle=True, drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=tr["batch_size"],
                                  shuffle=False)

        # --- Build model ---------------------------------------------------
        model = MeanPoolingClassifier(
            pretrained_name=pretrained,
            hidden_dim=mo["hidden_dim"],
            mlp_hidden=mo["mlp_hidden"],
            dropout=mo["dropout"],
            num_labels=mo["num_labels"],
            freeze_encoder=True,
        ).to(device)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=tr["learning_rate"],
        )

        # --- Training loop -------------------------------------------------
        best_val_acc  = 0.0
        best_val_loss = float("inf")
        best_val_f1   = 0.0
        log_rows: list[dict] = []

        for epoch in range(1, tr["epochs"] + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

            log_rows.append({
                "epoch":      epoch,
                "train_loss": f"{train_loss:.4f}",
                "val_loss":   f"{val_loss:.4f}",
                "val_acc":    f"{val_acc:.4f}",
                "val_f1":     f"{val_f1:.4f}",
            })
            print(f"  Epoch {epoch:02d}/{tr['epochs']}  "
                  f"train={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

            if val_acc > best_val_acc:
                best_val_acc  = val_acc
                best_val_loss = val_loss
                best_val_f1   = val_f1
                torch.save(model.state_dict(), fold_dir / "best.pt")
                print(f"    -> saved best (val_acc={best_val_acc:.4f})")

        torch.save(model.state_dict(), fold_dir / "last.pt")

        with open(fold_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "val_f1"]
            )
            writer.writeheader()
            writer.writerows(log_rows)

        fold_summaries.append({
            "fold":          fold,
            "best_val_acc":  round(best_val_acc, 4),
            "best_val_loss": round(best_val_loss, 4),
            "best_val_f1":   round(best_val_f1, 4),
            "n_pseudo":      len(pseudo_label_list),
        })
        print(f"  Fold {fold} done  "
              f"best_val_acc={best_val_acc:.4f}  best_val_f1={best_val_f1:.4f}")

    # -----------------------------------------------------------------------
    # CV summary
    # -----------------------------------------------------------------------
    accs = [s["best_val_acc"] for s in fold_summaries]
    mean_pseudo = round(float(np.mean(pseudo_label_counts)), 1)

    cv_summary = {
        "experiment":         exp_name,
        "run_id":             run_id,
        "timestamp":          datetime.now().isoformat(timespec="seconds"),
        "config":             cfg,
        "fold_results":       fold_summaries,
        "mean_val_acc":       round(float(np.mean(accs)), 4),
        "std_val_acc":        round(float(np.std(accs)), 4),
        "mean_pseudo_labels": mean_pseudo,
    }

    with open(run_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, ensure_ascii=False, indent=2)

    append_experiment_log(exp_results_dir / "experiment_log.json", cv_summary)
    append_experiment_csv(RESULTS_DIR / "experiment_log.csv", cv_summary)

    print(f"\n{'='*60}")
    print(f"Cross-Validation Summary  (run {run_id})")
    print(f"{'='*60}")
    for s in fold_summaries:
        print(f"  Fold {s['fold']}  "
              f"best_val_acc={s['best_val_acc']:.4f}  "
              f"best_val_loss={s['best_val_loss']:.4f}  "
              f"n_pseudo={s['n_pseudo']}")
    print(f"  Mean val_acc      : {cv_summary['mean_val_acc']:.4f}")
    print(f"  Std  val_acc      : {cv_summary['std_val_acc']:.4f}")
    print(f"  Mean pseudo-labels: {mean_pseudo:.0f}")
    print(f"  Run dir           : {run_dir}")


if __name__ == "__main__":
    main()
