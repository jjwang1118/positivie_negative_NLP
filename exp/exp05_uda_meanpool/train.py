"""
exp/exp05_uda_meanpool/train.py
===============================
UDA semi-supervised training with mean pooling (exp05).

Inherits exp04's UDA mechanism (TSA + KL consistency loss) but replaces the
learned Attention Layer + Weighted Sum with mask-aware mean pooling, which is
the aggregation method all-MiniLM-L6-v2 was originally trained with.

Architecture
------------
  Text
    → SBERT encoder (frozen)         [B, T, 384]
    → Mean Pooling (mask-aware)       [B, 384]
    → MLP: Linear(384→64) → ReLU → Dropout(0.1) → Linear(64→2)
    → logits

Training objective
------------------
  total_loss = supervised_loss + lambda_u * consistency_loss

  supervised_loss : CrossEntropyLoss on labeled data, gated by TSA threshold.
  consistency_loss: KL(p_orig || p_aug) on unlabeled data, filtered by
                    confidence threshold. p_orig is produced with stop-gradient.

Usage
-----
    python exp/exp05_uda_meanpool/train.py
"""

import csv
import itertools
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
import torch.nn.functional as F
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR      = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_REGISTRY, RESULTS_DIR, TEST_FILE, TRAIN_FILE
from src.model import MeanPoolingClassifier


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    "lambda_u", "confidence_threshold", "tsa_schedule", "augment_prob",
]


def append_experiment_csv(csv_path: Path, entry: dict) -> None:
    cfg  = entry["config"]
    tr   = cfg.get("training", {})
    mo   = cfg.get("model", {})
    uda  = cfg.get("uda", {})

    fold_data = {r["fold"]: r for r in entry["fold_results"]}
    row = {
        "exp":                  entry.get("experiment", ""),
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
        "lambda_u":             uda.get("lambda_u"),
        "confidence_threshold": uda.get("confidence_threshold"),
        "tsa_schedule":         uda.get("tsa_schedule"),
        "augment_prob":         uda.get("augment_prob"),
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
# TF-IDF Augmenter
# ---------------------------------------------------------------------------

class TFIDFAugmenter:
    """
    Word-level augmentation guided by TF-IDF scores.

    Low-TF-IDF words (generic function words, stop words) are candidates
    for random replacement.  High-TF-IDF words (sentiment keywords) are
    preserved to maintain semantic meaning.
    """

    def __init__(
        self,
        corpus: list[str],
        augment_prob: float = 0.1,
        low_tfidf_percentile: float = 40.0,
    ):
        self.augment_prob = augment_prob

        self.vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            min_df=1,
        )
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        vocab = self.vectorizer.get_feature_names_out()

        mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
        self.word_score: dict[str, float] = {
            w: float(mean_scores[i]) for i, w in enumerate(vocab)
        }

        all_scores = np.array(list(self.word_score.values()))
        self.low_threshold = float(np.percentile(all_scores, low_tfidf_percentile))

        self.low_vocab = [
            w for w, s in self.word_score.items()
            if s <= self.low_threshold
        ]
        if not self.low_vocab:
            self.low_vocab = list(self.word_score.keys())

    def _is_low_tfidf(self, word: str) -> bool:
        return self.word_score.get(word.lower(), 0.0) <= self.low_threshold

    def augment(self, text: str) -> str:
        tokens = text.split()
        augmented = []
        for token in tokens:
            word = token.lower().strip(".,!?;:'\"()[]{}—-")
            if word and self._is_low_tfidf(word) and random.random() < self.augment_prob:
                augmented.append(random.choice(self.low_vocab))
            else:
                augmented.append(token)
        return " ".join(augmented)


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


class UnlabeledDataset(Dataset):
    def __init__(self, texts: list[str], augmenter: TFIDFAugmenter, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.texts     = texts
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        orig_text = self.texts[idx]
        aug_text  = self.augmenter.augment(orig_text)

        orig_enc = self.tokenizer(
            orig_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        aug_enc = self.tokenizer(
            aug_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "orig_input_ids":      orig_enc["input_ids"].squeeze(0),
            "orig_attention_mask": orig_enc["attention_mask"].squeeze(0),
            "aug_input_ids":       aug_enc["input_ids"].squeeze(0),
            "aug_attention_mask":  aug_enc["attention_mask"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_labeled_csv(path: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["TEXT"].strip())
            labels.append(int(row["LABEL"].strip()))
    return texts, labels


def load_unlabeled_csv(path: Path) -> tuple[list[str], list[str]]:
    row_ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_ids.append(row["row_id"].strip())
            texts.append(row["TEXT"].strip())
    return row_ids, texts


# ---------------------------------------------------------------------------
# TSA (Training Signal Annealing)
# ---------------------------------------------------------------------------

def tsa_threshold(step: int, total_steps: int, schedule: str, num_labels: int) -> float:
    """
    Linear schedule: eta_t grows from 1/K to 1.0 over training.
    A labeled sample is included only when model confidence < eta_t.
    """
    alpha = step / max(total_steps - 1, 1)
    if schedule == "linear":
        threshold = alpha
    elif schedule == "log":
        threshold = 1 - math.exp(-alpha * 5)
    elif schedule == "exp":
        threshold = math.exp((alpha - 1) * 5)
    else:
        raise ValueError(f"Unknown TSA schedule: {schedule}")

    return 1.0 / num_labels + threshold * (1.0 - 1.0 / num_labels)


# ---------------------------------------------------------------------------
# UDA Trainer
# ---------------------------------------------------------------------------

class UDATrainer:
    def __init__(
        self,
        model: MeanPoolingClassifier,
        optimizer: AdamW,
        device: torch.device,
        cfg: dict,
        total_steps: int,
    ):
        self.model       = model
        self.optimizer   = optimizer
        self.device      = device
        self.cfg         = cfg
        self.total_steps = total_steps

        uda = cfg["uda"]
        mo  = cfg["model"]

        self.lambda_u       = uda["lambda_u"]
        self.confidence_thr = uda["confidence_threshold"]
        self.tsa_schedule   = uda["tsa_schedule"]
        self.num_labels     = mo["num_labels"]

        self.criterion   = nn.CrossEntropyLoss(reduction="none")
        self.global_step = 0

    def train_epoch(
        self,
        labeled_loader: DataLoader,
        unlabeled_iter,
    ) -> tuple[float, float, float]:
        self.model.train()
        sup_losses, con_losses, total_losses = [], [], []

        for labeled_batch in labeled_loader:
            unlabeled_batch = next(unlabeled_iter)

            # ---- Supervised branch ----------------------------------------
            input_ids      = labeled_batch["input_ids"].to(self.device)
            attention_mask = labeled_batch["attention_mask"].to(self.device)
            labels         = labeled_batch["label"].to(self.device)

            logits_sup = self.model(input_ids, attention_mask)
            probs_sup  = F.softmax(logits_sup, dim=-1)

            eta = tsa_threshold(
                self.global_step, self.total_steps,
                self.tsa_schedule, self.num_labels,
            )
            true_class_probs = probs_sup.gather(
                dim=1, index=labels.unsqueeze(1)
            ).squeeze(1)
            tsa_mask = (true_class_probs < eta).float()

            per_sample_loss = self.criterion(logits_sup, labels)
            n_keep = tsa_mask.sum().item()
            if n_keep > 0:
                sup_loss = (per_sample_loss * tsa_mask).sum() / tsa_mask.sum()
            else:
                sup_loss = torch.tensor(0.0, device=self.device)

            # ---- Consistency branch ---------------------------------------
            orig_ids  = unlabeled_batch["orig_input_ids"].to(self.device)
            orig_mask = unlabeled_batch["orig_attention_mask"].to(self.device)
            aug_ids   = unlabeled_batch["aug_input_ids"].to(self.device)
            aug_mask  = unlabeled_batch["aug_attention_mask"].to(self.device)

            with torch.no_grad():
                logits_orig = self.model(orig_ids, orig_mask)
                p = F.softmax(logits_orig, dim=-1)

            logits_aug = self.model(aug_ids, aug_mask)
            q = F.softmax(logits_aug, dim=-1)

            conf_mask = (p.max(dim=1).values >= self.confidence_thr).float()

            kl_per_sample = F.kl_div(
                F.log_softmax(logits_aug, dim=-1),
                p.detach(),
                reduction="none",
            ).sum(dim=-1)

            n_conf = conf_mask.sum().item()
            if n_conf > 0:
                con_loss = (kl_per_sample * conf_mask).sum() / conf_mask.sum()
            else:
                con_loss = torch.tensor(0.0, device=self.device)

            # ---- Update ---------------------------------------------------
            total_loss = sup_loss + self.lambda_u * con_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            sup_losses.append(sup_loss.item())
            con_losses.append(con_loss.item())
            total_losses.append(total_loss.item())
            self.global_step += 1

        return (
            float(np.mean(sup_losses)),
            float(np.mean(con_losses)),
            float(np.mean(total_losses)),
        )

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> tuple[float, float, float]:
        self.model.eval()
        criterion_full = nn.CrossEntropyLoss()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in val_loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["label"].to(self.device)

            logits = self.model(input_ids, attention_mask)
            total_loss += criterion_full(logits, labels).item()

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        val_loss = total_loss / len(val_loader)
        val_acc  = accuracy_score(all_labels, all_preds)
        val_f1   = f1_score(all_labels, all_preds, average="macro")
        return val_loss, val_acc, val_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg    = load_config(EXP_DIR / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr  = cfg["training"]
    mo  = cfg["model"]
    uda = cfg["uda"]

    set_seed(tr["seed"])

    exp_name        = cfg["experiment"]["name"]
    exp_results_dir = RESULTS_DIR / exp_name
    run_dir, run_id = next_run_dir(exp_results_dir)

    print(f"Experiment  : {exp_name}")
    print(f"Run         : {run_id}")
    print(f"Run dir     : {run_dir}")
    print(f"Device      : {device}")
    print(f"Pooling     : mean (no attention layer)")
    print(f"Lambda_u    : {uda['lambda_u']}")
    print(f"Conf. thr.  : {uda['confidence_threshold']}")
    print(f"TSA schedule: {uda['tsa_schedule']}")

    shutil.copy(EXP_DIR / "config.yaml", run_dir / "config.yaml")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")
    all_texts, all_labels = load_labeled_csv(TRAIN_FILE)
    label_arr = np.array(all_labels)
    print(f"\nLabeled samples  : {len(all_texts)}")

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")
    _, unlabeled_texts = load_unlabeled_csv(TEST_FILE)
    print(f"Unlabeled samples: {len(unlabeled_texts)}")

    print("\nFitting TF-IDF augmenter on labeled corpus...")
    augmenter = TFIDFAugmenter(
        corpus=all_texts,
        augment_prob=uda["augment_prob"],
    )
    print(f"  Vocabulary size    : {len(augmenter.word_score)}")
    print(f"  Low-TF-IDF pool   : {len(augmenter.low_vocab)} words")
    print(f"  Low-TF-IDF cutoff : {augmenter.low_threshold:.4f}")

    pretrained = MODEL_REGISTRY[mo["name"]]
    tokenizer  = AutoTokenizer.from_pretrained(pretrained)

    n_folds = tr["n_folds"]
    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=tr["seed"])
    splits  = list(skf.split(all_texts, label_arr))

    unlabeled_ds = UnlabeledDataset(
        unlabeled_texts, augmenter, tokenizer, tr["max_length"]
    )
    unlabeled_loader_base = DataLoader(
        unlabeled_ds,
        batch_size=uda["unlabeled_batch_size"],
        shuffle=True,
        drop_last=False,
    )

    fold_summaries: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{n_folds}  |  train={len(train_idx)}  val={len(val_idx)}")
        print(f"{'='*60}")

        fold_dir = run_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_texts  = [all_texts[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        val_texts    = [all_texts[i] for i in val_idx]
        val_labels   = [all_labels[i] for i in val_idx]

        labeled_train_ds = LabeledDataset(train_texts, train_labels, tokenizer, tr["max_length"])
        labeled_val_ds   = LabeledDataset(val_texts,   val_labels,   tokenizer, tr["max_length"])

        labeled_loader = DataLoader(
            labeled_train_ds,
            batch_size=tr["batch_size"],
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            labeled_val_ds,
            batch_size=tr["batch_size"],
            shuffle=False,
        )

        unlabeled_iter = itertools.cycle(unlabeled_loader_base)

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

        steps_per_epoch = len(labeled_loader)
        total_steps     = tr["epochs"] * steps_per_epoch

        trainer = UDATrainer(
            model       = model,
            optimizer   = optimizer,
            device      = device,
            cfg         = cfg,
            total_steps = total_steps,
        )

        best_val_acc  = 0.0
        best_val_loss = float("inf")
        best_val_f1   = 0.0
        log_rows: list[dict] = []

        for epoch in range(1, tr["epochs"] + 1):
            avg_sup, avg_con, avg_total = trainer.train_epoch(
                labeled_loader, unlabeled_iter
            )
            val_loss, val_acc, val_f1 = trainer.evaluate(val_loader)

            log_rows.append({
                "epoch":            epoch,
                "train_sup_loss":   f"{avg_sup:.4f}",
                "train_con_loss":   f"{avg_con:.4f}",
                "train_total_loss": f"{avg_total:.4f}",
                "val_loss":         f"{val_loss:.4f}",
                "val_acc":          f"{val_acc:.4f}",
                "val_f1":           f"{val_f1:.4f}",
            })
            print(
                f"  Epoch {epoch:02d}/{tr['epochs']}  "
                f"sup={avg_sup:.4f}  con={avg_con:.4f}  total={avg_total:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc  = val_acc
                best_val_loss = val_loss
                best_val_f1   = val_f1
                torch.save(model.state_dict(), fold_dir / "best.pt")
                print(f"    -> saved best (val_acc={best_val_acc:.4f})")

        torch.save(model.state_dict(), fold_dir / "last.pt")

        log_fields = [
            "epoch", "train_sup_loss", "train_con_loss", "train_total_loss",
            "val_loss", "val_acc", "val_f1",
        ]
        with open(fold_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()
            writer.writerows(log_rows)

        fold_summaries.append({
            "fold":          fold,
            "best_val_acc":  round(best_val_acc, 4),
            "best_val_loss": round(best_val_loss, 4),
            "best_val_f1":   round(best_val_f1,  4),
        })
        print(f"  Fold {fold} complete  best_val_acc={best_val_acc:.4f}  best_val_f1={best_val_f1:.4f}")

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
    append_experiment_csv(RESULTS_DIR / "experiment_log.csv", cv_summary)

    print(f"\n{'='*60}")
    print(f"Cross-Validation Summary  (run {run_id})")
    print(f"{'='*60}")
    for s in fold_summaries:
        print(
            f"  Fold {s['fold']}  "
            f"best_val_acc={s['best_val_acc']:.4f}  "
            f"best_val_loss={s['best_val_loss']:.4f}  "
            f"best_val_f1={s['best_val_f1']:.4f}"
        )
    print(f"  Mean val_acc : {cv_summary['mean_val_acc']:.4f}")
    print(f"  Std  val_acc : {cv_summary['std_val_acc']:.4f}")
    print(f"  Run dir      : {run_dir}")


if __name__ == "__main__":
    main()
