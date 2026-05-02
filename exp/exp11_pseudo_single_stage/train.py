"""
exp/exp11_pseudo_single_stage/train.py

Self-contained pseudo-label training — no external pseudo-label file required.

Two-phase pipeline
------------------
Phase 0  Train on ALL labeled data (no fold split) to generate pseudo-labels
         for the test set. Uses same hyperparameters, fixed epochs (no early stop).

Phase 1  5-fold CV: each fold trains on its labeled split + ALL pseudo-labels
         from Phase 0. WeightedTrainer, EarlyStop(patience=2), best by val F1.
         5-fold softmax ensemble → predictions.csv.
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
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

try:
    import regex as re
except ImportError:
    import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR      = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_REGISTRY, RESULTS_DIR, TEST_FILE, TRAIN_FILE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def next_run_dir(base: Path) -> tuple[Path, str]:
    base.mkdir(parents=True, exist_ok=True)
    ids = sorted([d.name for d in base.iterdir() if d.is_dir() and d.name.isdigit()], key=int)
    run_id  = f"{int(ids[-1]) + 1:02d}" if ids else "01"
    run_dir = base / run_id
    run_dir.mkdir()
    return run_dir, run_id


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_labeled(path: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(clean_text(row["TEXT"].strip()))
            labels.append(int(row["LABEL"].strip()))
    return texts, labels


def read_test(path: Path) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(row["row_id"].strip())
            texts.append(clean_text(row["TEXT"].strip()))
    return ids, texts


# ---------------------------------------------------------------------------
# Dataset & model
# ---------------------------------------------------------------------------

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, labels: list[int] | None = None):
        self.enc    = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.enc["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels  = inputs.pop("labels")
        out     = model(**inputs)
        loss_fn = torch.nn.CrossEntropyLoss(weight=self._class_weights)
        loss    = loss_fn(out.logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, out) if return_outputs else loss


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def eval_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1":       f1_score(labels, preds, average="binary"),
        "accuracy": accuracy_score(labels, preds),
    }


def make_args(out_dir: Path, cfg: dict, fp16: bool, seed: int,
              early_stop: bool = True) -> TrainingArguments:
    kwargs = dict(
        output_dir                  = str(out_dir),
        num_train_epochs            = cfg["epochs"],
        per_device_train_batch_size = cfg["batch_size"],
        per_device_eval_batch_size  = cfg["batch_size"] * 2,
        gradient_accumulation_steps = cfg["gradient_accumulation"],
        learning_rate               = cfg.get("learning_rate", 1e-5),
        weight_decay                = cfg["weight_decay"],
        warmup_ratio                = cfg["warmup_ratio"],
        save_strategy               = "epoch" if early_stop else "no",
        load_best_model_at_end      = early_stop,
        metric_for_best_model       = "f1" if early_stop else None,
        greater_is_better           = True if early_stop else None,
        logging_steps               = 50,
        report_to                   = "none",
        seed                        = seed,
        fp16                        = fp16,
    )
    if not early_stop:
        return TrainingArguments(**kwargs, eval_strategy="no")

    for eval_kw in ({"eval_strategy": "epoch"}, {"evaluation_strategy": "epoch"}):
        try:
            return TrainingArguments(**kwargs, **eval_kw)
        except TypeError:
            continue
    return TrainingArguments(**kwargs, evaluation_strategy="epoch", no_cuda=not fp16)


# ---------------------------------------------------------------------------
# Phase 0: generate pseudo-labels from full labeled set
# ---------------------------------------------------------------------------

def generate_pseudo_labels(
    labeled_texts: list[str],
    labeled_labels: list[int],
    test_texts: list[str],
    pretrained: str,
    tokenizer,
    collator,
    device: torch.device,
    cfg: dict,
    out_dir: Path,
    fp16: bool,
    seed: int,
) -> tuple[list[str], list[int]]:
    """Train on all labeled data, predict test set, return (texts, labels)."""
    print("\n[Phase 0] Generating pseudo-labels on full labeled set...")

    tr_enc   = tokenizer(labeled_texts, truncation=True, max_length=cfg["max_length"])
    cw       = compute_class_weight("balanced", classes=np.unique(labeled_labels), y=labeled_labels)
    cw_t     = torch.tensor(cw, dtype=torch.float).to(device)

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)

    # No eval set → save_strategy="no", no early stopping
    trainer = WeightedLossTrainer(
        model         = model,
        args          = make_args(out_dir / "phase0_ckpts", cfg, fp16, seed, early_stop=False),
        train_dataset = SentimentDataset(tr_enc, labeled_labels),
        tokenizer     = tokenizer,
        data_collator = collator,
        class_weights = cw_t,
    )
    trainer.train()

    te_enc = tokenizer(test_texts, truncation=True, max_length=cfg["max_length"])
    with torch.no_grad():
        pred_out = trainer.predict(SentimentDataset(te_enc))
    probs         = torch.softmax(torch.tensor(pred_out.predictions), dim=1).numpy()
    pseudo_labels = probs.argmax(axis=1).tolist()

    pos = sum(pseudo_labels)
    print(f"  Generated {len(pseudo_labels)} pseudo-labels  pos={pos}  neg={len(pseudo_labels)-pos}")

    del model, trainer, cw_t
    torch.cuda.empty_cache()

    return test_texts, pseudo_labels


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------

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


def save_json_log(path: Path, entry: dict) -> None:
    log = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    log.append(entry)
    path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv_log(path: Path, summary: dict) -> None:
    cfg   = summary["config"]
    tr    = cfg["training"]
    folds = {r["fold"]: r for r in summary["fold_results"]}

    row = {
        "exp":               summary["experiment"],
        "run_id":            summary["run_id"],
        "timestamp":         summary["timestamp"],
        "mean_val_acc":      summary["mean_val_acc"],
        "std_val_acc":       summary["std_val_acc"],
        **{f"fold{i}_acc":   folds.get(i, {}).get("val_acc")  for i in range(5)},
        **{f"fold{i}_loss":  folds.get(i, {}).get("val_loss") for i in range(5)},
        "learning_rate":     tr["learning_rate"],
        "batch_size":        tr["batch_size"],
        "epochs":            tr["epochs"],
        "max_length":        tr["max_length"],
        "seed":              tr["seed"],
        "n_folds":           tr["n_folds"],
        "model_name":        cfg["model"]["name"],
        "mean_pseudo_labels": summary["n_pseudo"],
        "warmup_ratio":      tr["warmup_ratio"],
    }

    need_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if need_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDNAMES})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg    = load_config(EXP_DIR / "config.yaml")
    tr     = cfg["training"]
    pg     = cfg["pseudo_gen"]
    mo     = cfg["model"]
    fp16   = torch.cuda.is_available()
    device = torch.device("cuda" if fp16 else "cpu")

    set_seed(tr["seed"])

    exp_name        = cfg["experiment"]["name"]
    run_dir, run_id = next_run_dir(RESULTS_DIR / exp_name)

    print(f"Experiment : {exp_name}  run={run_id}")
    print(f"Device     : {device}  fp16={fp16}")
    shutil.copy(EXP_DIR / "config.yaml", run_dir / "config.yaml")

    labeled_texts, labeled_labels = read_labeled(TRAIN_FILE)
    test_ids, test_texts          = read_test(TEST_FILE)

    pretrained = MODEL_REGISTRY[mo["name"]]
    tokenizer  = AutoTokenizer.from_pretrained(pretrained)
    collator   = DataCollatorWithPadding(tokenizer=tokenizer)

    # Phase 0: generate pseudo-labels independently
    pseudo_texts, pseudo_labels = generate_pseudo_labels(
        labeled_texts, labeled_labels, test_texts,
        pretrained, tokenizer, collator,
        device, pg, run_dir, fp16, tr["seed"],
    )
    n_pseudo = len(pseudo_texts)

    print(f"\nLabeled      : {len(labeled_texts)}")
    print(f"Pseudo-labels: {n_pseudo}")

    # Phase 1: 5-fold training with labeled + pseudo-labels
    skf    = StratifiedKFold(n_splits=tr["n_folds"], shuffle=True, random_state=tr["seed"])
    splits = list(skf.split(labeled_texts, labeled_labels))

    fold_results: list[dict]       = []
    fold_probs:   list[np.ndarray] = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold + 1}/{tr['n_folds']}")
        print("=" * 50)

        fold_dir = run_dir / f"fold{fold + 1}"
        fold_dir.mkdir()

        tr_texts  = [labeled_texts[i]  for i in tr_idx] + pseudo_texts
        tr_labels = [labeled_labels[i] for i in tr_idx] + pseudo_labels
        va_texts  = [labeled_texts[i]  for i in va_idx]
        va_labels = [labeled_labels[i] for i in va_idx]

        print(f"  train={len(tr_texts)} (labeled={len(tr_idx)} + pseudo={n_pseudo})  val={len(va_texts)}")

        tr_enc = tokenizer(tr_texts, truncation=True, max_length=tr["max_length"])
        va_enc = tokenizer(va_texts, truncation=True, max_length=tr["max_length"])

        cw   = compute_class_weight("balanced", classes=np.unique(tr_labels), y=tr_labels)
        cw_t = torch.tensor(cw, dtype=torch.float).to(device)

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained, num_labels=2, ignore_mismatched_sizes=True
        ).to(device)

        trainer = WeightedLossTrainer(
            model          = model,
            args           = make_args(fold_dir / "ckpts", tr, fp16, tr["seed"]),
            train_dataset  = SentimentDataset(tr_enc, tr_labels),
            eval_dataset   = SentimentDataset(va_enc, va_labels),
            tokenizer      = tokenizer,
            data_collator  = collator,
            compute_metrics= eval_metrics,
            callbacks      = [EarlyStoppingCallback(early_stopping_patience=2)],
            class_weights  = cw_t,
        )
        trainer.train()

        m        = trainer.evaluate()
        val_acc  = m.get("eval_accuracy", 0.0)
        val_loss = m.get("eval_loss",     0.0)
        val_f1   = m.get("eval_f1",       0.0)
        print(f"  acc={val_acc:.4f}  f1={val_f1:.4f}  loss={val_loss:.4f}")

        trainer.save_model(str(fold_dir / "best_model"))

        te_enc    = tokenizer(test_texts, truncation=True, max_length=tr["max_length"])
        pred_out  = trainer.predict(SentimentDataset(te_enc))
        test_probs = torch.softmax(torch.tensor(pred_out.predictions), dim=1).numpy()

        fold_results.append({"fold": fold, "val_acc": round(val_acc, 4),
                              "val_loss": round(val_loss, 4), "val_f1": round(val_f1, 4)})
        fold_probs.append(test_probs)

        del model, trainer, cw_t
        torch.cuda.empty_cache()

    # Ensemble
    preds     = np.mean(fold_probs, axis=0).argmax(axis=1).tolist()
    pred_path = run_dir / "predictions.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "LABEL"])
        w.writerows(zip(test_ids, preds))

    accs    = [r["val_acc"] for r in fold_results]
    summary = {
        "experiment":    exp_name,
        "run_id":        run_id,
        "timestamp":     datetime.now().isoformat(timespec="seconds"),
        "config":        cfg,
        "fold_results":  fold_results,
        "mean_val_acc":  round(float(np.mean(accs)), 4),
        "std_val_acc":   round(float(np.std(accs)),  4),
        "n_pseudo":      n_pseudo,
    }
    (run_dir / "cv_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    exp_dir = RESULTS_DIR / exp_name
    save_json_log(exp_dir / "experiment_log.json", summary)
    save_csv_log(RESULTS_DIR / "experiment_log.csv", summary)

    print(f"\n{'=' * 50}")
    print(f"CV Summary  run={run_id}")
    print("=" * 50)
    for r in fold_results:
        print(f"  Fold {r['fold']}  acc={r['val_acc']:.4f}  f1={r['val_f1']:.4f}")
    print(f"  mean_acc={summary['mean_val_acc']:.4f}  std={summary['std_val_acc']:.4f}")
    print(f"  pseudo={n_pseudo}  predictions → {pred_path}")


if __name__ == "__main__":
    main()
