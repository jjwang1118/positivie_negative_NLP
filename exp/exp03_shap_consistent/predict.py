"""
exp/exp03_shap_consistent/predict.py
======================================
SHAP-consistent ensemble prediction for exp03.

Problem solved
--------------
exp02 trained Phase 3 on SHAP key strings but predicted with original full
text → train/test distribution mismatch → val_acc 76.7% but test_acc 71.0%.

Fix: apply SHAP key-string filtering to the test set at inference time,
so the distribution the Phase 3 model sees matches training exactly.

Pipeline
--------
  1. Load Phase 1 best.pt (fold 0) → extract SHAP key strings for test set ONCE
  2. Load each Phase 3 fold best.pt → predict on the same test key strings
  3. Ensemble: average softmax probabilities across all folds → argmax

SHAP is only run once (not per-fold) because all Phase 1 fold models are
trained on the same data and architecture; key strings are stable across folds.

Fallback: if a test key string is empty, use the original text.

Output: results/exp03/{run_id}/predictions.csv
Format: row_id,LABEL

Usage
-----
    python exp/exp03_shap_consistent/predict.py            # latest run
    python exp/exp03_shap_consistent/predict.py --run 01
    python exp/exp03_shap_consistent/predict.py --shap-fold 2  # use fold 2 for SHAP
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import shap
import torch
import yaml
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

    if not exp_results_dir.exists():
        raise FileNotFoundError(
            f"No results directory found: {exp_results_dir}\n"
            "Run train.py first to generate a run."
        )
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
# Data helpers
# ---------------------------------------------------------------------------

def load_test_csv(path: Path) -> tuple[list[str], list[str]]:
    row_ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_ids.append(row["row_id"].strip())
            texts.append(row["TEXT"].strip())
    return row_ids, texts


def tokenise(texts: list[str], tokenizer, max_length: int) -> dict:
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


# ---------------------------------------------------------------------------
# SHAP key-string extraction (run once for all test texts)
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


def extract_key_strings(
    texts: list[str],
    model: AttentionPoolingClassifier,
    tokenizer,
    max_length: int,
    shap_max_evals: int,
    shap_label_idx: int | None,
    device: torch.device,
) -> list[str]:
    """Return SHAP key strings for a list of texts (same logic as training)."""
    predict_fn = make_predict_fn(model, tokenizer, max_length, device)
    masker     = shap.maskers.Text(r"\W+")
    explainer  = shap.Explainer(predict_fn, masker, output_names=["negative", "positive"])

    key_strings: list[str] = []
    for i, text in enumerate(texts):
        stripped_words = [w for w in re.split(r"\W+", text) if w]
        if len(stripped_words) < 2:
            key_strings.append(" ".join(stripped_words) or text)
            continue

        sv        = explainer([text], max_evals=shap_max_evals)
        words     = list(sv.data[0])
        values_2d = sv.values[0]  # (n_words, 2)

        if shap_label_idx is not None:
            cls = shap_label_idx
        else:
            probs = predict_fn([text])[0]
            cls   = int(probs.argmax())

        word_shap  = values_2d[:, cls]
        key_set    = {idx for idx, v in enumerate(word_shap) if v > 0}
        key_string = " ".join(words[idx] for idx in sorted(key_set))

        key_strings.append(key_string if key_string.strip() else text)

        if (i + 1) % 50 == 0:
            print(f"    SHAP processed {i + 1}/{len(texts)}")

    return key_strings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",       type=str, default=None, help="Run ID (e.g. 01). Defaults to latest.")
    parser.add_argument("--shap-fold", type=int, default=None, help="Which Phase 1 fold to use for SHAP (default: best val_acc fold).")
    args = parser.parse_args()

    cfg    = load_config(EXP_DIR / "config.yaml")
    hp     = flatten_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name        = cfg["experiment"]["name"]
    exp_results_dir = RESULTS_DIR / exp_name
    run_dir, run_id = resolve_run_dir(exp_results_dir, args.run)

    print(f"Experiment  : {exp_name}")
    print(f"Run         : {run_id}")
    print(f"Device      : {device}")

    pretrained     = MODEL_REGISTRY[hp["name"]]
    tokenizer      = AutoTokenizer.from_pretrained(pretrained)
    shap_max_evals = cfg.get("extraction", {}).get("shap_max_evals", 500)
    shap_label_idx = cfg.get("extraction", {}).get("shap_label_idx", None)

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_FILE}")
    row_ids, test_texts = load_test_csv(TEST_FILE)
    print(f"Test samples: {len(test_texts)}")

    # Determine which Phase 1 fold to use for SHAP extraction.
    # Default: the fold with the highest Phase 1 val_acc from cv_summary.json
    # (that file records Phase 3 results; Phase 1 best fold is inferred by
    # checking which phase1/fold{k}/best.pt was saved — we use cv_summary's
    # fold ordering as a proxy, defaulting to fold 0 if summary unavailable).
    if args.shap_fold is not None:
        shap_fold = args.shap_fold
        print(f"SHAP fold   : {shap_fold} (user-specified)")
    else:
        # Read Phase 1 val_acc from cv_summary if available, else fold 0
        import json as _json
        cv_path = run_dir / "cv_summary.json"
        shap_fold = 0
        if cv_path.exists():
            with open(cv_path, encoding="utf-8") as _f:
                _summary = _json.load(_f)
            # cv_summary records Phase 3 fold results; use as proxy for Phase 1
            # A cleaner alternative is to track Phase 1 metrics separately,
            # but for now pick the fold whose Phase 3 val_acc is highest
            # (Phase 1 and Phase 3 rankings are generally correlated).
            fold_results = _summary.get("fold_results", [])
            if fold_results:
                shap_fold = max(fold_results, key=lambda r: r["best_val_acc"])["fold"]
        print(f"SHAP fold   : {shap_fold} (best val_acc fold)")

    # -----------------------------------------------------------------------
    # Step 1: SHAP extraction — run ONCE using Phase 1 best fold
    # -----------------------------------------------------------------------
    phase1_ckpt = run_dir / "phase1" / f"fold{shap_fold}" / "best.pt"
    if not phase1_ckpt.exists():
        raise FileNotFoundError(f"Phase 1 checkpoint not found: {phase1_ckpt}")

    print(f"\nLoading Phase 1 fold {shap_fold} for SHAP extraction...")
    phase1_model = AttentionPoolingClassifier(
        pretrained_name=pretrained,
        hidden_dim=hp["hidden_dim"],
        mlp_hidden=hp["mlp_hidden"],
        dropout=hp["dropout"],
        num_labels=hp["num_labels"],
        freeze_encoder=True,
    ).to(device)
    phase1_model.load_state_dict(torch.load(phase1_ckpt, map_location=device, weights_only=True))
    phase1_model.eval()

    print(f"Extracting SHAP key strings for {len(test_texts)} test samples...")
    key_strings = extract_key_strings(
        test_texts, phase1_model, tokenizer,
        hp["max_length"], shap_max_evals, shap_label_idx, device,
    )
    del phase1_model  # free memory before loading Phase 3 models

    # -----------------------------------------------------------------------
    # Step 2: encode key strings once, then run all Phase 3 fold models
    # -----------------------------------------------------------------------
    enc            = tokenise(key_strings, tokenizer, hp["max_length"])
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    n_folds   = hp["n_folds"]
    all_probs: list[np.ndarray] = []

    print(f"\nRunning Phase 3 ensemble ({n_folds} folds)...")
    for fold in range(n_folds):
        phase3_ckpt = run_dir / f"fold{fold}" / "best.pt"
        if not phase3_ckpt.exists():
            raise FileNotFoundError(f"Phase 3 checkpoint not found: {phase3_ckpt}")

        model = AttentionPoolingClassifier(
            pretrained_name=pretrained,
            hidden_dim=hp["hidden_dim"],
            mlp_hidden=hp["mlp_hidden"],
            dropout=hp["dropout"],
            num_labels=hp["num_labels"],
            freeze_encoder=True,
        ).to(device)
        model.load_state_dict(torch.load(phase3_ckpt, map_location=device, weights_only=True))
        model.eval()
        print(f"  Loaded fold {fold}: {phase3_ckpt}")

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)

    ensemble_probs = np.mean(all_probs, axis=0)
    predictions    = ensemble_probs.argmax(axis=1).tolist()

    out_path = run_dir / "predictions.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "LABEL"])
        for rid, pred in zip(row_ids, predictions):
            writer.writerow([rid, pred])

    print(f"\nPredictions saved to: {out_path}")
    print(f"  Total predictions : {len(predictions)}")
    pos = sum(predictions)
    print(f"  Positive (1)      : {pos}  ({pos / len(predictions):.1%})")
    print(f"  Negative (0)      : {len(predictions) - pos}  ({(len(predictions) - pos) / len(predictions):.1%})")


if __name__ == "__main__":
    main()
