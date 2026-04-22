"""
exp/exp01_input_reduction/extract_key_strings.py
=================================================
SHAP-based key string extraction on each fold's val set.

For each sample, uses shap.PartitionExplainer to compute per-word Shapley values,
which measure each word's average marginal contribution to the model's prediction
across all possible word subsets. This avoids the distribution shift issue of
greedy token addition (model always sees a complete, re-encoded sentence).

Key string = words with positive SHAP value for the predicted class,
sorted in their original sentence order.

Loads checkpoints from:
  results/{exp_name}/{run_id}/fold{k}/best.pt

Saves results under the same run directory:
  results/{exp_name}/{run_id}/fold{k}/key_strings.json
  results/{exp_name}/{run_id}/fold{k}/token_importance.csv
  results/{exp_name}/{run_id}/key_strings.json       (aggregated)
  results/{exp_name}/{run_id}/token_importance.csv   (aggregated)

Usage
-----
    python exp/exp01_input_reduction/extract_key_strings.py           # latest run
    python exp/exp01_input_reduction/extract_key_strings.py --run 02
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import shap
import torch
import yaml
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
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_csv(path: Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["TEXT"].strip())
            labels.append(int(row["LABEL"].strip()))
    return texts, labels


def make_predict_fn(model: AttentionPoolingClassifier, tokenizer, hp: dict, device: torch.device):
    """
    Return a function: list[str] → np.ndarray of shape (n, 2).
    SHAP calls this repeatedly with masked versions of the input text.
    Each call re-tokenizes and re-encodes the text, so the model always
    sees a properly-formed sentence (no distribution shift).
    """
    @torch.no_grad()
    def predict(texts):
        enc = tokenizer(
            list(texts),
            max_length=hp["max_length"],
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


# ---------------------------------------------------------------------------
# SHAP extraction (one fold)
# ---------------------------------------------------------------------------

def extract_fold_shap(
    fold: int,
    val_indices: list[int],
    all_texts: list[str],
    all_labels: list[int],
    model: AttentionPoolingClassifier,
    tokenizer,
    hp: dict,
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    """
    Compute SHAP values for each val sample and derive key strings.

    Masking strategy: word-level (split on whitespace/punctuation).
    SHAP replaces absent words with empty string, then re-tokenizes.
    The model always receives a full-length padded sequence.

    Key string = words with positive SHAP value for the predicted class,
    in their original sentence order.
    """
    predict_fn = make_predict_fn(model, tokenizer, hp, device)

    # Word-level masker: splits on non-word characters (spaces, punctuation)
    masker    = shap.maskers.Text(r"\W+")
    explainer = shap.Explainer(
        predict_fn,
        masker,
        output_names=["negative", "positive"],
    )

    label_idx          = hp.get("shap_label_idx")   # None = use predicted class
    max_evals          = hp.get("shap_max_evals", 500)
    key_string_records: list[dict] = []
    token_imp_records:  list[dict] = []

    for i, global_idx in enumerate(val_indices):
        text  = all_texts[global_idx]
        label = all_labels[global_idx]

        # Compute SHAP values
        sv         = explainer([text], max_evals=max_evals)
        # sv.data[0]   : list of words (after masker tokenization)
        # sv.values[0] : (n_words, 2) — axis 1 = [neg_class, pos_class]
        words      = list(sv.data[0])
        values_2d  = sv.values[0]          # (n_words, 2)

        # Determine which class to inspect
        if label_idx is not None:
            cls = label_idx
        else:
            probs = predict_fn([text])[0]
            cls   = int(probs.argmax())

        word_shap = values_2d[:, cls]      # (n_words,)

        # Rank by absolute SHAP value (largest contribution first)
        ranked_indices = np.argsort(np.abs(word_shap))[::-1].tolist()

        # Key tokens: words that contribute positively to predicted class
        key_set     = {idx for idx, v in enumerate(word_shap) if v > 0}
        key_indices = [idx for idx in ranked_indices if idx in key_set]
        key_tokens  = [words[idx] for idx in key_indices]

        # Preserve original sentence order in key_string
        key_string  = " ".join(words[idx] for idx in sorted(key_set))

        total_words     = len(words)
        reduction_ratio = len(key_indices) / total_words if total_words > 0 else 1.0

        key_string_records.append({
            "fold":             fold,
            "index":            global_idx,
            "text":             text,
            "label":            label,
            "predicted_class":  cls,
            "key_tokens":       key_tokens,
            "key_string":       key_string,
            "token_count":      len(key_indices),
            "total_tokens":     total_words,
            "reduction_ratio":  round(reduction_ratio, 4),
        })

        for rank, idx in enumerate(ranked_indices, 1):
            token_imp_records.append({
                "fold":        fold,
                "index":       global_idx,
                "label":       label,
                "token":       words[idx],
                "shap_value":  round(float(word_shap[idx]), 6),
                "rank":        rank,
                "selected":    int(idx in key_set),
            })

        if (i + 1) % 10 == 0:
            print(f"    processed {i + 1}/{len(val_indices)}")

    return key_string_records, token_imp_records


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
    set_seed(hp["seed"])

    exp_name        = cfg["experiment"]["name"]
    exp_results_dir = RESULTS_DIR / exp_name
    run_dir, run_id = resolve_run_dir(exp_results_dir, args.run)

    print(f"Experiment : {exp_name}")
    print(f"Run        : {run_id}")
    print(f"Device     : {device}")
    print(f"SHAP max_evals: {hp.get('shap_max_evals', 500)}")

    pretrained = MODEL_REGISTRY[hp["name"]]
    tokenizer  = AutoTokenizer.from_pretrained(pretrained)

    all_texts, all_labels = load_csv(TRAIN_FILE)
    label_arr = np.array(all_labels)

    n_folds = hp["n_folds"]
    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=hp["seed"])

    all_key_strings: list[dict] = []
    all_token_imp:   list[dict] = []

    for fold, (_, val_idx) in enumerate(skf.split(all_texts, label_arr)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_folds}  |  val size={len(val_idx)}")

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
        model.eval()
        print(f"  Loaded: {ckpt_path}")

        ks_records, ti_records = extract_fold_shap(
            fold, val_idx.tolist(), all_texts, all_labels,
            model, tokenizer, hp, device,
        )

        fold_dir = run_dir / f"fold{fold}"
        with open(fold_dir / "key_strings.json", "w", encoding="utf-8") as f:
            json.dump(ks_records, f, ensure_ascii=False, indent=2)

        with open(fold_dir / "token_importance.csv", "w", newline="", encoding="utf-8") as f:
            fieldnames = ["fold", "index", "label", "token", "shap_value", "rank", "selected"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ti_records)

        avg_tok   = sum(r["token_count"] for r in ks_records) / len(ks_records)
        avg_ratio = sum(r["reduction_ratio"] for r in ks_records) / len(ks_records)
        print(f"  Avg key tokens  : {avg_tok:.2f}")
        print(f"  Avg reduction   : {avg_ratio:.2%}")

        all_key_strings.extend(ks_records)
        all_token_imp.extend(ti_records)

    # --- Aggregate ---
    with open(run_dir / "key_strings.json", "w", encoding="utf-8") as f:
        json.dump(all_key_strings, f, ensure_ascii=False, indent=2)

    with open(run_dir / "token_importance.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["fold", "index", "label", "token", "shap_value", "rank", "selected"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_token_imp)

    total     = len(all_key_strings)
    avg_tok   = sum(r["token_count"] for r in all_key_strings) / total
    avg_ratio = sum(r["reduction_ratio"] for r in all_key_strings) / total

    print(f"\n{'='*50}")
    print(f"Aggregated ({n_folds} folds)  run={run_id}")
    print(f"  Total samples      : {total}")
    print(f"  Avg key tokens     : {avg_tok:.2f}")
    print(f"  Avg reduction ratio: {avg_ratio:.2%}")
    print(f"  key_strings.json   -> {run_dir / 'key_strings.json'}")
    print(f"  token_importance   -> {run_dir / 'token_importance.csv'}")


if __name__ == "__main__":
    main()
