"""
config.py — Project-wide constants, model registry, and experiment hyperparameters.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root & directory constants
# ---------------------------------------------------------------------------

PROJECT_ROOT   = Path(__file__).resolve().parent.parent   # c:/.../P_N
DATA_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = DATA_DIR / "train_2022.csv"
TEST_FILE  = PROJECT_ROOT / "tests" / "test.csv"
RESULTS_DIR    = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
LOGS_DIR       = RESULTS_DIR / "logs"
FIGURES_DIR    = RESULTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Model registry
# Maps short alias → HuggingFace model identifier
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    # Experiment 01 backbone (frozen SBERT encoder)
    "sbert-minilm":  "sentence-transformers/all-MiniLM-L6-v2",

    # General BERT-family baselines (available for future experiments)
    "bert-base":     "bert-base-uncased",
    "bert-large":    "bert-large-uncased",
    "roberta-base":  "roberta-base",
    "distilbert":    "distilbert-base-uncased",
}

# ---------------------------------------------------------------------------
# Exp01 — Input Reduction / Key String extraction
# ---------------------------------------------------------------------------

EXP01_HPARAMS = {
    # Training
    "model_name":    "sbert-minilm",            # key into MODEL_REGISTRY
    "learning_rate": 2e-4,
    "batch_size":    32,
    "epochs":        10,
    "max_length":    128,                        # max token length for tokeniser
    "seed":          42,
    "optimizer":     "AdamW",
    "scheduler":     None,                       # no LR scheduler

    # Model head
    "hidden_dim":    384,                        # SBERT all-MiniLM-L6-v2 output dim
    "mlp_hidden":    64,                         # MLP intermediate dimension
    "dropout":       0.1,
    "num_labels":    2,                          # binary: 0=negative, 1=positive

    # Cross-validation
    "n_folds":       3,

    # Key-string extraction
    "confidence_threshold": 0.95,

    # Output paths (relative to RESULTS_DIR)
    "checkpoint_dir":   "checkpoints/exp01",
    "log_dir":          "logs/exp01",
    "key_strings_path": "exp01/key_strings.json",
    "token_imp_path":   "exp01/token_importance.csv",
    "figures_dir":      "figures/exp01",
}
