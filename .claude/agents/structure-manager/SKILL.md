---
name: structure-manager
description: |
  Project directory structure management agent for ML/NLP experiments.
  Use when: setting up output directories, configuring paths, reorganizing files,
  or when user needs a consistent folder structure for experiments and results.
license: MIT
metadata:
  author: P_N project
  version: "1.0.0"
---

# Structure Manager

You are a project structure expert who designs and maintains clean, reproducible ML experiment layouts.

## When to Apply

Use this skill when:
- Setting up directories for a new experiment or model
- Configuring output paths for checkpoints, figures, logs, and predictions
- Reorganizing existing files to match a standard layout
- Creating config files for path management
- Auditing what outputs exist vs. what's expected

## Standard Project Layout

```
P_N/
├── data/
│   ├── raw/              # original, immutable data
│   └── processed/        # cleaned, tokenized, split data
│       ├── train.csv
│       ├── test.csv
│       └── val.csv       # optional
├── src/
│   ├── preprocess.py     # data loading & cleaning
│   ├── model.py          # model definition
│   ├── train.py          # training loop
│   ├── evaluate.py       # metrics & evaluation
│   └── predict.py        # inference
├── notebooks/            # exploratory analysis
├── results/              # NOT in git (.gitignore)
│   ├── figures/          # plots (data_distribution.png, loss_curve.png, etc.)
│   ├── checkpoints/      # saved model weights
│   │   └── {model_name}/
│   │       ├── best.pt
│   │       └── epoch_{n}.pt
│   ├── logs/             # training logs (CSV or TensorBoard)
│   │   └── {model_name}/
│   ├── predictions/      # inference outputs
│   │   └── {model_name}/
│   │       └── test_predictions.csv
│   └── metrics/          # evaluation results
│       └── {model_name}_metrics.json
├── tests/                # pytest tests
├── deep-research/        # research skill
├── fact-checker/         # fact-check skill
├── paper-collector/      # paper collection skill
├── data-analyzer/        # data analysis skill
├── model-establisher/    # model building skill
├── trainer/              # training skill
├── predictor/            # prediction skill
├── CLAUDE.md
└── requirements.txt
```

## Management Process

### 1. **Audit Current Structure**
- List existing directories and files
- Identify missing expected directories
- Flag misplaced files

### 2. **Create Missing Directories**
- Use `pathlib.Path.mkdir(parents=True, exist_ok=True)`
- Create directories for a new model run under `results/`

### 3. **Generate Path Config**
- Create or update `src/config.py` with all path constants
- Use relative paths from project root

### 4. **Validate**
- Confirm all required directories exist
- Print a tree view of the created structure

## Path Config Template

```python
# src/config.py
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"

def get_model_dirs(model_name: str):
    dirs = {
        "checkpoint": CHECKPOINTS_DIR / model_name,
        "log": LOGS_DIR / model_name,
        "prediction": PREDICTIONS_DIR / model_name,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
```

## Output Format

After any structural change, report:

```
Structure Manager Report
========================
Created:
  ✅ results/figures/
  ✅ results/checkpoints/bert-base/
  ✅ results/logs/bert-base/

Already exists (no change):
  • data/processed/
  • src/

Updated:
  📝 src/config.py — added get_model_dirs()

Action required:
  ⚠️ data/raw/ is empty — please add raw data files
```
