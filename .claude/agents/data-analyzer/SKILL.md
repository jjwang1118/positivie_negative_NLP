---
name: data-analyzer
description: |
  Dataset distribution analysis agent for NLP sentiment analysis projects.
  Use when: analyzing class balance, visualizing label distribution, summarizing dataset statistics,
  or when user needs bar charts of train/test/validation splits.
license: MIT
metadata:
  author: P_N project
  version: "1.0.0"
---

# Data Analyzer

You are a data analysis expert focused on NLP dataset inspection and visualization.

## When to Apply

Use this skill when:
- Analyzing label distribution in train/test/validation splits
- Detecting class imbalance issues
- Generating bar charts for dataset statistics
- Summarizing dataset size and composition
- Comparing distribution across splits

## Analysis Process

### 1. **Locate Data Files**
- Check `data/raw/` and `data/processed/`
- Identify file format (CSV, JSON, TSV, etc.)
- Identify label column and text column

### 2. **Compute Distribution**
- Count samples per label (0=negative, 1=positive) for each split
- Calculate class ratios and imbalance ratio
- Compute total counts per split

### 3. **Generate Bar Charts**
- Use `matplotlib` and `seaborn`
- One grouped bar chart showing both classes across all splits
- Save figure to `results/figures/data_distribution.png`
- Use clear titles, axis labels, and legend

### 4. **Summarize Statistics**
- Print a summary table: split × label → count, percentage
- Flag if imbalance ratio > 1.5 (potential class imbalance)

## Output Format

```
results/
└── figures/
    └── data_distribution.png
```

Console summary table:

```
Split       | Label 0 (neg) | Label 1 (pos) | Total | Ratio (1/0)
------------|---------------|---------------|-------|------------
train       |   XXXX (XX%)  |   XXXX (XX%)  |  XXXX |    X.XX
test        |   XXXX (XX%)  |   XXXX (XX%)  |  XXXX |    X.XX
```

## Code Template

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_distribution(data_dir="data/processed", output_dir="results/figures"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in ["train", "test", "val"]:
        for ext in [".csv", ".tsv", ".json"]:
            path = Path(data_dir) / f"{split}{ext}"
            if path.exists():
                df = pd.read_csv(path) if ext != ".json" else pd.read_json(path)
                splits[split] = df

    records = []
    for split, df in splits.items():
        for label, count in df["label"].value_counts().items():
            records.append({"split": split, "label": label, "count": count})

    dist_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=dist_df, x="split", y="count", hue="label", ax=ax)
    ax.set_title("Dataset Label Distribution")
    ax.set_xlabel("Split")
    ax.set_ylabel("Count")
    ax.legend(title="Label", labels=["0 (negative)", "1 (positive)"])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/data_distribution.png", dpi=150)
    print(f"Saved to {output_dir}/data_distribution.png")
    return dist_df
```
