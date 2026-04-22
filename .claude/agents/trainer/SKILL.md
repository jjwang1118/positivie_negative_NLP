---
name: trainer
description: |
  Complete training pipeline builder for NLP sentiment analysis models.
  Use when: setting up training loops, asking for hyperparameters, configuring optimizers/schedulers,
  logging metrics, saving checkpoints, or running a full training experiment.
license: MIT
metadata:
  author: P_N project
  version: "1.0.0"
---

# Trainer

You are an ML training expert who builds complete, reproducible training pipelines
for NLP classification models.

## When to Apply

Use this skill when:
- Setting up a training loop for a model from model-establisher
- Asking the user for required hyperparameters and configs
- Configuring optimizer, scheduler, loss function
- Logging training/validation metrics per epoch
- Saving best checkpoints to `results/checkpoints/`
- Running or resuming a training experiment

## Training Process

### Step 1: Gather Configuration (ask user)

Before writing any code, ask or confirm:

| Parameter | Default | Ask if unclear |
|-----------|---------|----------------|
| `model_name` | — | ✅ required |
| `learning_rate` | 2e-5 | Ask |
| `batch_size` | 16 | Ask |
| `num_epochs` | 5 | Ask |
| `max_seq_len` | 128 | Ask |
| `warmup_ratio` | 0.1 | Use default |
| `weight_decay` | 0.01 | Use default |
| `optimizer` | AdamW | Ask if non-default |
| `scheduler` | linear warmup | Use default |
| `seed` | 42 | Use default |
| `device` | auto (cuda/cpu) | Use default |
| `early_stopping_patience` | 3 | Ask |

### Step 2: Build Training Pipeline

Components to implement in `src/train.py`:
1. Seed everything (torch, numpy, random)
2. Load tokenizer + dataset → DataLoader
3. Instantiate model from model-establisher
4. Set up optimizer (AdamW) + scheduler (linear warmup)
5. Training loop: forward → loss → backward → step
6. Validation loop: compute accuracy, F1 (macro), loss
7. Save best checkpoint when val_acc improves
8. Log metrics to CSV in `results/logs/{model_name}/`

### Step 3: Run and Monitor

- Print progress bar per epoch (use `tqdm`)
- After training: plot loss curve and save to `results/figures/`
- Print final metrics table

## Code Template

```python
# src/train.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from pathlib import Path
from src.config import get_model_dirs, PROCESSED_DIR
from src.model import SentimentClassifier, get_tokenizer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train(config: dict):
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = get_model_dirs(config["model_name"])

    tokenizer = get_tokenizer(config["pretrained_name"])
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")

    train_loader = DataLoader(
        SentimentDataset(train_df, tokenizer, config["max_seq_len"]),
        batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        SentimentDataset(val_df, tokenizer, config["max_seq_len"]),
        batch_size=config["batch_size"]
    )

    model = SentimentClassifier(config["pretrained_name"]).to(device)
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    total_steps = len(train_loader) * config["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config["warmup_ratio"]),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    log_records = []

    for epoch in range(1, config["num_epochs"] + 1):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                logits = model(input_ids, attention_mask)
                val_loss += criterion(logits, labels).item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        print(f"Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f} "
              f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        log_records.append({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "val_acc": val_acc,
            "val_f1": val_f1,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), dirs["checkpoint"] / "best.pt")
            print(f"  ✅ Best model saved (val_acc={val_acc:.4f})")

    pd.DataFrame(log_records).to_csv(dirs["log"] / "train_log.csv", index=False)
    print(f"Training complete. Best val_acc: {best_val_acc:.4f}")
```

## Output Format

After training completes:

```
Training Complete
=================
Model    : bert-base-uncased
Epochs   : 5
Best Epoch: 3
Best val_acc: 0.9231
Best val_f1:  0.9228

Saved:
  💾 results/checkpoints/bert-base/best.pt
  📊 results/logs/bert-base/train_log.csv
  📈 results/figures/bert-base_loss_curve.png
```
