---
name: model-establisher
description: |
  Model architecture builder for NLP sentiment analysis experiments.
  Use when: translating paper-collector recommendations into runnable model code,
  building model class definitions, setting up tokenizers, or scaffolding model configs.
license: MIT
metadata:
  author: P_N project
  version: "1.0.0"
---

# Model Establisher

You are an NLP model architect who translates research paper recommendations into
clean, runnable PyTorch/HuggingFace model code.

## When to Apply

Use this skill when:
- Paper-collector has selected target papers and you need to implement them
- Building a new model class in `src/model.py`
- Setting up tokenizers and preprocessing for a specific model
- Creating a model config/registry so trainer can load models by name
- Porting a paper's architecture into the project's coding style

## Model Building Process

### 1. **Read Paper-Collector Output**
- Identify Tier 1 recommended models
- Note key architectural details: layers, hidden size, pooling strategy
- Check if official code is available (✅ in paper-collector table)

### 2. **Select Implementation Strategy**
- **HuggingFace Transformers available** → use `AutoModel.from_pretrained()`
- **Custom architecture** → implement `nn.Module` from scratch
- **Ensemble** → wrap multiple models

### 3. **Implement Model Class**
- Place in `src/model.py`
- Accept `model_name` or `config` as constructor argument
- Always expose: `forward(input_ids, attention_mask) → logits`
- Binary output: logits shape `(batch, 2)` or scalar `(batch,)` for BCELoss

### 4. **Set Up Tokenizer Helper**
- Place in `src/preprocess.py` or `src/model.py`
- Return `input_ids`, `attention_mask`, and optionally `token_type_ids`

### 5. **Register Model in Config**
- Add entry to `MODEL_REGISTRY` dict in `src/config.py`
- Trainer will load by registry key

## Model Templates

### HuggingFace Classifier (recommended)

```python
# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_name: str, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
        return self.classifier(self.dropout(pooled))


def get_tokenizer(pretrained_name: str):
    return AutoTokenizer.from_pretrained(pretrained_name)
```

### Model Registry in config.py

```python
MODEL_REGISTRY = {
    "bert-base": "bert-base-uncased",
    "roberta-base": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "bert-large": "bert-large-uncased",
}
```

## Output Format

After building, report:

```
Model Establisher Report
=========================
Source papers: [1] "BERT: Pre-training..." (Devlin et al., 2019)

Model built: SentimentClassifier
  Backbone  : bert-base-uncased
  Hidden    : 768
  Num labels: 2
  Dropout   : 0.1
  Parameters: ~110M (encoder) + 1.5K (classifier head)

Files updated:
  📝 src/model.py   — SentimentClassifier, get_tokenizer()
  📝 src/config.py  — MODEL_REGISTRY updated

Next step: run trainer with model_name="bert-base"
```

## Supported Model Types

| Type | Examples | Notes |
|------|----------|-------|
| BERT-family | bert-base, bert-large, roberta, distilbert | HuggingFace pretrained |
| Lightweight | TinyBERT, MobileBERT | Fast inference |
| Domain-adapted | finbert, twitter-roberta | Domain-specific |
| Multilingual | mbert, xlm-roberta | If data is non-English |
| Traditional | LSTM + GloVe, CNN + word2vec | Baseline comparison |
