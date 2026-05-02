# Exp11 — 獨立生成 Pseudo-labels + 5-fold Ensemble

**設計原則**：Pseudo-labels 由實驗本身獨立生成（Phase 0），不依賴任何外部預測檔案

---

## 兩階段流程

```
Phase 0  用全部 labeled data 訓練一個模型（fixed epochs，無 EarlyStop）
         → 預測 test set → 取得 pseudo-labels（全部使用，無 threshold）
         → 釋放 GPU 記憶體

Phase 1  5-fold CV
  for each fold:
    train_set = labeled_fold + ALL pseudo-labels from Phase 0
    class_weights = balanced(train_set)
    model = roberta-large (AutoModelForSequenceClassification)
    WeightedLossTrainer + EarlyStop(patience=2) + FP16
    → save best_model/
    → collect test softmax probs

predictions = mean softmax (5 folds) → argmax
```

---

## 超參數

```yaml
# Phase 0 (pseudo-label generation)
pseudo_gen:
  epochs: 3
  batch_size: 8
  gradient_accumulation: 2
  max_length: 192
  warmup_ratio: 0.1
  weight_decay: 0.05

# Phase 1 (5-fold training)
training:
  learning_rate: 1e-5
  batch_size: 8
  gradient_accumulation: 2   # effective batch = 16
  epochs: 3
  max_length: 192
  weight_decay: 0.05
  warmup_ratio: 0.1
```

---

## 產出結構

```
results/exp11/
  experiment_log.json
  01/
    config.yaml
    cv_summary.json
    predictions.csv
    phase0_ckpts/     ← Phase 0 temporary checkpoints
    fold1/ ~ fold5/
      ckpts/          ← HF Trainer checkpoints
      best_model/     ← HF format checkpoint
```

---

## 執行方式

```bash
python exp/exp11_pseudo_single_stage/train.py

# 重新從 checkpoint 生成預測
python exp/exp11_pseudo_single_stage/predict.py
python exp/exp11_pseudo_single_stage/predict.py --run 01
```
