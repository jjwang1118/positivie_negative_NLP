# Exp11-01 詳細實驗方案
> 此文件被 `notebooks/report_framework.md` 引用，作為 **EXP02** 方案的參考來源。

**實驗名稱**：獨立生成 Pseudo-labels + 5-fold Ensemble（Single-Stage）  
**Run ID**：01  
**時間戳**：2026-04-27T02:29:40  
**最終結果**：mean_val_acc = **0.8805** ± 0.0153，mean_val_F1 ≈ 0.8810

---

## 一、核心思路

核心設計如下：
- Pseudo-labeling（test set 加入訓練，由 Phase 0 自行生成，不依賴外部檔案）
- Balanced class weights（WeightedTrainer）
- 縮寫修正預處理（`clean_for_transformer`）
- EarlyStopping（patience=2，以 val F1 為準）
- 5-fold softmax ensemble

---

## 二、模型架構

| 項目 | 設定 |
|------|------|
| Base model | `roberta-large`（HuggingFace `AutoModelForSequenceClassification`） |
| 分類頭 | HF 預設（Linear(1024 → 2)，無額外 MLP） |
| 精度 | FP16（mixed precision） |

---

## 三、訓練流程

### Phase 0 — 偽標籤生成
```
輸入：全部 labeled data（不做 fold 切割）
目標：訓練一個暖機模型，預測 test set 獲得 pseudo-labels
偽標籤策略：全部採用（無 confidence threshold 篩選）
產出：test set pseudo-labels（11,000 筆）→ 儲存至 phase0_ckpts/
```

| 超參數 | 值 |
|--------|----|
| Learning rate | 1e-5 |
| Batch size | 8 |
| Gradient accumulation steps | 2（effective batch = 16） |
| Epochs | 3 |
| Max length | 192 |
| Warmup ratio | 0.1 |
| Weight decay | 0.05 |

> Phase 0 結束後立即釋放 GPU 記憶體，再進入 Phase 1。

### Phase 1 — 5-fold 訓練
```
for each fold（fold 1 ~ fold 5）：
  train_set = labeled_fold_train + ALL Phase 0 pseudo-labels（11,000 筆）
  class_weights = balanced class weights（根據 train_set 計算）
  model = roberta-large（從 pre-trained 重新載入）
  WeightedLossTrainer + EarlyStoppingCallback(patience=2) + FP16
  → 每個 epoch 以 val F1 評估，保存最佳 checkpoint
  → 收集 test set 的 softmax probabilities

最終預測 = 5 個 fold 的 softmax probs 平均 → argmax
```

| 超參數 | 值 |
|--------|----|
| Learning rate | 1e-5 |
| Batch size | 8 |
| Gradient accumulation steps | 2（effective batch = 16） |
| Epochs | 3（含 EarlyStop） |
| Max length | 192 |
| Warmup ratio | 0.1 |
| Weight decay | 0.05 |
| n_folds | 5（seed=42） |

---

## 四、各 Fold 結果

| Fold | Val Acc | Val Loss | Val F1 |
|------|---------|----------|--------|
| 0 | 0.8725 | 0.6221 | 0.8765 |
| 1 | **0.9025** | 0.4379 | **0.9051** |
| 2 | 0.8650 | 0.6580 | 0.8670 |
| 3 | 0.8675 | **0.4105** | 0.8630 |
| 4 | 0.8950 | 0.4358 | 0.8934 |
| **Mean** | **0.8805** | — | **0.8810** |
| **Std** | 0.0153 | — | — |

偽標籤數量：**11,000 筆**（test set 全量，無篩選）

---

## 五、關鍵設計說明

### WeightedTrainer
- 計算各類別的 balanced class weights
- 在 loss function 中對少數類別給予更高權重
- 緩解訓練集中正負樣本不平衡問題

### clean_for_transformer 預處理
- 修正資料集中被切斷的縮寫，例如：
  - `"can t"` → `"cannot"`
  - `"won t"` → `"will not"`
- 使 tokenizer 能正確識別縮寫語意

### EarlyStoppingCallback(patience=2)
- 監控指標：val F1（非 val Loss）
- 最多允許 2 個 epoch 無改善即停止
- 搭配 3 epochs 上限，實際訓練通常在 2-3 epochs 完成

### 5-fold Softmax Ensemble
- 每個 fold 輸出 test set 的 softmax probabilities（shape: [n_test, 2]）
- 5 個 fold 取平均後 argmax 得最終預測
- 降低單一模型的 variance，提升泛化性

---

---

## 七、產出檔案結構

```
results/exp11/01/
  config.yaml           ← 完整超參數
  cv_summary.json       ← 所有 fold 結果匯總
  predictions.csv       ← 最終預測（5-fold ensemble softmax argmax）
  phase0_ckpts/         ← Phase 0 暫存 checkpoint
  fold1/
    ckpts/              ← HF Trainer 中間 checkpoints
    best_model/         ← HF format 最佳模型
  fold2/ ~ fold5/       ← 同上
```

---

## 八、執行指令

```bash
# 訓練（Phase 0 → Phase 1）
python exp/exp11_pseudo_single_stage/train.py

# 使用 run01 重新預測
python exp/exp11_pseudo_single_stage/predict.py --run 01
```
