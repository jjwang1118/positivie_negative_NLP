# Exp09-05 詳細實驗方案

**實驗名稱**：Self-training + Partial Encoder Unfreeze  
**Run ID**：05  
**時間戳**：2026-04-28T04:58:05  
**最終結果**：mean_val_acc = **0.8315** ± 0.0163

---

## 一、核心思路

在資料量有限（1,600 labeled samples）的情況下，直接微調整個 encoder 會嚴重過擬合。  
Exp09 採取**漸進解凍（progressive unfreezing）＋信度驅動偽標籤（confidence-based pseudo-labeling）**的三階段策略：

1. **Stage 1**：凍結 encoder，先用 labeled data 訓練穩定的分類頭，並篩選高信度偽標籤
2. **Stage 2**：解凍 encoder 最後 3 層，以擴充後的資料微調，讓 encoder 適應本資料集分布
3. **Stage 3**：小幅精煉（更低 LR），收斂至更佳局部最優

---

## 二、模型架構

| 項目 | 設定 |
|------|------|
| Base model | `roberta-large` |
| 分類頭結構 | Linear(1024 → 64) → ReLU → Dropout(0.1) → Linear(64 → 2) |
| 總參數量（Stage 1） | ~65k（MLP only） |
| 總參數量（Stage 2+） | ~65k + last 3 RoBERTa-large layers（≈ 50M 額外） |

---

## 三、訓練流程

### Cross-validation 設定
- 5-fold CV（seed=42）
- 每個 fold 各自獨立執行完整三階段
- 最終預測 = 5 個 fold 的 best model softmax 平均（ensemble）

### Stage 1 — 凍結 encoder 訓練
```
目標：建立穩定初始分類器 + 生成初始偽標籤
資料：1,600 labeled samples（train fold）
```

| 超參數 | 值 |
|--------|----|
| Learning rate | 1e-4 |
| Batch size | 32 |
| Epochs | 10 |
| Max length | 128 |
| Label smoothing | 0.1 |
| Warmup ratio | 0.1 |
| Confidence threshold（偽標籤篩選）| 0.90 |

> **注意**：run05 中所有 fold 的 `n_pseudo_s1 = 0`，原因是 0.90 閾值對 Stage 1 模型仍偏嚴格，無偽標籤被引入 Stage 2 初始訓練（實際 Stage 2 仍透過 Stage 2 自身的機制擴充資料）。

### Stage 2 — 解凍最後 3 層微調
```
目標：讓 encoder 表示適應本資料集的情緒模式
資料：1,600 labeled + pseudo_pool_S1（threshold=0.90 篩選，本次為 0）
      ＋ Stage 2 過程中動態累積的高信度偽標籤（mean_pseudo_s2 ≈ 3,194）
```

| 超參數 | 值 |
|--------|----|
| Encoder LR | 1e-5（比 head 小 10x，防止 catastrophic forgetting） |
| Head LR | 1e-4 |
| Batch size | 32 |
| Epochs | 10 |
| Max length | 128 |
| Label smoothing | 0.1 |
| Warmup ratio | 0.1 |
| Unfreeze last N layers | 3（共 24 層） |
| Confidence threshold（偽標籤篩選）| 0.90 |

### Stage 3 — 低 LR 精煉
```
目標：最後一輪細調，收斂至更佳局部最優
資料：1,600 labeled + pseudo_pool_S2（Stage 2 生成，threshold=0.90）
warm-start from Stage 2 best weights
```

| 超參數 | 值 |
|--------|----|
| Encoder LR | 2e-6 |
| Head LR | 5e-5 |
| Batch size | 32 |
| Epochs | 15 |
| Max length | 128 |
| Label smoothing | 0.1 |
| Warmup ratio | 0.05 |

---

## 四、各 Fold 結果

| Fold | Stage1 Acc | Stage2 Acc | Best Val Acc | Best Val Loss | n_pseudo_s1 | n_pseudo_s2 |
|------|-----------|-----------|-------------|--------------|------------|------------|
| 0 | 0.7550 | 0.8325 | **0.8325** | 0.4827 | 0 | 3,040 |
| 1 | 0.7650 | 0.8225 | **0.8425** | 0.4847 | 0 | 3,036 |
| 2 | 0.7475 | 0.7875 | **0.8000** | 0.5258 | 0 | 2,879 |
| 3 | 0.7400 | 0.8225 | **0.8375** | 0.4812 | 0 | 3,563 |
| 4 | 0.7300 | 0.8300 | **0.8450** | 0.4636 | 0 | 3,451 |
| **Mean** | 0.7475 | 0.8210 | **0.8315** | — | **0.0** | **3,193.8** |
| **Std** | — | — | 0.0163 | — | — | — |

---

## 五、偽標籤策略說明

```
Stage 1 model（1,600 labeled 訓練）
  → softmax confidence ≥ 0.90 的 test/unlabeled samples → pseudo_pool_S1
  → 本次 run05 全部 fold 為 0（閾值對初始模型偏嚴）

Stage 2 model（labeled + pseudo_pool_S1 訓練）
  → softmax confidence ≥ 0.90 → pseudo_pool_S2（平均 3,194 筆）

Stage 3 使用 pseudo_pool_S2 作為訓練資料的一部分（不再生成新一輪）
```

**Threshold 設計邏輯**：
- Stage 1 模型偏誤較大 → 只信任最高 confidence（原設計 0.95，run05 已放寬至 0.90）
- Stage 2 模型用更多資料訓練，品質更好 → 放寬至 0.90 可接受更多偽標籤

---

## 六、與前版本 (run02) 的差異

| 項目 | run02 | run05 |
|------|-------|-------|
| Base model | siebert (sentiment-roberta-large) | roberta-large |
| Stage1 threshold | 0.95 → n_pseudo=0 | 0.90 → 仍為 0（但有 S2 偽標籤） |
| Stage1 epochs | 20 | 10 |
| Stage2 encoder_lr | 5e-6 | 1e-5 |
| Stage2 epochs | 20 | 10 |
| mean_val_acc | 0.8565 | 0.8315 |

---

## 七、產出檔案結構

```
results/exp09/05/
  config.yaml         ← 完整超參數
  cv_summary.json     ← 所有 fold 結果匯總
  predictions.csv     ← 最終預測（5-fold ensemble softmax argmax）
  fold0/
    stage1_best.pt
    stage1_log.csv
    stage2_best.pt
    stage2_log.csv
    stage3_best.pt
    stage3_log.csv
    best.pt           ← = stage3_best.pt（供 predict.py 讀取）
  fold1/ ~ fold4/     ← 同上
```

---

## 八、執行指令

```bash
# 訓練
python exp/exp09_self_training_unfreeze/train.py

# 使用 run05 重新預測
python exp/exp09_self_training_unfreeze/predict.py --run 05
```
