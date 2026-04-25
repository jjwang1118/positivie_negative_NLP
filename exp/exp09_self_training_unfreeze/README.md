# Exp09 — Self-training + Partial Encoder Unfreeze

**基礎**：exp08（siebert frozen encoder + cosine scheduler + label smoothing）  
**新增**：方向 A（部分解凍 encoder）+ 方向 B（模型信度驅動的迭代偽標籤）  
**目標**：突破 exp08 的 74.4% 天花板，讓 encoder 自適應我們的資料分布

---

## 與 exp08 的核心差異

| | exp08 | exp09 |
|---|---|---|
| Encoder 狀態 | 完全凍結 | Stage1 凍結 → Stage2+ 解凍最後 3 層 |
| 偽標籤來源 | K-means cluster purity | 分類模型 softmax confidence |
| 迭代 | 一次（K-means → assign） | 3 個 stage，每 stage 重新生成偽標籤 |
| Encoder LR | 無 | 5e-6（比 head 小 20x，防止 catastrophic forgetting） |
| Head LR | 1e-4 | 1e-4（Stage2）→ 5e-5（Stage3） |
| 可訓練參數 | ~65k（MLP only） | ~65k + 最後 3 layers（~50M 額外） |

---

## 每個 Fold 的三階段流程

```
Stage 1  凍結 encoder  ─────────────────────────────────────
  訓練集 = 1,600 labeled
  目標   = 訓練穩定的初始分類器
  產出   = stage1_best.pt + pseudo_pool_S1 (confidence ≥ 0.95)

Stage 2  解凍最後 3 layers  ─────────────────────────────────
  訓練集 = 1,600 labeled + pseudo_pool_S1 (~4,000-6,000 samples)
  Warm-start from Stage 1 head weights
  Encoder LR = 5e-6  /  Head LR = 1e-4  (layerwise decay)
  目標   = 讓 encoder 表示適應本資料集的情緒模式
  產出   = stage2_best.pt + pseudo_pool_S2 (confidence ≥ 0.90)

Stage 3  (enabled=true 時執行)  ────────────────────────────
  訓練集 = 1,600 labeled + pseudo_pool_S2 (更多，threshold 放寬)
  Warm-start from Stage 2 best weights
  Encoder LR = 2e-6  /  Head LR = 5e-5  (再降低，細調)
  目標   = 最後一輪精煉，收斂至更好的局部最優
  產出   = stage3_best.pt（= fold 的最終 best.pt）
```

---

## 為何這樣設計

### 為何 Stage 1 保持凍結？
- 只有 1,600 labeled 樣本，解凍 355M 參數的 encoder 會嚴重過擬合
- 先用凍結 encoder 建立穩定的初始 classifier，再用其偽標籤擴大訓練集後才解凍

### 為何偽標籤從 K-means 換成 confidence thresholding？
- K-means 的幾何距離不直接對齊分類目標
- 模型 softmax confidence 直接反映分類任務的不確定性
- 每個 stage 的偽標籤品質會隨 model 提升而提升

### 為何 Encoder LR 比 Head LR 小 20 倍？
- siebert 的 encoder 已有大量 sentiment 知識，過大的 LR 會破壞它（catastrophic forgetting）
- 小的 encoder LR → 細微調整而非重置

### Threshold 逐步放寬的邏輯
```
Stage 1 model → 用 1,600 labeled 訓練，偏誤較大 → 只信任最高 confidence (0.95)
Stage 2 model → 用 6,000+ 樣本訓練，偏誤減小 → 可接受 0.90
(Stage 3 threshold = Stage 2 產出，不再生成新一輪)
```

---

## 超參數

```yaml
stage1:
  confidence_threshold: 0.95    # pseudo-label filter for S1→S2
  epochs: 20
  learning_rate: 1e-4

stage2:
  unfreeze_last_n: 3            # last 3 of 24 RoBERTa-large layers
  encoder_lr: 5e-6
  head_lr: 1e-4
  confidence_threshold: 0.90   # pseudo-label filter for S2→S3
  epochs: 20

stage3:
  enabled: true
  encoder_lr: 2e-6
  head_lr: 5e-5
  epochs: 15
```

---

## 產出結構

```
results/exp09/
  experiment_log.json
  01/
    config.yaml
    cv_summary.json
    fold0/
      stage1_best.pt   stage1_log.csv
      stage2_best.pt   stage2_log.csv
      stage3_best.pt   stage3_log.csv
      best.pt          ← copy of stage3_best.pt (used by predict.py)
    fold1/  fold2/  fold3/  fold4/
    predictions.csv
```

---

## 執行方式

```bash
# 訓練（每個 fold 執行 3 個 stage）
python exp/exp09_self_training_unfreeze/train.py

# 預測（5-fold ensemble of final stage best.pt）
python exp/exp09_self_training_unfreeze/predict.py
python exp/exp09_self_training_unfreeze/predict.py --run 01
```

---

## 調參建議

| 問題 | 調整方向 |
|---|---|
| Stage 2 val_acc 低於 Stage 1 | 縮小 encoder_lr（5e-6 → 2e-6）或減少 unfreeze_last_n（3 → 2） |
| pseudo_S1 數量太少（< 3,000） | 放寬 stage1.confidence_threshold（0.95 → 0.90） |
| pseudo_S1 數量太多但品質差 | 嚴格 threshold（0.95 → 0.97）或增加 Stage 1 epochs |
| Stage 3 比 Stage 2 差 | 停用 Stage 3（stage3.enabled: false） |
