# Exp03 — SHAP-Consistent Inference

**對應概念**：解決 exp02 的 train/test 分布不一致問題，同時加入 warm-start 與 CosineAnnealingLR  
**改進自**：[exp02_shap_retrain](../exp02_shap_retrain/README.md)  
**目標**：讓訓練與預測時的輸入分布完全一致，縮小 val_acc → test_acc 的差距

---

## 問題分析（exp02 的瓶頸）

| 指標 | exp01 | exp02 |
|------|-------|-------|
| val_acc | 0.758 | 0.767 |
| test_acc | 73.80% | **70.99%** |
| val→test gap | ~2pp | **~6pp** |

exp02 的 Phase 3 用 SHAP key string 訓練，但 `predict.py` 直接用原始文本推論。  
模型學到「從稀疏 key string 分類」，但測試時面對完整文本 → **分布不一致**，泛化力下降。

---

## 新方法：兩個改進

### 改進 1 — Phase 3 Warm-Start

Phase 3 初始化時，載入**同一 fold 的 Phase 1 best.pt** 作為起點，而非 random init。

```
Phase 1 fold k  ────────────────────────────────────────────────────────
  train/val split (seed=42)   →   model 訓練完成   →  best.pt
                                                          │
                                            warm-start 載入同一份 best.pt
                                                          │
Phase 3 fold k  ────────────────────────────────────────────────────────
  相同 train/val split        →   SHAP key string 微調   →  best.pt
```

> **重點**：Phase 1 與 Phase 3 的 fold k 使用**完全相同的 train/val indices**（同一個 `StratifiedKFold(seed=42)` 產生的 `splits`），因此 warm-start 載入的模型就是「在這份 train set 上已收斂的版本」。Phase 3 不是重新學習，而是接著學習 key string 的分布。

**效果**：attention + MLP 已學到基本情緒特徵，Phase 3 只需適應 key string 分布。

### 改進 2 — CosineAnnealingLR（Phase 3）

```
lr: 2e-4 → (cosine decay) → 2e-6  (over 30 epochs)
```

避免固定 lr 在後期 epoch 引起的震盪，讓 key string 微調更穩定收斂。

### 改進 3 — SHAP-Consistent Inference（核心改進）

`predict.py` 對每個 fold：
1. 用 **Phase 1 模型** 對 test 文本做 SHAP 提取 → test key strings
2. 用 **Phase 3 模型** 對 test key strings 做預測

```
test text
  → Phase 1 model + SHAP → test key string
  → Phase 3 model → softmax prob
  → 5-fold ensemble → argmax → LABEL
```

train 與 test 同樣接收 SHAP key string，分布完全一致。

---

## 資料集設定

| 項目 | 說明 |
|------|------|
| 語言 | 英文 |
| 來源檔案 | `data/raw/train_2022.csv` |
| 切割方式 | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |

---

## 模型架構

與 exp02 相同：Attention Pooling Classifier（SBERT frozen）。

```
輸入文本（Phase 1：原文；Phase 3：SHAP key string）
  → SBERT (frozen, sentence-transformers/all-MiniLM-L6-v2)
  → Token embeddings [n_tokens × 384]
  → Attention Layer: Linear(384→1) + masked softmax
  → Weighted Sum → [384]
  → MLP: Linear(384→64) → ReLU → Dropout → Linear(64→2)
  → logits → 0 / 1
```

---

## 三 Phase 流程

### Phase 1 — 初始訓練（原始文字）

與 exp02 相同，5-fold CV，checkpoint 存於 `run_dir/phase1/fold{k}/best.pt`。

### Phase 2 — SHAP Key String 擷取

與 exp02 相同，`shap.PartitionExplainer`，word-level masking。

### Phase 3 — SHAP 過濾後重訓（改進）

| 步驟 | 說明 |
|------|------|
| 輸入替換 | 以 key string 取代每筆訓練樣本的原文；key string 為空時 fallback 回原文 |
| **Warm-start** | 載入同一 fold 的 Phase 1 best.pt 作為初始化 |
| **LR Scheduler** | `CosineAnnealingLR(T_max=30, eta_min=lr*0.01)` |
| Tokenize | `max_length=64` |
| Checkpoint | `run_dir/fold{k}/best.pt` |

### Predict — SHAP-Consistent Inference（改進）

| 步驟 | 說明 |
|------|------|
| **SHAP 只跑一次** | 用 Phase 1 fold 0 模型對 test 文本做 SHAP 提取（不需 per-fold） |
| Phase 3 Ensemble | 5 個 fold 的 Phase 3 模型都對同一份 test key strings 預測 |
| Ensemble | 5 個 fold 的 softmax 平均 → argmax |

> `--shap-fold` 可指定哪個 Phase 1 fold 做 SHAP（預設 fold 0）。

---

## 參數設定（config.yaml）

與 exp02 相同，修改後執行 `train.py` 即產生新 run。

---

## 實驗腳本

```bash
# Step 1：三 phase 一次完成
python exp/exp03_shap_consistent/train.py

# Step 2：SHAP-consistent 預測
python exp/exp03_shap_consistent/predict.py            # 預設最新 run
python exp/exp03_shap_consistent/predict.py --run 01   # 指定 run
```

---

## 實驗產出結構

```
results/
  experiment_log.csv
  exp03/
    experiment_log.json
    01/
      config.yaml
      cv_summary.json
      key_strings.json
      phase1/
        fold0/  fold1/  fold2/  fold3/  fold4/
      fold0/  fold1/  fold2/  fold3/  fold4/
      predictions.csv
```

---

## 與 exp02 的關係

| | exp02 | exp03 |
|---|---|---|
| Phase 3 初始化 | Random init | **Warm-start from Phase 1** |
| Phase 3 LR | Fixed AdamW | **CosineAnnealingLR** |
| 預測輸入 | 原始文本 | **SHAP key string（與 train 一致）** |
| 預期效果 | val 76.7%，test 71.0% | val→test gap 縮小 |
