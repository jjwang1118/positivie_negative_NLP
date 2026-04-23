# Exp07 — Cluster-based Pseudo-labeling (siebert/sentiment-roberta-large-english)

**方法**：K-means 分群 + 高純度群投票產生 pseudo-label，擴充訓練集  
**架構**：`MeanPoolingClassifier`（同 exp05/06）  
**差異**：Encoder 改用 `siebert/sentiment-roberta-large-english`（1024-dim，情緒 fine-tune）  
**目標**：情緒特化 encoder 使 K-means 分群軸對齊情緒，提高群純度與 pseudo-label 品質

---

## 與 exp06 的差異

| | exp06 | exp07 |
|---|---|---|
| Encoder | `all-MiniLM-L6-v2`（通用，384-dim）| `siebert/sentiment-roberta-large-english`（情緒特化，1024-dim）|
| Embedding 主軸 | 語意主題 | 情緒傾向 |
| hidden_dim | 384 | 1024 |
| 參數量 | 22M | 355M |

---

## 核心概念

`siebert/sentiment-roberta-large-english` 是在大量情緒標注資料上 fine-tune 過的 RoBERTa-large。其 embedding 空間相較通用模型更傾向以**情緒**作為分群軸，理論上能產生更高純度的 K-means 群，進而增加可用的 pseudo-label 數量與準確率。

```
13k 文本（2k labeled + 11k unlabeled）
  → siebert encoder (frozen) [13k × 1024]
  → K-means（K=100）→ 100 個群

每個群：
  有標籤樣本投票（只用 train split）
  purity ≥ 0.8 且 有標籤樣本 ≥ 3 → 群內無標籤樣本獲得 pseudo-label

訓練集 = 原始 labeled train (1,600) + pseudo-labeled
```

---

## 流程圖

```
Step 0 (optional): python analyze.py
  → t-SNE 視覺化（預期正負面點分布比 exp06 更分開）
  → purity histogram（預期高純度群更多）
  → dendrogram（預期藍/紅葉節點更集中於各自子樹）
  → 儲存至 results/figures/exp07/

Step 1: python train.py
  → 編碼 13k 文本（batch_size=32，frozen RoBERTa-large）
  → K-means（一次）
  → 5-fold CV

Step 2: python predict.py
  → 5-fold ensemble 預測 test.csv（批次推論，避免 OOM）
```

---

## 避免資料洩漏的設計

與 exp06 相同：
- K-means 完全無監督，可在 CV 前執行
- Pseudo-label 投票僅使用每 fold 的 train split 標籤

---

## 模型架構

```
輸入文本
  → siebert/sentiment-roberta-large-english (frozen, 355M params)
  → Token embeddings [B × T × 1024]
  → Mask-aware Mean Pooling → [B × 1024]
  → MLP: Linear(1024→64) → ReLU → Dropout(0.1) → Linear(64→2)
  → logits → 0 / 1
```

---

## 超參數

```yaml
clustering:
  n_clusters: 100
  purity_threshold: 0.8
  min_labeled_per_cluster: 3

model:
  hidden_dim: 1024
  mlp_hidden: 64
  dropout: 0.1

training:
  learning_rate: 2.0e-4
  batch_size: 32
  epochs: 30
  n_folds: 5
```

---

## 執行方式

```bash
# Step 0：分析 embedding 空間
python exp/exp07_cluster_pseudo_siebert/analyze.py

# Step 1：訓練
python exp/exp07_cluster_pseudo_siebert/train.py

# Step 2：預測
python exp/exp07_cluster_pseudo_siebert/predict.py
python exp/exp07_cluster_pseudo_siebert/predict.py --run 01
```

---

## 產出結構

```
results/
  experiment_log.csv
  figures/
    exp07/
      tsne_by_label.png
      purity_hist.png
      dendrogram_centroids.png
      pseudo_label_stats.txt
  exp07/
    experiment_log.json
    01/
      config.yaml
      cv_summary.json
      fold0/
        best.pt  last.pt  train_log.csv
      fold1/  fold2/  fold3/  fold4/
      predictions.csv
```
