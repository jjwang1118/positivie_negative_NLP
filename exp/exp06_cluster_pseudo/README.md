# Exp06 — Cluster-based Pseudo-labeling

**方法**：K-means 分群 + 高純度群投票產生 pseudo-label，擴充訓練集  
**架構**：`MeanPoolingClassifier`（繼承自 exp05）  
**目標**：利用 11k 無標籤資料的 embedding 空間結構，自動標注高信心樣本，突破 ~0.767 的 val_acc 天花板

---

## 核心概念

SBERT embedding 空間中，語意相近的文本被聚在同一群。若某群裡的有標籤樣本幾乎都是同一類情緒，則該群的無標籤樣本也很可能是同一類。

```
13k 文本（2k labeled + 11k unlabeled）
  → SBERT 編碼 [13k × 384]
  → K-means（K=100）→ 100 個群

每個群：
  有標籤樣本投票（只用 train split）
  purity = max(pos, neg) / total
  purity ≥ 0.8 且 有標籤樣本 ≥ 3 → 群內無標籤樣本獲得 pseudo-label

訓練集 = 原始 labeled train (1,600) + pseudo-labeled
```

---

## 為什麼 K=100 而非 K=2

K=2 假設 SBERT embedding 空間天然沿情緒軸分成兩群，但 SBERT 是通用語意模型，分群軸可能是主題、長度等與情緒無關的特徵。K=100 讓每群更小、更純，投票結果才可信。

---

## 流程圖

```
Step 0 (optional): python analyze.py
  → t-SNE 視覺化 embedding 空間可分性
  → 各 purity 閾值下可產生多少 pseudo-labels
  → 儲存圖表至 results/figures/exp06/

Step 1: python train.py
  → 編碼 13k 文本（一次，共用於所有 fold）
  → K-means（一次）
  → 5-fold CV：
      每 fold 用 train split 投票 → pseudo-label → 訓練 → 驗證

Step 2: python predict.py
  → 5-fold ensemble 預測 test.csv
```

---

## 避免資料洩漏的設計

K-means 是完全無監督（不使用任何標籤），因此在 CV 之前跑一次 K-means 不引入洩漏。

Pseudo-label 的投票**僅使用每 fold 的 train split 標籤**，val split 的標籤在整個訓練過程中不可見。

```
Fold k:
  train_idx  ──→ 投票決定每個群的標籤
  unlabeled  ──→ 依據投票結果獲得 pseudo-label
  train_idx + pseudo-labeled ──→ 訓練
  val_idx    ──→ 評估（純 labeled，不含 pseudo-label）
```

---

## 模型架構

```
輸入文本
  → SBERT (frozen, sentence-transformers/all-MiniLM-L6-v2)
  → Token embeddings [B × T × 384]
  → Mask-aware Mean Pooling → [B × 384]
  → MLP: Linear(384→64) → ReLU → Dropout(0.1) → Linear(64→2)
  → logits → 0 / 1
```

與 exp05 完全相同，差異只在訓練資料：exp05 用 UDA，exp06 用 pseudo-labeling。

---

## 超參數

```yaml
clustering:
  n_clusters: 100
  purity_threshold: 0.8       # 群純度門檻
  min_labeled_per_cluster: 3  # 最少有標籤樣本數才信任投票

model:
  hidden_dim: 384
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
# Step 0：分析 embedding 空間（建議先執行）
python exp/exp06_cluster_pseudo/analyze.py

# Step 1：訓練
python exp/exp06_cluster_pseudo/train.py

# Step 2：預測
python exp/exp06_cluster_pseudo/predict.py
python exp/exp06_cluster_pseudo/predict.py --run 01
```

---

## 產出結構

```
results/
  experiment_log.csv
  figures/
    exp06/
      tsne_by_label.png          ← analyze.py 產出
      purity_hist.png            ← analyze.py 產出
      dendrogram_centroids.png   ← analyze.py 產出
      pseudo_label_stats.txt     ← analyze.py 產出
  exp06/
    experiment_log.json
    01/
      config.yaml
      cv_summary.json
      fold0/
        best.pt  last.pt  train_log.csv
      fold1/  fold2/  fold3/  fold4/
      predictions.csv
```

---

## 與前實驗的比較

| | exp05 | exp06 |
|---|---|---|
| 架構 | MeanPoolingClassifier | MeanPoolingClassifier |
| SBERT | Frozen | Frozen |
| 利用無標籤資料方式 | UDA (KL 一致性) | K-means pseudo-labeling |
| 訓練集大小 | 1,600（labeled train）| 1,600 + N pseudo-labels |
| 額外假設 | 增強後輸出要一致 | 同群文本情緒相同 |
