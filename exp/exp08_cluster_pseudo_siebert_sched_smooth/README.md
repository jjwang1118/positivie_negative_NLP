# Exp08 — Cluster Pseudo-labeling + Cosine Scheduler + Label Smoothing

**基礎**：exp07（K-means cluster pseudo-labeling + siebert encoder）  
**新增**：Cosine LR Scheduler with linear warmup + Label Smoothing  
**目標**：改善 exp07 的 LR 平台震盪問題，並透過 label smoothing 降低 pseudo-label 硬標籤帶來的過度自信

---

## 與 exp07 的差異

| | exp07 | exp08 |
|---|---|---|
| LR 排程 | 固定 2e-4（全程不變）| Cosine decay + linear warmup（10% steps） |
| 損失函數 | CrossEntropyLoss | CrossEntropyLoss(label_smoothing=0.1) |
| train_log | epoch/loss/acc/f1 | 新增 `lr` 欄位 |

其餘完全相同：encoder（siebert, frozen）、K-means（K=100）、pseudo-labeling 流程、MLP head（64 hidden）。

---

## Cosine Scheduler 說明

```
LR
↑
2e-4 |       ★ 峰值
     |      / \
     |     /   \
     |    /     \  \
  0  |___/       \___→ steps
     warmup↑     cosine decay
```

- **Warmup**：前 10% steps LR 從 0 線性上升到 2e-4，避免初期更新過猛
- **Cosine decay**：之後從峰值平滑下降到接近 0，讓模型後期細調而非震盪

---

## Label Smoothing 說明

```
Hard label (exp07): [0.0, 1.0]  → 強迫模型 100% 相信 pseudo-label
Smooth (exp08):     [0.05, 0.95] → 保留 5% 不確定性，降低 pseudo-label 噪聲影響
```

---

## 超參數

```yaml
training:
  learning_rate: 2.0e-4
  batch_size: 32
  epochs: 30
  label_smoothing: 0.1
  warmup_ratio: 0.1

clustering:
  n_clusters: 100
  purity_threshold: 0.8
  min_labeled_per_cluster: 3
```

---

## 執行方式

```bash
# Step 0：分析 embedding 空間（可略過，與 exp07 結果相同）
python exp/exp08_cluster_pseudo_siebert_sched_smooth/analyze.py

# Step 1：訓練
python exp/exp08_cluster_pseudo_siebert_sched_smooth/train.py

# Step 2：預測
python exp/exp08_cluster_pseudo_siebert_sched_smooth/predict.py
python exp/exp08_cluster_pseudo_siebert_sched_smooth/predict.py --run 01
```

---

## 產出結構

```
results/
  experiment_log.csv
  exp08/
    experiment_log.json
    01/
      config.yaml
      cv_summary.json
      fold0/
        best.pt  last.pt  train_log.csv   ← 含 lr 欄位
      fold1/  fold2/  fold3/  fold4/
      predictions.csv
```
