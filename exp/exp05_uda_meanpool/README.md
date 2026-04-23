# Exp05 — UDA + Mean Pooling (Attention Pooling Ablation)

**方法**：半監督一致性訓練（UDA）+ Mask-aware Mean Pooling  
**繼承自**：exp04（UDA: TSA + KL 一致性損失）  
**目標**：驗證移除 Attention Layer 後，改用 SBERT 原設計的 mean pooling，是否在 UDA 框架下表現更好或相當

---

## 與 exp04 的差異

| 項目 | exp04 | exp05 |
|------|-------|-------|
| 池化方式 | Attention Layer: Linear(384→1) + masked softmax + Weighted Sum | Mask-aware Mean Pooling |
| 可訓練參數（池化） | 385（Linear 384→1 的 weights + bias） | 0 |
| 其餘架構 | 完全相同 | 完全相同 |
| UDA 機制 | TSA + KL 一致性 | TSA + KL 一致性 |

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

**Mean Pooling 計算方式**：

```python
mask = attention_mask.unsqueeze(-1).float()         # (B, T, 1)
sentence_vec = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)  # (B, 384)
```

Padding token 不參與平均，與 SBERT 的原始訓練方式一致。

---

## 核心概念（繼承自 exp04）

### Training Signal Annealing (TSA)

```
eta_t = 1/K + (t / T) * (1 - 1/K)
```

- K = 2（二元分類），T = 總 training steps
- eta_t 從 0.5 線性增長到 1.0
- true_class_prob < eta_t → 納入 supervised_loss（模型還在學）
- true_class_prob ≥ eta_t → 排除（模型已確定，防止過擬合）

### 一致性損失

```
p = softmax(model(original_text))    # stop_grad
q = softmax(model(augmented_text))   # student
L_con = KL(p || q)，只對 max(p) ≥ 0.8 的樣本計算
```

### 增強策略：TF-IDF 詞替換

- 低 TF-IDF 詞（通用詞）為替換候選，高 TF-IDF 詞（情緒關鍵詞）保留
- `augment_prob=0.1`：每個候選詞有 10% 機率被替換

---

## 超參數（config.yaml）

```yaml
model:
  name: sbert-minilm
  hidden_dim: 384
  mlp_hidden: 64
  dropout: 0.1
  num_labels: 2

training:
  learning_rate: 2.0e-4
  batch_size: 32
  epochs: 30
  max_length: 128
  seed: 42
  n_folds: 5

uda:
  lambda_u: 1.0
  confidence_threshold: 0.8
  tsa_schedule: linear
  augment_prob: 0.1
  unlabeled_batch_size: 32
```

---

## 執行方式

```bash
# Step 1：訓練
python exp/exp05_uda_meanpool/train.py

# Step 2：預測
python exp/exp05_uda_meanpool/predict.py            # 最新 run
python exp/exp05_uda_meanpool/predict.py --run 01   # 指定 run
```

---

## 產出結構

```
results/
  experiment_log.csv
  exp05/
    experiment_log.json
    01/
      config.yaml
      cv_summary.json
      fold0/
        best.pt
        last.pt
        train_log.csv
      fold1/  fold2/  fold3/  fold4/
      predictions.csv
```

---

## 與前實驗的比較

| | exp01 | exp02 | exp03 | exp04 | exp05 |
|---|---|---|---|---|---|
| 池化 | Attention | Attention | Attention | Attention | **Mean** |
| UDA | 無 | 無 | 無 | 有 | **有** |
| 無標籤資料 | 不使用 | 不使用 | 不使用 | 使用 | **使用** |
| val_acc（目標） | 0.758 | 0.767 | 0.763 | TBD | TBD |

exp05 vs exp04 的差異純粹在池化方式，可直接比較 Attention Pooling 與 Mean Pooling 在 UDA 框架下的效果。
