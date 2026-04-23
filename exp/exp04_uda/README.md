# Exp04 — UDA (Unsupervised Data Augmentation)

**方法**：半監督一致性訓練 — 監督損失 (TSA) + 無監督一致性損失 (KL)  
**論文**：Xie et al., "Unsupervised Data Augmentation for Consistency Training", NeurIPS 2020, arXiv:1904.12848  
**目標**：利用 11,000 筆無標籤 test 文本輔助訓練，縮小 val_acc 與真實 test 準確率的落差

---

## 核心概念

UDA 的訓練目標同時包含兩個損失項：

```
total_loss = supervised_loss + lambda_u * consistency_loss
```

| 損失項 | 資料來源 | 計算方式 |
|--------|----------|----------|
| supervised_loss | 有標籤資料（2,000 筆） | CrossEntropy + TSA 遮罩 |
| consistency_loss | 無標籤資料（11,000 筆） | KL(p_orig \|\| p_aug)，信心閾值過濾 |

### Training Signal Annealing (TSA)

防止模型在小標籤集上過擬合。只有當模型對某樣本的 true class 信心低於動態閾值 η_t 時，才將該樣本計入監督損失。

**Linear schedule**：

```
eta_t = 1/K + (t / T) * (1 - 1/K)
```

- K = 2（二元分類），T = 總 training steps
- eta_t 從 0.5 線性增長到 1.0
- 初期：只有模型最不確定的樣本計入損失（防止早期過擬合）
- 後期：幾乎所有樣本都計入損失

### 一致性損失

```
p = softmax(model(original_text))      # stop gradient (teacher)
q = softmax(model(augmented_text))     # trainable (student)
L_con = KL(p || q)
```

- **stop gradient**：p 使用 `p.detach()`，梯度不回傳到 teacher 分支
- **信心閾值過濾**：只對 max(p) ≥ 0.8 的樣本計算一致性損失，排除模型本身不確定的樣本

### 增強策略：TF-IDF 詞替換

不依賴外部 API，純 sklearn 實作：

1. 對訓練集所有文本計算每個詞的平均 TF-IDF 分數
2. 低 TF-IDF 詞（通用詞、功能詞）為替換候選
3. 以 `augment_prob` 機率將候選詞替換為詞彙表中的其他低 TF-IDF 詞
4. 高 TF-IDF 詞（情緒關鍵詞）保持不變

---

## 模型架構

與 exp01–03 完全相同，確保可比性：

```
輸入文本
  → SBERT (frozen, sentence-transformers/all-MiniLM-L6-v2)
  → Token embeddings [n_tokens × 384]
  → Attention Layer: Linear(384→1) + masked softmax
  → Weighted Sum → [384]
  → MLP: Linear(384→64) → ReLU → Dropout(0.1) → Linear(64→2)
  → logits → 0 / 1
```

- SBERT encoder 完全 frozen，只訓練 Attention Layer 和 MLP head
- 使用 `src/model.py` 的 `AttentionPoolingClassifier`（不修改 src/）

**exp04 的差異不在模型架構，而在輸入端**：同一段無標籤文字進來兩次，一次原版、一次 TF-IDF 換詞後的版本，分別跑過同一個模型，得到兩個輸出分布 p 和 q，再計算 KL 損失：

```
原版文本  ──────────────────────────→ [架構圖] → p (stop_grad)
                                                    ↓ KL → con_loss
增強版文本（TF-IDF 換詞）→ [架構圖] → q
```

模型權重只有一份，架構完全不變。

TSA 則作用在有標籤的 train set 上，決定哪些樣本計入 supervised_loss，與無標籤的一致性分支互相獨立：

```
train set（有標籤）→ [架構圖] → 預測信心 vs η_t → TSA 遮罩 → sup_loss
                                                                          ↓ 相加 → total_loss
test set（無標籤）─ 原版 / 增強版 → [架構圖] → p, q → KL → con_loss ──┘
```

---

## 資料集設定

| 項目 | 說明 |
|------|------|
| 有標籤資料 | `data/raw/train_2022.csv`（2,000 筆） |
| 無標籤資料 | `tests/test.csv`（約 11,000 筆，無 LABEL 欄位） |
| 切割方式 | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| 無標籤循環 | `itertools.cycle`，一個 epoch 的迭代次數由 labeled_loader 決定 |

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

## 訓練流程

每個 epoch 的迭代邏輯如下：

```
for each labeled_batch in labeled_loader:
    unlabeled_batch = next(cycle(unlabeled_loader))

    # Supervised branch
    logits_sup = model(labeled_text)
    eta_t = TSA_threshold(step, total_steps)
    tsa_mask = (predicted_confidence_on_true_class < eta_t)
    sup_loss = mean(CrossEntropy[tsa_mask])

    # Consistency branch (stop-gradient on teacher)
    with no_grad():
        p = softmax(model(original_unlabeled))   # teacher
    q = softmax(model(augmented_unlabeled))       # student
    conf_mask = (max(p) >= confidence_threshold)
    con_loss = mean(KL(p || q)[conf_mask])

    total_loss = sup_loss + lambda_u * con_loss
    total_loss.backward()
    optimizer.step()
```

---

## 日誌格式

每 fold 的 `train_log.csv` 記錄以下欄位：

| 欄位 | 說明 |
|------|------|
| epoch | 當前訓練 epoch |
| train_sup_loss | 有標籤監督損失（含 TSA 遮罩） |
| train_con_loss | 無標籤一致性損失（KL 散度） |
| train_total_loss | 總損失 |
| val_loss | 驗證集 CrossEntropy（無 TSA） |
| val_acc | 驗證集準確率 |
| val_f1 | 驗證集 macro F1 |

---

## 執行方式

```bash
# Step 1：訓練 5 折 UDA（含 TSA + 一致性損失）
python exp/exp04_uda/train.py

# Step 2：5-fold ensemble 預測
python exp/exp04_uda/predict.py            # 使用最新 run
python exp/exp04_uda/predict.py --run 01   # 指定 run
```

---

## 產出結構

```
results/
  experiment_log.csv                  (全域實驗 CSV，追加 exp04 列)
  exp04/
    experiment_log.json               (本實驗所有 run 的 JSON 記錄)
    01/
      config.yaml                     (本次 run 的參數快照)
      cv_summary.json                 (5-fold CV 結果摘要)
      fold0/
        best.pt                       (最佳 checkpoint by val_acc)
        last.pt                       (最後 epoch checkpoint)
        train_log.csv                 (per-epoch 訓練指標)
      fold1/  fold2/  fold3/  fold4/
      predictions.csv                 (ensemble 預測結果)
```

---

## 三個機制的直覺說明

三個機制分屬兩條平行分支，不是依序套用在同一批資料上：

### Supervised Branch（有標籤 2k 筆）

TSA 作用在這裡，決定哪些有標籤樣本貢獻 supervised_loss：

```
labeled_data → model → 預測信心 vs η_t → TSA 遮罩 → supervised_loss
```

**TSA 篩選邏輯（注意方向）**：

```
true_class_prob < η_t  →  納入損失（模型還不確定，繼續學）
true_class_prob ≥ η_t  →  排除（模型已確定，停止強化）
```

機率較小（模型不確定）→ 納入；機率較大（模型已確定）→ 排除。

直覺：模型已「學會」的樣本繼續算損失只會加深過擬合，只挑「還沒學好」的樣本繼續練，類似老師只出你還不熟的題目。

η_t 本身從 0.5 線性增長到 1.0：訓練越後期，「學好」的標準越高，被排除的樣本越少，最終所有樣本都參與訓練。

### Consistency Branch（無標籤 11k 筆）

TF-IDF 增強和信心閾值都作用在這裡：

```
unlabeled_text ──────────────────→ model(stop_grad) → p → 信心閾值過濾
                                                             ↓ KL → consistency_loss
unlabeled_text → TF-IDF 增強 → model              → q →
```

### 具體流程（以一筆無標籤文本為例）

**Step 1 — TF-IDF 增強**：產生兩份輸入

```
原版：   "the movie was great"
增強版：  "a film was great"     ← "the"→"a", "movie"→"film"（低TF-IDF詞被換掉）
```

**Step 2 — 各跑一次 model**

```
model("the movie was great") → p = [0.05, 0.95]   ← teacher（不更新梯度）
model("a film was great")    → q = [0.10, 0.90]   ← student（要更新梯度）
```

**Step 3 — 信心閾值篩選**

- max(p) = 0.95 ≥ 0.8 → 保留，p 是可信的訓練訊號
- 若 p = [0.48, 0.52]，max(p) = 0.52 < 0.8 → 跳過，teacher 自己不確定，不值得拿來對齊

**Step 4 — KL 損失**

```
KL(p ∥ q)：衡量 q 和 p 差多遠
梯度只往 student 回傳，讓模型學到「輸入輕微改動時預測要穩定」
```

### 三個動作的分工

| 動作 | 機制 | 作用對象 | 目的 |
|------|------|----------|------|
| 挑選 train 樣本 | TSA | 有標籤資料 | 過濾掉模型已確定的樣本，防止過擬合 |
| 增強文本 | TF-IDF | 無標籤資料 | 製造「略有不同」的 student 輸入 |
| 挑選 test 樣本 | 信心閾值 | 無標籤資料 | 確保 teacher 的 p 是可信的訓練訊號 |
| 計算損失 | KL(p ∥ q) | 無標籤資料 | 逼 student 輸出對齊 teacher 輸出 |

---

## 與前實驗的比較

| | exp01 | exp02 | exp03 | exp04 |
|---|---|---|---|---|
| 訓練資料 | 有標籤 (2k) | 有標籤 (2k) | 有標籤 SHAP key string | 有標籤 + 無標籤 (11k) |
| 方法 | Attention 分類 | SHAP retrain | Warm-start + SHAP consistent | **UDA 半監督** |
| 特殊設計 | Attention pooling | SHAP key string | Warm-start, Cosine LR | **TSA + KL 一致性** |
| 無標籤資料 | 不使用 | 不使用 | 不使用 | **使用（test.csv）** |
| val_acc（目標） | 0.758 | 0.767 | 0.763 | TBD |
