# Exp01 — Input Reduction 捷徑偵測與 Key String 驗證

**對應論文**：Pathologies of Neural Models Make Interpretations Difficult (EMNLP 2018)  
**論文筆記**：[notebooks/idea/InputReduction_Note.md](../../notebooks/idea/InputReduction_Note.md)  
**目標**：訓練 SBERT Attention Pooling 分類器，並以 SHAP 分析各詞對預測的貢獻，擷取語意上具辨識力的 key string

---

## 資料集設定

| 項目 | 說明 |
|------|------|
| 語言 | 英文 |
| 來源檔案 | `data/raw/train_2022.csv`（格式：`row_id,TEXT,LABEL`） |
| 切割方式 | `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` |
| 每折訓練集 | train set 的約 2/3 |
| 每折驗證集 | train set 的約 1/3（用於訓練驗證 + SHAP 分析） |
| 正負比例 | 每折保持一致（stratified） |

**資料集使用策略**：

| 資料集 | 用途 |
|--------|------|
| Train set | 3-fold 交叉驗證訓練 |
| Val set（每折） | 訓練期間驗證 + SHAP Key String 擷取分析 |
| Test set | 完全保留，不介入當前實驗，留給最終評估 |

---

## 可重現性設定

以下設定統一在各腳本的 `set_seed(42)` 中生效：

| 項目 | 設定 | 說明 |
|------|------|------|
| 資料切割 | `StratifiedKFold(random_state=42)` | 每次產生相同的 fold 分割 |
| Python random | `random.seed(42)` | 標準函式庫隨機狀態固定 |
| NumPy | `np.random.seed(42)` | NumPy 隨機狀態固定 |
| PyTorch CPU | `torch.manual_seed(42)` | 模型初始化與 CPU 運算固定 |
| PyTorch GPU | `torch.cuda.manual_seed_all(42)` | 所有 GPU 隨機狀態固定 |
| cuDNN 確定性 | `torch.backends.cudnn.deterministic = True` | 強制確定性 kernel（速度略降 5–15%） |
| cuDNN benchmark | `torch.backends.cudnn.benchmark = False` | 關閉自動 kernel 選擇 |

---

## 模型架構：Attention Pooling Classifier

### Pipeline

```
原始文本
  → SBERT (frozen, sentence-transformers/all-MiniLM-L6-v2)
  → Token embeddings [n_tokens × 384]
  → Attention Layer: Linear(384→1) + masked softmax  ← trainable
  → Weighted Sum → [384]
  → MLP: Linear(384→64) → ReLU → Dropout → Linear(64→2)  ← trainable
  → logits → 0 / 1
```

### 各元件說明

| 元件 | 設定 | 備註 |
|------|------|------|
| SBERT backbone | `all-MiniLM-L6-v2` | frozen，不參與 backprop |
| Token embeddings | `[n × 384]` contextual | 每個 token 已編碼周圍語意 |
| Attention Layer | `Linear(384→1)` + masked softmax | 輸出每個 token 的重要性 weight；padding 遮蔽 |
| Weighted Sum | `Σ weight_i × e_i` | 合成單一 384 維句子向量 |
| MLP | `Linear(384→64) → ReLU → Dropout(0.1) → Linear(64→2)` | 分類頭 |
| 輸出 | softmax → argmax | Label 0（負面）或 1（正面） |

### 訓練目標

訓練 Attention Layer + MLP（約 25K 參數），SBERT encoder（約 22M 參數）保持 frozen。

---

## Key String 擷取：SHAP

**方法**：`shap.PartitionExplainer`（word-level masking）

### 為何使用 SHAP 而非 Greedy Token Addition

Greedy Token Addition 逐步加入 token 時，SBERT 對截斷輸入的 encode 與訓練時的完整句子不同，造成 distribution shift。SHAP 透過採樣所有詞的子集並重新 encode 完整句子，避免此問題，Shapley value 同時具有理論上的公平分配保證（否定詞 "not" 等語意修飾詞可被正確評估）。

### 流程

| 步驟 | 說明 |
|------|------|
| 遮蔽策略 | Word-level（以空白/標點切詞）；遮蔽的詞從文本移除後重新 tokenize |
| 採樣 | 每個樣本最多 `shap_max_evals` 次 model call，計算各詞 Shapley value |
| 評分對象 | Predicted class 的 SHAP value（或 `shap_label_idx` 指定固定類別） |
| Key string 選取 | SHAP value > 0 的詞（對預測類別有正向貢獻），保留原句順序 |

### 輸出格式（key_strings.json）

```json
{
  "fold": 0,
  "index": 42,
  "text": "The movie was absolutely amazing",
  "label": 1,
  "predicted_class": 1,
  "key_tokens": ["amazing", "absolutely"],
  "key_string": "absolutely amazing",
  "token_count": 2,
  "total_tokens": 5,
  "reduction_ratio": 0.40
}
```

### 輸出格式（token_importance.csv）

| 欄位 | 說明 |
|------|------|
| `fold` | 所屬 fold 編號 |
| `index` | 全域樣本索引 |
| `label` | 真實標籤 |
| `token` | 詞 |
| `shap_value` | Shapley value（正 = 對 predicted class 有貢獻） |
| `rank` | 依絕對值排序的重要性名次 |
| `selected` | 1 = 被選入 key string |

---

## 參數設定（config.yaml）

所有超參數由 [`config.yaml`](config.yaml) 管理，修改後執行 `train.py` 即產生新的 run：

| 區塊 | 參數 | 預設值 | 說明 |
|------|------|--------|------|
| `model` | `name` | `sbert-minilm` | MODEL_REGISTRY 的 key |
| `model` | `hidden_dim` | `384` | SBERT 輸出維度 |
| `model` | `mlp_hidden` | `64` | MLP 中間層維度 |
| `model` | `dropout` | `0.1` | Dropout 機率 |
| `model` | `num_labels` | `2` | 分類數 |
| `training` | `learning_rate` | `2e-4` | AdamW 學習率 |
| `training` | `batch_size` | `32` | 批次大小 |
| `training` | `epochs` | `10` | 訓練 epoch 數 |
| `training` | `max_length` | `128` | Tokenizer 最大長度 |
| `training` | `seed` | `42` | 全域隨機種子 |
| `training` | `n_folds` | `3` | Cross-validation fold 數 |
| `extraction` | `shap_max_evals` | `500` | 每樣本最多 SHAP model calls |
| `extraction` | `shap_label_idx` | `null` | null = predicted class；0 or 1 = 固定類別 |

---

## 實驗腳本

執行順序：

```bash
# Step 1：訓練（3-fold CV），自動建立新 run 目錄
python exp/exp01_input_reduction/train.py

# Step 2：SHAP Key String 擷取
python exp/exp01_input_reduction/extract_key_strings.py             # 預設最新 run
python exp/exp01_input_reduction/extract_key_strings.py --run 01   # 指定 run

# Step 3：預測
python exp/exp01_input_reduction/predict.py                         # 預設最新 run
python exp/exp01_input_reduction/predict.py --run 01               # 指定 run
```

**注意**：`--run` 後面接兩位數字的 run 編號（如 `01`、`02`），不可寫成 `--01`。

| 腳本 | 說明 |
|------|------|
| [`config.yaml`](config.yaml) | 實驗參數設定，修改此檔後執行 train.py 即產生新 run |
| [`train.py`](train.py) | 3-fold CV 訓練，自動建立 `results/exp01/{run_id}/`，並更新 `experiment_log.json` |
| [`extract_key_strings.py`](extract_key_strings.py) | SHAP 分析，結果存於對應 run 目錄，`--run` 可指定 run |
| [`predict.py`](predict.py) | 3-fold ensemble 預測，`predictions.csv` 存於對應 run 目錄，`--run` 可指定 run |

**前置條件**：
- `data/raw/train_2022.csv`（格式：`row_id,TEXT,LABEL`）
- 安裝套件：`pip install shap pyyaml`

---

## 實驗產出結構

每次執行 `train.py` 自動新增一個編號資料夾：

```
results/
  experiment_log.csv                ← 所有實驗所有 run 的摘要表格（持續累積，每 run 一列）
exp01/
  experiment_log.json               ← exp01 所有 run 的摘要，JSON 格式（含完整 config）
  01/                               ← 第 1 次執行
    config.yaml                     ← 該次使用的參數快照
    cv_summary.json                 ← 3 折結果與平均 acc/loss
    fold0/
      best.pt                       ← 最佳 val_acc checkpoint
      last.pt                       ← 最終 epoch checkpoint
      train_log.csv                 ← 每 epoch 的 train/val loss & acc
      key_strings.json              ← SHAP 擷取後新增
      token_importance.csv          ← SHAP 擷取後新增
    fold1/  fold2/
    key_strings.json                ← 跨折聚合（SHAP 擷取後新增）
    token_importance.csv            ← 跨折聚合（SHAP 擷取後新增）
    predictions.csv                 ← predict 後新增
  02/                               ← 第 2 次執行（調整 config.yaml 後）
    ...
```

### experiment_log.csv 格式（每 run 一列）

| 欄位 | 說明 |
|------|------|
| `run_id` | Run 編號（01, 02, ...） |
| `timestamp` | 執行時間 |
| `mean_val_acc` | 3 折平均最佳 val_acc |
| `std_val_acc` | 標準差 |
| `fold0_acc` / `fold0_loss` | Fold 0 最佳 acc / loss |
| `fold1_acc` / `fold1_loss` | Fold 1 最佳 acc / loss |
| `fold2_acc` / `fold2_loss` | Fold 2 最佳 acc / loss |
| `learning_rate` | 訓練超參數 |
| `batch_size` | 訓練超參數 |
| `epochs` | 訓練超參數 |
| `max_length` | Tokenizer 最大長度 |
| `seed` | 隨機種子 |
| `n_folds` | Fold 數 |
| `model_name` | 模型名稱 |
| `hidden_dim` | SBERT 輸出維度 |
| `mlp_hidden` | MLP 中間層維度 |
| `dropout` | Dropout |
| `shap_max_evals` | SHAP 每樣本最大 model calls |

### experiment_log.json 格式（含完整 config）

```json
[
  {
    "experiment": "exp01",
    "run_id": "01",
    "timestamp": "2026-04-22T10:00:00",
    "config": { "model": {...}, "training": {...}, "extraction": {...} },
    "fold_results": [
      { "fold": 0, "best_val_acc": 0.85, "best_val_loss": 0.42 },
      { "fold": 1, "best_val_acc": 0.87, "best_val_loss": 0.39 },
      { "fold": 2, "best_val_acc": 0.86, "best_val_loss": 0.41 }
    ],
    "mean_val_acc": 0.86,
    "std_val_acc": 0.01
  }
]
```

---
