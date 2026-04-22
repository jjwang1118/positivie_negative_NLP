# Exp03 — Full Fine-tune（RoBERTa-base 效能上限）

**對應論文**：RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)  
**論文筆記**：[notebooks/nlp/RoBERTa_Note.md](../../notebooks/nlp/RoBERTa_Note.md)  
**目標**：對 RoBERTa-base 進行 end-to-end full fine-tune，作為本系列實驗的效能上限 baseline

---

## 資料集設定

| 項目 | 說明 |
|------|------|
| 語言 | 英文 |
| 來源檔案 | `data/raw/train_2022.csv`（格式：`row_id,TEXT,LABEL`） |
| 切割方式 | `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` |
| 每折訓練集 | train set 的約 2/3 |
| 每折驗證集 | train set 的約 1/3 |
| 正負比例 | 每折保持一致（stratified） |

**資料集使用策略**：

| 資料集 | 用途 |
|--------|------|
| Train set | 3-fold 交叉驗證訓練 |
| Val set（每折） | 訓練期間驗證 |
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

## 模型架構：SequenceClassifier

### Pipeline

```
原始文本
  → roberta-base（全部 trainable，約 125M 參數）
  → last_hidden_state[:, 0, :]  → [CLS] token embedding [768]
  → Dropout(0.1)
  → Linear(768 → 2)
  → logits → 0 / 1
```

### 各元件說明

| 元件 | 設定 | 備註 |
|------|------|------|
| RoBERTa backbone | `roberta-base` | 全部參數參與 backprop |
| [CLS] pooling | `last_hidden_state[:, 0, :]` | 取第一個 token 作為句子向量 |
| Dropout | `Dropout(0.1)` | 正規化 |
| Linear head | `Linear(768 → 2)` | 分類頭 |
| 輸出 | softmax → argmax | Label 0（負面）或 1（正面） |

### 訓練目標

所有參數（Encoder + Head，共約 125M）端對端訓練。

### 訓練穩定技巧

| 技巧 | 設定 | 說明 |
|------|------|------|
| 小學習率 | `2e-5` | Full fine-tune 必須小 lr，避免 catastrophic forgetting |
| Linear warmup | `warmup_ratio=0.06` | 前 6% steps 線性升溫，穩定初期訓練 |
| Linear decay | 自動（scheduler） | Warmup 後線性降至 0 |
| Gradient clipping | `max_norm=1.0` | 防止 gradient explosion |

---

## 參數設定（config.yaml）

所有超參數由 [`config.yaml`](config.yaml) 管理，修改後執行 `train.py` 即產生新的 run：

| 區塊 | 參數 | 預設值 | 說明 |
|------|------|--------|------|
| `model` | `name` | `roberta-base` | MODEL_REGISTRY 的 key |
| `model` | `hidden_dim` | `768` | RoBERTa-base 輸出維度 |
| `model` | `dropout` | `0.1` | Dropout 機率 |
| `model` | `num_labels` | `2` | 分類數 |
| `training` | `learning_rate` | `2e-5` | AdamW 學習率 |
| `training` | `batch_size` | `16` | 批次大小 |
| `training` | `epochs` | `5` | 訓練 epoch 數 |
| `training` | `max_length` | `128` | Tokenizer 最大長度 |
| `training` | `seed` | `42` | 全域隨機種子 |
| `training` | `n_folds` | `3` | Cross-validation fold 數 |
| `training` | `warmup_ratio` | `0.06` | Warmup 佔總 steps 的比例 |

---

## 實驗腳本

執行順序：

```bash
# Step 1：訓練（3-fold CV），自動建立新 run 目錄
python exp/exp03_finetune/train.py

# Step 2：預測
python exp/exp03_finetune/predict.py            # 預設最新 run
python exp/exp03_finetune/predict.py --run 01   # 指定 run
```

**注意**：`--run` 後面接兩位數字的 run 編號（如 `01`、`02`），不可寫成 `--01`。

| 腳本 | 說明 |
|------|------|
| [`config.yaml`](config.yaml) | 實驗參數設定，修改此檔後執行 train.py 即產生新 run |
| [`train.py`](train.py) | 3-fold CV 訓練，自動建立 `results/exp03/{run_id}/`，並更新 `experiment_log.json` |
| [`predict.py`](predict.py) | 3-fold ensemble 預測，`predictions.csv` 存於對應 run 目錄，`--run` 可指定 run |

**前置條件**：
- `data/raw/train_2022.csv`（格式：`row_id,TEXT,LABEL`）
- 安裝套件：`pip install transformers pyyaml`

---

## 實驗產出結構

每次執行 `train.py` 自動新增一個編號資料夾：

```
results/
  experiment_log.csv                ← 所有實驗所有 run 的摘要表格（持續累積，每 run 一列）
  exp03/
    experiment_log.json             ← exp03 所有 run 的摘要，JSON 格式（含完整 config）
    01/                             ← 第 1 次執行
      config.yaml                   ← 該次使用的參數快照
      cv_summary.json               ← 3 折結果與平均 acc/loss
      fold0/
        best.pt                     ← 最佳 val_acc checkpoint
        last.pt                     ← 最終 epoch checkpoint
        train_log.csv               ← 每 epoch 的 train/val loss & acc
      fold1/  fold2/
      predictions.csv               ← predict 後新增
    02/                             ← 第 2 次執行（調整 config.yaml 後）
      ...
```

### experiment_log.csv 格式（每 run 一列）

| 欄位 | 說明 |
|------|------|
| `experiment` | 實驗名稱（exp03） |
| `run_id` | Run 編號（01, 02, ...） |
| `timestamp` | 執行時間 |
| `mean_val_acc` | 3 折平均最佳 val_acc |
| `std_val_acc` | 標準差 |
| `fold0_acc` / `fold0_loss` | Fold 0 最佳 acc / loss |
| `fold1_acc` / `fold1_loss` | Fold 1 最佳 acc / loss |
| `fold2_acc` / `fold2_loss` | Fold 2 最佳 acc / loss |
| `learning_rate` | 訓練超參數 |
| `batch_size` | 訓練超參數 |
| `epochs` | 訓練 epoch 數 |
| `max_length` | Tokenizer 最大長度 |
| `seed` | 隨機種子 |
| `n_folds` | Fold 數 |
| `warmup_ratio` | Warmup 比例 |
| `model_name` | 模型名稱 |
| `hidden_dim` | RoBERTa 輸出維度 |
| `dropout` | Dropout |

### experiment_log.json 格式（含完整 config）

```json
[
  {
    "experiment": "exp03",
    "run_id": "01",
    "timestamp": "2026-04-22T10:00:00",
    "config": { "model": {...}, "training": {...} },
    "fold_results": [
      { "fold": 0, "best_val_acc": 0.92, "best_val_loss": 0.28 },
      { "fold": 1, "best_val_acc": 0.93, "best_val_loss": 0.26 },
      { "fold": 2, "best_val_acc": 0.91, "best_val_loss": 0.30 }
    ],
    "mean_val_acc": 0.92,
    "std_val_acc": 0.01
  }
]
```

---

## 與 exp01/02 的比較

| | exp01 | exp02 | exp03 |
|---|---|---|---|
| 模型 | SBERT + Attention Pooling | SBERT + Attention Pooling | RoBERTa-base + [CLS] |
| 輸入 | 原始文本 | SHAP key string（重訓）| 原始文本 |
| Encoder | Frozen | Frozen | 全部訓練 |
| 可訓練參數 | ~25K | ~25K | ~125M |
| 學習率 | 2e-4 | 2e-4 | 2e-5 |
| Scheduler | 無 | 無 | Linear warmup |
| Gradient clipping | 無 | 無 | max_norm=1.0 |
| 定位 | 輕量 + 可解釋 | SHAP 篩選驗證 | 效能上限 |
