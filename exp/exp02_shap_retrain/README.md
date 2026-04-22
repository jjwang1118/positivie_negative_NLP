# Exp02 — SHAP Key String Retrain（獨立運行）

**對應概念**：SHAP（SHapley Additive exPlanations）詞重要性篩選後重訓  
**論文筆記**：[notebooks/idea/SHAP_Note.md](../../notebooks/idea/SHAP_Note.md)  
**目標**：在單一腳本內完成「初始訓練 → SHAP 提取 → 過濾文字重訓」三個 phase，驗證 SHAP key string 是否保留足以分類的語意資訊，且無需依賴 exp01

---

## 資料集設定

| 項目 | 說明 |
|------|------|
| 語言 | 英文 |
| 來源檔案 | `data/raw/train_2022.csv`（格式：`row_id,TEXT,LABEL`） |
| 切割方式 | `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` |
| 每折訓練集 | train set 的約 2/3 |
| 每折驗證集 | train set 的約 1/3（Phase 1 驗證 + Phase 2 SHAP 擷取） |
| 正負比例 | 每折保持一致（stratified） |

**資料集使用策略**：

| 資料集 | 用途 |
|--------|------|
| Train set | Phase 1 & Phase 3 的 3-fold 交叉驗證訓練 |
| Val set（每折） | Phase 1 驗證 + Phase 2 SHAP Key String 擷取 |
| Test set | 完全保留，不介入當前實驗，留給最終評估 |

---

## 可重現性設定

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

Phase 1 與 Phase 3 使用相同架構。

### Pipeline

```
輸入文本（Phase 1：原文；Phase 3：SHAP key string）
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

## 三 Phase 流程

### Phase 1 — 初始訓練（原始文字）

用完整原始文本訓練 3-fold CV，目的是取得可用於 SHAP 分析的高品質 checkpoint。  
Checkpoint 存於 `run_dir/phase1/fold{k}/best.pt`。

### Phase 2 — SHAP Key String 擷取

**方法**：`shap.PartitionExplainer`（word-level masking）

#### 為何使用 SHAP 而非 Greedy Token Addition

Greedy Token Addition 逐步加入 token 時，SBERT 對截斷輸入的 encode 與訓練時的完整句子不同，造成 distribution shift。SHAP 透過採樣詞的子集並重新 encode 完整句子，避免此問題，Shapley value 同時具有理論上的公平分配保證（否定詞 "not" 等語意修飾詞可被正確評估）。

#### 流程

| 步驟 | 說明 |
|------|------|
| 遮蔽策略 | Word-level（以空白/標點切詞）；遮蔽的詞從文本移除後重新 tokenize |
| 採樣 | 每個樣本最多 `shap_max_evals` 次 model call，計算各詞 Shapley value |
| 評分對象 | Predicted class 的 SHAP value（或 `shap_label_idx` 指定固定類別） |
| Key string 選取 | SHAP value > 0 的詞（對預測類別有正向貢獻），保留原句順序 |
| 覆蓋率 | 透過 CV 使每個訓練樣本都在其 val fold 被處理過，聚合後達到全覆蓋 |

Key strings 聚合存於 `run_dir/key_strings.json`（格式見下方）。

### Phase 3 — SHAP 過濾後重訓

| 步驟 | 說明 |
|------|------|
| 輸入替換 | 以 key string 取代每筆訓練樣本的原文；key string 為空時 fallback 回原文 |
| Tokenize | `max_length=64`（key string 遠短於原文） |
| 訓練 | 與 Phase 1 相同的架構與超參數，3-fold CV |
| Checkpoint | `run_dir/fold{k}/best.pt`（與 `predict.py` 相容） |

---

## Key String 輸出格式

### key_strings.json

```json
[
  {
    "fold": 0,
    "index": 42,
    "label": 1,
    "key_string": "absolutely amazing"
  }
]
```

---

## 參數設定（config.yaml）

所有超參數由 [`config.yaml`](config.yaml) 管理，修改後執行 `train.py` 即產生新的 run：

### `model` 區塊

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `name` | `sbert-minilm` | MODEL_REGISTRY 的 key |
| `hidden_dim` | `384` | SBERT 輸出維度 |
| `mlp_hidden` | `64` | MLP 中間層維度 |
| `dropout` | `0.1` | Dropout 機率 |
| `num_labels` | `2` | 分類數 |

### `phase1` 區塊（初始訓練）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `learning_rate` | `2e-4` | AdamW 學習率 |
| `batch_size` | `32` | 批次大小 |
| `epochs` | `30` | 訓練 epoch 數 |
| `max_length` | `128` | Tokenizer 最大長度（原始文本） |
| `seed` | `42` | 全域隨機種子 |
| `n_folds` | `3` | Cross-validation fold 數 |

### `extraction` 區塊（SHAP）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `shap_max_evals` | `500` | 每樣本最多 SHAP model calls |
| `shap_label_idx` | `null` | null = predicted class；0 or 1 = 固定類別 |

### `phase2` 區塊（重訓）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `learning_rate` | `2e-4` | AdamW 學習率 |
| `batch_size` | `32` | 批次大小 |
| `epochs` | `30` | 訓練 epoch 數 |
| `max_length` | `64` | Tokenizer 最大長度（key string 較短） |
| `seed` | `42` | 全域隨機種子 |
| `n_folds` | `3` | Cross-validation fold 數 |

---

## 實驗腳本

執行順序：

```bash
# Step 1：三 phase 一次完成（Phase 1 初訓 → Phase 2 SHAP → Phase 3 重訓）
python exp/exp02_shap_retrain/train.py

# Step 2：預測（載入 Phase 3 checkpoint，使用原始文本推論）
python exp/exp02_shap_retrain/predict.py            # 預設最新 run
python exp/exp02_shap_retrain/predict.py --run 01   # 指定 run
```

**注意**：`--run` 後面接兩位數字的 run 編號（如 `01`、`02`）。

| 腳本 | 說明 |
|------|------|
| [`config.yaml`](config.yaml) | 實驗參數設定，修改後執行 `train.py` 即產生新 run |
| [`train.py`](train.py) | 三 phase 獨立流程，自動建立 `results/exp02/{run_id}/`，並更新 `experiment_log.json` |
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
  exp02/
    experiment_log.json             ← exp02 所有 run 的摘要，JSON 格式（含完整 config）
    01/                             ← 第 1 次執行
      config.yaml                   ← 該次使用的參數快照
      cv_summary.json               ← Phase 3 三折結果與平均 acc/loss
      key_strings.json              ← Phase 2 聚合所有訓練樣本的 key strings
      phase1/                       ← Phase 1 初始訓練 checkpoint
        fold0/
          best.pt
          last.pt
        fold1/  fold2/
      fold0/                        ← Phase 3 重訓 checkpoint
        best.pt
        last.pt
        train_log.csv               ← 每 epoch 的 train/val loss & acc
      fold1/  fold2/
      predictions.csv               ← predict 後新增
    02/                             ← 第 2 次執行（調整 config.yaml 後）
      ...
```

### experiment_log.csv 格式（每 run 一列）

| 欄位 | 說明 |
|------|------|
| `experiment` | 實驗名稱（exp02） |
| `run_id` | Run 編號（01, 02, ...） |
| `timestamp` | 執行時間 |
| `mean_val_acc` | Phase 3 三折平均最佳 val_acc |
| `std_val_acc` | 標準差 |
| `fold0_acc` / `fold0_loss` | Fold 0 最佳 acc / loss |
| `fold1_acc` / `fold1_loss` | Fold 1 最佳 acc / loss |
| `fold2_acc` / `fold2_loss` | Fold 2 最佳 acc / loss |
| `learning_rate` | Phase 2 學習率 |
| `batch_size` | Phase 2 批次大小 |
| `epochs` | Phase 2 epoch 數 |
| `max_length` | Phase 2 Tokenizer 最大長度 |
| `seed` | 隨機種子 |
| `n_folds` | Fold 數 |
| `model_name` | 模型名稱 |
| `hidden_dim` | SBERT 輸出維度 |
| `mlp_hidden` | MLP 中間層維度 |
| `dropout` | Dropout |

### experiment_log.json 格式（含完整 config）

```json
[
  {
    "experiment": "exp02",
    "run_id": "01",
    "timestamp": "2026-04-22T10:00:00",
    "config": { "model": {...}, "phase1": {...}, "extraction": {...}, "phase2": {...} },
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

## 與 exp01 的關係

| | exp01 | exp02 |
|---|---|---|
| 流程 | 訓練 → SHAP 擷取（獨立腳本）→ 重訓（exp02） | 三 phase 全合一，不依賴 exp01 |
| SHAP 分析 | `extract_key_strings.py` 獨立執行 | 整合於 `train.py` Phase 2 |
| Phase 1 checkpoint | `results/exp01/{run}/fold{k}/best.pt` | `results/exp02/{run}/phase1/fold{k}/best.pt` |
| 最終 checkpoint（predict 用） | `results/exp01/{run}/fold{k}/best.pt`（原文模型） | `results/exp02/{run}/fold{k}/best.pt`（key string 重訓模型） |
| 預測時輸入 | 原始文本 | 原始文本（Phase 3 模型仍可泛化） |
