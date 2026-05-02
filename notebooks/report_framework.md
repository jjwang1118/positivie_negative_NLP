# NLP 情緒分析報告框架
> 對應 400BadRequest PDF 結構，填入 EXP01（exp09-05）& EXP02（exp11-01）兩方案內容

---

## Abstract


本研究使用Kaggle的「Positive/Negative: Text Polarity Classification 26」資料集，針對僅有 1,600 筆標記樣本、11,000 筆無標記測試樣本的情境，進行二元情緒分（正面／負面）任務。標記資料量嚴重不足使模型容易過擬合，泛化能力受限，因此如何透過偽標籤策略（pseudo-labeling）有效利用大量無標記測試集成為本研究的核心問題。本究以 RoBERTa-large 為基底模型，設計並比較兩種偽標籤策略：EXP01 採三階段漸進解凍訓練流程，Stage 1 凍結整個 encoder 僅訓練分類頭以建立穩定初始模型，Stage 2–3 逐步解凍後三層並以信度閾值（0.90）篩選偽標籤擴充訓練集；EXP02 採兩階段策略，Phase 0 以全量標記資料訓練暖機模型並對測試集全 11,000 筆生成偽標籤（不設信度閾值）Phase 1 再以各 fold 標記資料加上全量偽標籤進行 5-fold 訓練，搭配類別平衡加權與 Early Stopping。實驗結果顯示，EXP02 在各項指標上均優於 EXP01（驗證準確率：08805 vs. 0.8315；F1：0.8810；標準差：0.0153 vs. 0.0163 ; public test: 0.77465 vs 0.77988 ; private test: 0.77815 vs 0.79864），說明在小樣本情境下，以力預訓練模型為基底、無條件引入全量偽標籤的策略，比信度篩選加漸進解凍的多階段策略具有更穩定的泛化性能。

---

## Introduction


- exp09-05（EXP01）流程圖
![alt text](exp09-05.png)

- exp11-01（EXP02）流程圖
![alt text](exp11-01.png)



情緒分析（Sentiment Analysis）是自然語言處理（NLP）領域的核心任務之一，廣泛應用於輿情監控、產品評論分析與社群媒體理解等場景。其目標是自動識別文本所傳遞的情緒向，以二元分類任務而言，即判斷一段文字為正面或負面情緒。儘管近年來以 BERT 為代表的預訓練語言模型（Pre-trained Language Models, PLMs）已大幅提升此類任務的性能然而在實務場景中，人工標記資料的取得往往成本高昂，如何在標記樣本稀少的條件下充分發揮模型潛力，仍是當前研究的重要課題。

本研究使用 Kaggle 競賽資料集「Positive/Negative: Text Polarity Classification 26」，任務目標為對給定文本進行二元情緒分類，標籤 1 代表正面情緒，標籤 0 代表面情緒。資料集共包含 1,600 筆已標記訓練樣本與 11,000 筆無標記測試樣本，資料規模嚴重失衡。初步分析顯示，訓練集中存在類別不平衡問題（正面樣本多於負面樣本），且部文本含有縮寫斷詞等噪聲（例如 `"can t"` 應為 `"cannot"`、`"i m"` 應為 `"i am"`、`"won t"` 應為 `"will not"`），這些被 tokenizer 切斷的縮寫若未修正，將導致型無法正確理解語意，進一步加大了模型學習的難度。在此條件下，若直接對全部 encoder 參數進行微調，極易因可用標記樣本過少而過擬合，導致模型泛化能力嚴重受限。

面對上述挑戰，本研究的核心問題為：如何在小樣本條件下，有效利用大量無標記測試樣本來擴充訓練資料，以提升模型的泛化性能。半監督學習中的偽標籤策略（pseudo-labeling提供了一條可行路徑——先以有限標記資料訓練出初始模型，再對無標記樣本進行預測並將高信度預測結果作為偽標籤，重新加入訓練集，迭代強化模型表現。然而，偽標籤的品質與引時機對最終結果影響甚鉅，過早引入低品質偽標籤可能引發誤差累積，而過嚴的篩選則會使偽標籤數量過少，難以帶來有效增益。

為探討不同偽標籤策略在此任務下的效果，本研究設計並比較兩種以 RoBERTa-large 為基底的訓練方案：EXP01 (exp09-05.png)採三階段漸進解凍流程，Stage 1 凍結 encoder 立穩定初始分類器並以信度閾值（0.90）篩選偽標籤，Stage 2–3 逐步解凍 encoder 後三層進行細微調整；EXP02 (exp11-01.png)則採兩階段策略，Phase 0 以全量標記資料訓暖機模型對測試集生成全量偽標籤（不設篩選閾值），Phase 1 以各 fold 的標記資料加上全量偽標籤進行 5-fold 訓練，並以類別平衡加權與 Early Stopping 強化訓練穩定性。

兩種實驗方案共用相同的五階段整體架構（如圖所示）：EDA、資料預處理、偽標籤策略、模型訓練、評估，並皆以 5-fold 交叉驗證確保結果穩定性。兩者的差異集中在第三、四段：EXP01 以信度閾值篩選偽標籤並搭配漸進解凍進行三階段訓練；EXP02 則以全量偽標籤搭配類別加權與 Early Stopping 進行兩階段訓練。本研究透過控制其他條件不變，比較種偽標籤策略在小樣本情緒分析任務下的性能差異。

---

## Methodology

### 實驗簡述

本研究設計兩種以偽標籤為核心的半監督訓練方案，皆以 RoBERTa-large 為基底模型，並採用 5-fold 交叉驗證進行評估。EXP01 的核心概念為漸進式解凍（Progressive Unfreezing）結合自訓練（Self-training）：透過三個訓練階段，先凍結 encoder 建立穩定分類頭，再逐步開放 encoder 後三層，並以信度閾值篩選偽標籤逐步擴充訓練資料。EXP02 則採取更直接的全量偽標籤策略：以 Phase 0 暖機模型對測試集全量預測產生 11,000 筆偽標籤，再於 Phase 1 的 5-fold 訓練中直接引入，搭配類別平衡加權與 Early Stopping 控制訓練穩定性。兩者的主要差異在於偽標籤的篩選策略與 encoder 的解凍方式。

為確保實驗結果可重現，兩方案均統一固定隨機種子（seed = 42），並於訓練開始前呼叫 `set_seed()` 函式，依序設定 Python `random`、NumPy、PyTorch（CPU 與 CUDA）的隨機種子。EXP01 額外設定 `torch.backends.cudnn.deterministic = True` 與 `torch.backends.cudnn.benchmark = False` 以消除 CUDA 卷積運算的不確定性；EXP02 則透過 HuggingFace `TrainingArguments(seed=42)` 將種子傳入 Trainer 框架。兩方案的 5-fold 切割均以 `StratifiedKFold(random_state=42)` 確保每次執行的 fold 分配一致。

---

### 資料前處理

EXP01 未對文字進行任何清理，直接以原始文本輸入模型。EXP02 則在讀取資料前先進行文字標準化處理，針對資料集中常見的縮寫斷詞問題加以修正。由於部分文本中的縮寫在收集過程中被切斷（如 `"can t"` 應為 `"cannot"`、`"i m"` 應為 `"i am"`），若不加以還原，模型的 tokenizer 將無法正確識別縮寫語意，進而影響特徵擷取的品質。因此，EXP02 透過正規表達式將常見的否定縮寫、人稱縮寫與助動詞縮寫統一還原為完整形式，同時將所有文字轉為小寫以消除大小寫差異，並壓縮連續重複字元（如 `"loooove"` → `"loove"`）以降低表面噪聲對模型的干擾。



---

### 模型選擇

表格1:

| 項目 | siebert（run02） | roberta-large（run05） |
|------|----------------|----------------------|
| 預訓練性質 | 情緒分類特化版 | 通用 MLM 預訓練 |
| 參數量 | ~355M | ~355M |
| mean val acc | **0.8565**（std=0.0094） | 0.8315（std=0.0163） |
| Kaggle public score | 0.74490 | **0.77465** |
| val → public 落差 | ~0.1115（過擬合） | ~0.057（泛化較佳） |
| 與自定義流程相容性 | 低（權重已特化，解凍空間有限） | 高（完整語言表示，可靈活調整） |


 本研究初期針對基底模型的選擇進行比較實驗，候選模型為 `siebert/sentiment-roberta-large-english`（Hartmann et al., 2023）[2] 與 `roberta-large`（Liu et al., 2019）[1]。前者為 RoBERTa-large 在大規模情緒分類資料集上微調後的特化版本，理論上具備較強的情緒語意理解能力；後者則為以 Masked Language Modeling（MLM）為目標訓練的通用預訓練模型，語言表示空間較為完整且可塑。如表格1所示，以 siebert 為基底的早期實驗（run02）雖取得較高的驗證準確率（mean val acc = 0.8565），但 Kaggle public score 僅為 0.74490，兩者落差高達 0.1115，顯示 siebert 在本小樣本資料集上存在嚴重過擬合的問題。相較之下，改用 roberta-large 為基底後（run05），驗證準確率雖略降至 0.8315，但 Kaggle public score 大幅提升至 0.77465，val 與 public 的落差縮小至 0.057，泛化性能顯著改善。此結果說明，siebert 已固化的情緒特化表示在小樣本且需要多階段自訓練的情境下反而造成限制，難以透過漸進解凍與偽標籤策略靈活調整；而 roberta-large 保留的完整語言表示空間則更適合作為本研究自定義訓練流程的基底。因此，本研究最終採用 roberta-large 作為兩個實驗方案的共同基底模型。

---

### 實驗流程

### EXP01 — Self-training + Progressive Encoder Unfreezing

EXP01 的核心設計理念為在資料量嚴重不足的條件下，透過「先穩定、後擴展」的漸進策略降低過擬合風險。整體訓練分為三個階段，並於各階段間利用模型對測試集的預測結果生成偽標籤，逐步擴充訓練資料。所有階段均在 5-fold 交叉驗證框架下獨立執行，最終以各 fold 最佳模型的 softmax 機率平均作為最終預測。

#### 模型架構

EXP01 在 roberta-large 的 encoder 之上接一個自定義的分類頭：Linear(1024→64) → ReLU → Dropout(0.1) → Linear(64→2)。中間層將 1,024 維的 encoder 輸出壓縮至 64 維，以 ReLU 作為激活函數引入非線性建模能力，並以 Dropout(0.1) 抑制過擬合。此輕量瓶頸設計在 Stage 1 凍結 encoder 的情境下尤為關鍵：Stage 1 僅以 1,600 筆資料訓練分類頭，將中間維度壓縮至 64 減少可訓練參數數量，避免分類頭在小樣本下記憶訓練資料的雜訊。

#### 訓練流程（三階段 × 5-fold CV）

**Stage 1 — 凍結 encoder**

Stage 1 凍結 roberta-large 的所有 encoder 層，僅以 1,600 筆標記資料訓練分類頭（LR = 1e-4，epochs = 10，max_length = 128，label smoothing = 0.1，warmup ratio = 0.1）。此階段的目的有二：一是在不破壞預訓練表示的前提下建立穩定的初始分類器；二是以此模型對 11,000 筆測試樣本進行預測，篩選 softmax confidence ≥ 0.90 的樣本作為第一批偽標籤（pseudo_pool_S1）。

**偽標籤引入量（Stage 1 → Stage 2）**：由於 0.90 閾值對 Stage 1 初始模型偏嚴，本次實驗所有 fold 的 pseudo_pool_S1 均為 **0 筆**，Stage 2 實際上以純標記資料開始訓練。

**Stage 2 — 解凍後 3 層**

Stage 2 解凍 roberta-large 最後 3 層 Transformer encoder（共 24 層），以分層學習率（encoder LR = 1e-5、head LR = 1e-4，epochs = 10，max_length = 128，label smoothing = 0.1，warmup ratio = 0.1）進行微調。訓練資料為 1,600 筆標記樣本加上 pseudo_pool_S1（本次為 0）。encoder LR 設為 head LR 的 1/10，使 encoder 高層表示能逐步適應本資料集的情緒分布，同時避免因學習率過大導致底層通用語言特徵遭到破壞（catastrophic forgetting）。訓練完成後，再次對 11,000 筆測試樣本進行預測，以 confidence ≥ 0.90 篩選生成 pseudo_pool_S2，供 Stage 3 使用。

**偽標籤引入量（Stage 2 → Stage 3）**：Stage 2 模型品質明顯提升，各 fold 篩選出的 pseudo_pool_S2 表格2：

表格2:
| Fold | n_pseudo_S2 |
|------|-------------|
| 0 | 3,040 |
| 1 | 3,036 |
| 2 | 2,879 |
| 3 | 3,563 |
| 4 | 3,451 |
| **Mean** | **3,193.8** |

**Stage 3 — 低 LR 精煉**

Stage 3 以 Stage 2 的最佳權重為起點（warm-start），以更低的學習率（encoder LR = 2e-6、head LR = 5e-5，epochs = 15，warmup ratio = 0.05）對整個模型進行最終精煉。訓練資料為 1,600 筆標記樣本加上 pseudo_pool_S2（平均 3,194 筆），不再產生新一輪偽標籤，目標是讓模型在已有訓練資料上收斂至更佳的局部最優解。

#### 偽標籤策略

EXP01 採用信度閾值篩選（confidence ≥ 0.90）的偽標籤策略，僅引入模型最有把握的預測結果，以維持偽標籤品質。然而，此設計在 Stage 1 階段因初始模型表現有限，導致本次實驗中所有 fold 均未能產生任何偽標籤（pseudo_pool_S1 = 0），實際上 Stage 2 等同於僅以原始標記資料進行訓練。偽標籤的有效引入直至 Stage 2 完成後才實現，平均為每個 fold 提供約 3,194 筆額外訓練資料。

#### Ensemble

5-fold 交叉驗證中，每個 fold 各自獨立完成三個訓練階段，並以 Stage 3 的最佳模型對測試集輸出 softmax 機率分佈。最終預測結果為 5 個 fold 的 softmax 機率逐樣本取平均後，再以 argmax 轉換為類別標籤。此集成策略基於各 fold 模型在不同資料切割下的互補性，對最終預測結果取 softmax 機率平均，以降低單一 fold 切割帶來的隨機誤差。

---

### EXP02 — Phase 0 Pseudo-label + 5-fold Ensemble

EXP02 的核心設計理念為以最低的複雜度充分利用 11,000 筆無標記測試資料。整體訓練分為兩個階段：Phase 0 以全量標記資料訓練一個暖機模型並生成全量偽標籤；Phase 1 再以各 fold 的標記資料加上全量偽標籤進行 5-fold 訓練，搭配類別平衡加權與 Early Stopping 控制訓練品質。

#### 模型架構

EXP02 採用 HuggingFace AutoModelForSequenceClassification 載入 roberta-large，使用預設的 RobertaClassificationHead（Dropout → Linear(1024→1024) → Tanh → Dropout → Linear(1024→2)），以 FP16 混合精度訓練以節省顯存。相較於 EXP01 的自定義輕量分類頭，EXP02 維持預設全維度分類頭，搭配 HuggingFace Trainer 框架進行訓練，減少自定義程式碼的複雜度。

#### 文字預處理

EXP02 在讀取資料時套用文字標準化處理：將所有文字轉為小寫、透過正規表達式修正縮寫斷詞（如 `"can t"` → `"cannot"`、`"i m"` → `"i am"` 等），並壓縮連續重複字元。詳細規則見 Methodology — 資料前處理節。

#### 訓練流程（兩階段 × 5-fold CV）

**Phase 0 — 偽標籤生成（暖機模型）**

Phase 0 以全量 1,600 筆標記資料（不做 fold 切割）訓練 roberta-large（LR = 1e-5，batch size = 8，gradient accumulation = 2，effective batch = 16，epochs = 3，max_length = 192，warmup ratio = 0.1，weight decay = 0.05）。訓練完成後對 11,000 筆測試樣本全量預測，不設信度閾值，直接將所有預測結果作為偽標籤。Phase 0 結束後立即釋放 GPU 記憶體，再進入 Phase 1。

**Phase 1 — 5-fold 訓練**

Phase 1 對每個 fold 各自以 labeled_fold_train（約 1,280 筆）加上 Phase 0 生成的全量 11,000 筆偽標籤進行訓練（LR = 1e-5，batch size = 8，gradient accumulation = 2，effective batch = 16，epochs = 3，max_length = 192，warmup ratio = 0.1，weight decay = 0.05）。訓練過程中以 Balanced class weights（WeightedTrainer）處理類別不平衡，並以 EarlyStoppingCallback（patience = 2，監控 val F1）防止過擬合，每個 epoch 保存最佳 checkpoint。

#### 偽標籤策略

EXP02 捨棄信度閾值篩選，直接將 Phase 0 暖機模型對測試集的全量預測（11,000 筆）作為偽標籤引入 Phase 1 訓練。此設計的核心考量在於：以信度高的樣本為主雖能保證偽標籤品質，但在資料量嚴重不足的情境下，過嚴的篩選反而使擴充效益有限。全量引入雖包含部分低信度的偽標籤，但 roberta-large 整體預測品質已足以作為弱監督訊號，搭配 class weights 與 Early Stopping 可進一步緩解低品質偽標籤的負面影響。

#### Ensemble

5-fold 交叉驗證中，每個 fold 各自完成 Phase 1 訓練，並以最佳 checkpoint 對測試集輸出 softmax 機率分佈。最終預測結果為 5 個 fold 的 softmax 機率逐樣本取平均後，再以 argmax 轉換為類別標籤。此集成策略基於各 fold 模型在不同資料切割下的互補性，對最終預測取平均，以降低單一 fold 切割帶來的隨機誤差。

---
## Results

### 兩方案共同設定


表格3:
| 項目 | EXP01 | EXP02 |
|------|-------|-------|
| Base model | roberta-large | roberta-large |
| Seed | 42 | 42 |
| CV | 5-fold Stratified | 5-fold Stratified |
| 評估主指標 | Accuracy | Accuracy + F1 |
| 類別平衡 | — | Balanced class weights |
| Early stopping | 無 | patience=2（val F1） |
| Label smoothing | 0.1 | 無 |

如表格3所示，兩方案均以 roberta-large 為基底、5-fold Stratified CV 為驗證框架，並固定隨機種子（seed=42）確保結果可重現。主要差異在於類別不平衡處理、Early Stopping 設置及 Label Smoothing 的使用。

### EXP01 各 Fold 結果
表格4:
| Fold | Stage1 Acc | Stage2 Acc | Best Val Acc | n_pseudo_s2 |
|------|-----------|-----------|-------------|------------|
| 0 | 0.7550 | 0.8325 | 0.8325 | 3,040 |
| 1 | 0.7650 | 0.8225 | 0.8425 | 3,036 |
| 2 | 0.7475 | 0.7875 | 0.8000 | 2,879 |
| 3 | 0.7400 | 0.8225 | 0.8375 | 3,563 |
| 4 | 0.7300 | 0.8300 | 0.8450 | 3,451 |
| **Mean** | 0.7475 | 0.8210 | **0.8315** | 3,193.8 |
| **Std** | — | — | 0.0163 | — |

如表格4所示，EXP01 各 fold 的 Stage 1 準確率介於 0.730 至 0.765 之間，顯示凍結 encoder 僅訓練分類頭的初始性能有限。Stage 2 解凍後 3 層後，各 fold 準確率顯著提升（0.788 至 0.833），最終 Best Val Acc 平均達 0.8315（std = 0.0163）。值得注意的是，所有 fold 的 pseudo_pool_S1 均為 0，偽標籤僅由 Stage 2 產生，平均為每個 fold 提供約 3,194 筆額外訓練資料。Fold 2 表現明顯偏低（0.8000），使整體標準差相對較大（0.0163）。Kaggle public score 為 0.77465，private score 為 0.77815。

### EXP02 各 Fold 結果

表格5:
| Fold | Val Acc | Val Loss | Val F1 |
|------|---------|----------|--------|
| 0 | 0.8725 | 0.6221 | 0.8765 |
| 1 | 0.9025 | 0.4379 | 0.9051 |
| 2 | 0.8650 | 0.6580 | 0.8670 |
| 3 | 0.8675 | 0.4105 | 0.8630 |
| 4 | 0.8950 | 0.4358 | 0.8934 |
| **Mean** | **0.8805** | — | **0.8810** |
| **Std** | 0.0153 | — | — |

如表格5所示，EXP02 各 fold 的驗證準確率介於 0.865 至 0.903 之間，mean Val Acc 達 0.8805（std = 0.0153），mean Val F1 為 0.8810。Fold 1 表現最佳（Val Acc = 0.9025），Fold 2 相對較低（0.8650）但仍優於 EXP01 所有 fold。整體標準差（0.0153）略低於 EXP01（0.0163），顯示 EXP02 各 fold 間的表現較為一致。Kaggle public score 為 0.77988，private score 為 0.79864。

### 綜合比較

表格6:
| 項目 | EXP01 | EXP02 |
|------|-------|-------|
| mean Val Acc | 0.8315 | **0.8805** |
| Val Acc Std | 0.0163 | **0.0153** |
| mean Val F1 | — | **0.8810** |
| Kaggle public score | 0.77465 | **0.77988** |
| Kaggle private score | 0.77815 | **0.79864** |
| Pseudo-label 數量 | ~3,194（Stage 2） | 11,000（全量） |
| Confidence threshold | 0.90 | 無 |
| Encoder 解凍 | 後 3 層（Stage 2+） | 全部（Phase 0 & 1） |
| 訓練階段數 | 3 × 5 fold | 2 × 5 fold |

如表格6所示，EXP02 在驗證集與 Kaggle 測試集的各項指標上均優於 EXP01。驗證準確率提升約 4.9 個百分點（0.8805 vs. 0.8315），Kaggle private score 提升約 2 個百分點（0.79864 vs. 0.77815）。此結果顯示，全量偽標籤搭配類別加權與 Early Stopping 的兩階段策略，在本小樣本情境下優於信度篩選加漸進解凍的三階段策略。EXP01 因 Stage 1 閾值過嚴（pseudo_pool_S1 = 0），實際上未能在 Stage 2 之前有效引入偽標籤，削弱了多階段設計的預期效益。

---

## Future Work

- [ ] 調整 EXP01 Stage 1 閾值（e.g., 0.80）或增加訓練輪次，讓 pseudo_pool_S1 > 0

> 改善EXP01 stage1，在同樣的方式下，如果有辦法產生更多的假標籤作為 stage2 的訓練資料，是否可以讓 stage2 的訓練更有效率，甚至讓 stage3 的訓練更有成效。

- [ ] 比較 EXP02 中「有/無 confidence 篩選」的偽標籤效果

> 探討在 EXP02 中，如果對偽標籤進行信度篩選，是否能進一步提升模型性能，或者是否會因資料量減少而影響效果。

- [ ] 嘗試其他 BERT 系模型（DeBERTa-v3 等）

> 經過多輪嘗試，發現 RoBERTa-large 在本任務下的表現已相當優異，但仍可嘗試其他架構（如 DeBERTa-v3）以驗證是否能帶來進一步提升。

- [ ] 加入 label smoothing 到 EXP02

> 根據前面幾次在Kaggle上嘗試的結果，最佳結果未必是最優解;最不好的結果也未必是最差解;因此，對於如何防止overfit和在小樣本情境下提升模型泛化能力和準確度，加入 label smoothing 可能是一個值得探索的方向。

---

## References

[1] Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., \& Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv preprint arXiv:1907.11692*.

[2] Hartmann, J., Heitmann, M., Siebert, C., \& Schamp, C. (2023). More than a Feeling: Accuracy and Application of Sentiment Analysis. *International Journal of Research in Marketing*, 40(1), 75–87.

> 

---

## Appendix

### A. EXP01 完整超參數設定

**模型架構**

| 項目 | 設定 |
|------|------|
| Base model | roberta-large |
| 分類頭 | Linear(1024→64) → ReLU → Dropout(0.1) → Linear(64→2) |
| 精度 | FP32 |

**Stage 1**

| 超參數 | 值 |
|--------|----|
| 資料 | labeled only（1,600） |
| Encoder | 全凍結 |
| LR（head） | 1e-4 |
| Batch size | 32 |
| Epochs | 10 |
| Max length | 128 |
| Label smoothing | 0.1 |
| Warmup ratio | 0.1 |
| Confidence threshold | 0.90 |
| Seed | 42 |
| n_folds | 5 |

**Stage 2**

| 超參數 | 值 |
|--------|----|
| 資料 | labeled + pseudo_pool_S1（本次為 0） |
| Encoder | 解凍後 3 層（共 24 層） |
| LR（encoder） | 1e-5 |
| LR（head） | 1e-4 |
| Batch size | 32 |
| Epochs | 10 |
| Max length | 128 |
| Label smoothing | 0.1 |
| Warmup ratio | 0.1 |
| Confidence threshold | 0.90 |

**Stage 3**

| 超參數 | 值 |
|--------|----|
| 資料 | labeled + pseudo_pool_S2 |
| LR（encoder） | 2e-6 |
| LR（head） | 5e-5 |
| Batch size | 32 |
| Epochs | 15 |
| Max length | 128 |
| Label smoothing | 0.1 |
| Warmup ratio | 0.05 |

---

### B. EXP02 完整超參數設定

**模型架構**

| 項目 | 設定 |
|------|------|
| Base model | roberta-large |
| 分類頭 | HF 預設 RobertaClassificationHead |
| 精度 | FP16（mixed precision） |

**Phase 0**

| 超參數 | 值 |
|--------|----|
| 資料 | labeled 全量（不做 fold 切割，1,600 筆） |
| LR | 1e-5 |
| Batch size | 8 |
| Gradient accumulation | 2（effective batch = 16） |
| Epochs | 3 |
| Max length | 192 |
| Warmup ratio | 0.1 |
| Weight decay | 0.05 |
| Confidence threshold | 無（全量 11,000 筆） |
| Seed | 42 |

**Phase 1**

| 超參數 | 值 |
|--------|----|
| 資料 | labeled_fold_train + 11,000 pseudo-labels |
| LR | 1e-5 |
| Batch size | 8 |
| Gradient accumulation | 2（effective batch = 16） |
| Epochs | 3（+ EarlyStop） |
| Early stopping patience | 2（監控 val F1） |
| Max length | 192 |
| Warmup ratio | 0.1 |
| Weight decay | 0.05 |
| 類別平衡 | Balanced class weights（WeightedTrainer） |
| n_folds | 5 |
| Seed | 42 |

---

### C. clean_text 縮寫修正規則（EXP02）

| 原始型態 | 修正後 |
|---------|--------|
| `(\w+) t` | `not`（e.g. `can t` → `cannot`） |
| `i m` | `i am` |
| `it s` | `it is` |
| `(\w+) re` | ` are` |
| `(\w+) ve` | ` have` |
| `(\w+) ll` | ` will` |
