# SSMix: Saliency-Based Span Mixup for Text Classification

**Authors**: Soyoung Yoon, Gyuwan Kim, Kyumin Park  
**Venue / Journal**: Findings of ACL 2021  
**Year**: 2021  
**arXiv ID**: 2106.08062  
**PDF**: https://arxiv.org/pdf/2106.08062.pdf

---

## 1. 目標解決問題 (Problem)

Mixup 是一種在電腦視覺領域高效的資料增強技術，但直接套用至文本資料面臨挑戰：文本的離散性使得插值操作難以保留語義完整性，且隨機混合詞彙會破壞情緒極性一致性。本文提出 SSMix，一種基於**顯著性（saliency）**引導的文本 Mixup 方法，智慧選擇最具分類意義的文本片段進行混合，特別適用於情緒分類場景。

## 2. 方法 (Approach)

SSMix 在**輸入詞彙空間（token space）**進行 Mixup，而非嵌入空間，流程如下：

1. **顯著性分數計算（Saliency Scoring）**：計算每個 token 對當前分類決策的梯度顯著性分數（gradient-based saliency），識別對情緒判斷最重要的 span（如 "absolutely wonderful"、"terrible experience"）。
2. **Span 選擇（Span Selection）**：以高顯著性分數的 span 作為「關鍵保留區域」，低顯著性的 span 作為「可替換區域」。
3. **Mixup 操作**：將樣本 A 的低顯著性 span 替換為樣本 B 的對應 span，同時依照替換比例對標籤進行軟標籤混合（soft label blending）。

此方法確保高度相關的情緒詞彙/片段在增強過程中被保留，同時引入語境多樣性。

## 3. 結果 (Results)

在多個文本分類基準上驗證（SST-2、TREC、AGNews、IMDB 等）：

- SSMix 在所有基準上**優於** vanilla Mixup 和 TMix（在嵌入空間的 Mixup）。
- 在 **SST-2** 情緒分類任務上，SSMix 達到 **95.4% 準確率**，優於 TMix（95.0%）和標準微調（94.8%）。
- 在 **IMDB** 長文本情緒分類上，準確率提升 **+0.5%**，達到 **95.6%**。
- 小資料集（500 樣本）實驗顯示 SSMix 帶來 **+1.3%–+2.1%** 的顯著準確率提升。

## 4. 與本專案的關聯 (Relevance to Our Project)

SSMix 是本專案在**訓練過程中即時增強**的理想方案：相比靜態增強方法，SSMix 在每個 batch 訓練時動態生成混合樣本，無需預先生成大量增強資料。實作方式：在 `BertForSequenceClassification` 訓練迴圈中，計算每個 token 的梯度顯著性並按批次執行 SSMix。此方法特別適合本專案資料集（文本較短的評論/評論類資料），可在不增加磁碟儲存空間的前提下擴充模型訓練的資料多樣性。建議搭配軟標籤損失（soft label loss）而非硬標籤損失以充分利用混合標籤。
