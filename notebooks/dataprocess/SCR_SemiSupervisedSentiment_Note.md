# Semantic Consistency Regularization with Large Language Models for Semi-supervised Sentiment Analysis

**Authors**: Kunrong Li, Xinyu Liu, Zhen Chen  
**Venue / Journal**: ICONIP 2024  
**Year**: 2025 (arXiv submission) / 2024 (conference)  
**arXiv ID**: 2501.17598  
**PDF**: https://arxiv.org/pdf/2501.17598.pdf

---

## 1. 目標解決問題 (Problem)

人工標注大規模情緒語料耗時費力，如何有效利用大量未標籤文本（unlabeled data）輔助情緒分類訓練，是半監督學習的核心挑戰。現有半監督方法依賴未標籤資料的內在分布特性，泛化能力有限且容易在情緒場景過擬合。本文的問題設定與本專案高度相關：我們同樣擁有少量標籤樣本（2,000 筆）與大量未標籤樣本（11,000 筆測試集）。

## 2. 方法 (Approach)

SCR（Semantic Consistency Regularization）框架採用兩種 LLM 提示策略對未標籤文本進行語義增強：

1. **實體增強（SCR-EE, Entity-based Enhancement）**：從原始句子中提取實體和數字資訊，引導 LLM 重新建構等語義文本，保留核心情緒實體。
2. **概念增強（SCR-CE, Concept-based Enhancement）**：直接將原始句子輸入 LLM，要求進行語義重建（semantic reconstruction）。

增強後的文本與原始文本計算一致性損失（consistency loss），搭配**信心閾值（confidence thresholding）**篩選高品質樣本，提供額外的半監督訓練信號。此外提出**類別重組策略（class re-assembling）**，充分利用不確定性高的樣本。

## 3. 結果 (Results)

在多個情緒分類基準資料集上與主流半監督方法比較：

- SCR 在所有評估指標上**顯著優於**先前的半監督文本分類方法（MixText、UDA、BERT-KNN 等）。
- 在低標籤率設定（每類 20 個標籤樣本）下，SCR-CE 達到 **89.3% 準確率**，較次優基線提升 **+2.1%**。
- SCR-EE 在含有豐富實體的資料集上效果尤為突出，語義一致性指標提升明顯。

## 4. 與本專案的關聯 (Relevance to Our Project)

本文框架對本專案具有直接指導意義：可將 2,000 筆標籤樣本作為有標籤資料，11,000 筆未標籤測試集的文本部分作為無標籤資料，採用 SCR 的半監督框架訓練情緒分類器。SCR-CE 策略（概念增強）實作成本最低，可使用開源 LLM 或 ChatGPT API 進行語義重建，再配合信心篩選機制利用偽標籤提升模型效能。
