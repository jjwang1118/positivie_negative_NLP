# RankAug: Augmented Data Ranking for Text Classification

**Authors**: Tiasa Singha Roy, Priyam Basu  
**Venue / Journal**: GEM Workshop at EMNLP 2023  
**Year**: 2023  
**arXiv ID**: 2311.04535  
**PDF**: https://arxiv.org/pdf/2311.04535.pdf

---

## 1. 目標解決問題 (Problem)

文本增強研究長期聚焦於生成模型的改進，而忽略了對生成合成資料品質的系統性評估與篩選。低品質增強樣本不僅無益於模型訓練，甚至可能引入語義噪音，損害分類效能。本文探討如何透過資料排序（data ranking）機制，精選最有價值的增強樣本，這對本專案確保增強後 2,000+ 樣本的品質尤為重要。

## 2. 方法 (Approach)

RankAug 提出以「文本排序」（text ranking）取代傳統的隨機取樣方式，流程如下：

1. 利用標準增強方法（如 EDA、回譯、LLM 生成等）生成大量候選增強樣本。
2. 計算每個候選樣本與原始樣本的**語義相似度**（如 BERTScore、cosine similarity），確保語義一致性（faithfulness）。
3. 同時評估候選樣本的**詞彙與句法多樣性**（lexical and syntactical diversity），如 n-gram 重疊率、BLEU 指標的逆向使用。
4. 依據兩項指標的綜合分數對候選樣本排序，選取排名靠前的樣本加入訓練集，**過濾低品質增強樣本**。

## 3. 結果 (Results)

在意圖分類（Intent Classification）和情緒分類（Sentiment Classification）任務上，針對**不平衡類別**進行測試：

- RankAug 相比直接使用所有增強樣本（無過濾），對**低頻（under-represented）類別**的分類準確率提升高達 **+35%**。
- 整體 macro F1-score 在多個資料集上提升 **+5%～+12%**。
- 對比實驗顯示，資料篩選策略比生成方法的選擇更影響最終性能。

## 4. 與本專案的關聯 (Relevance to Our Project)

RankAug 的核心洞察對本專案至關重要：在應用 EDA、AugGPT 或 CoDa 等方法生成大量增強樣本後，**不應直接全部使用**，而應先以 BERTScore 或 cosine similarity 篩選語義一致的樣本，再以 n-gram 多樣性指標排除過度相似的樣本。實作上可使用 `sentence-transformers` 計算相似度矩陣，建議最終保留每個原始樣本前 Top-3 個增強版本，以平衡品質與資料量。
