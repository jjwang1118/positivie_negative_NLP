# NLP Data Processing Papers

本目錄收錄針對 **NLP 資料處理**（文本增強、預處理、偽標籤、雜訊處理、半監督學習）的學術論文筆記，用於支援二元情緒分類任務（Positive=1 / Negative=0）。

## 論文清單

| # | 筆記檔案 | 論文標題 | 年份 | 主題 |
|---|----------|----------|------|------|
| 1 | [MAGE_Note.md](MAGE_Note.md) | MAGE: Multi-Head Attention Guided Embeddings for Low Resource Sentiment Classification | 2025 | 低資源增強 + 注意力加權嵌入 |
| 2 | [SCR_SemiSupervisedSentiment_Note.md](SCR_SemiSupervisedSentiment_Note.md) | Semantic Consistency Regularization with LLMs for Semi-supervised Sentiment Analysis | 2025 | 半監督學習 + LLM 語義增強 |
| 3 | [CoDa_Note.md](CoDa_Note.md) | CoDa: Constrained Generation based Data Augmentation for Low-Resource NLP | 2024 | 約束生成式增強（NAACL 2024） |
| 4 | [LimitedDataSentiment_Note.md](LimitedDataSentiment_Note.md) | New Directions in Text Classification: Maximizing Performance from Limited Data | 2024 | 少樣本情緒分類策略比較 |
| 5 | [CoTAM_Note.md](CoTAM_Note.md) | Controllable Data Augmentation for Few-Shot Text Mining with CoT Attribute Manipulation | 2024 | Chain-of-Thought 屬性操控增強 |
| 6 | [AugGPT_Note.md](AugGPT_Note.md) | AugGPT: Leveraging ChatGPT for Text Data Augmentation | 2023 | ChatGPT 改寫增強 |
| 7 | [RankAug_Note.md](RankAug_Note.md) | RankAug: Augmented Data Ranking for Text Classification | 2023 | 增強資料品質排序篩選（EMNLP 2023） |
| 8 | [RobustSentimentAug_Note.md](RobustSentimentAug_Note.md) | Robust Sentiment Analysis Using Data Augmentation: A Case Study in Marathi | 2023 | 多策略增強比較（回譯 + BERT + GPT + 偽標籤） |
| 9 | [SSMix_Note.md](SSMix_Note.md) | SSMix: Saliency-Based Span Mixup for Text Classification | 2021 | 顯著性引導 Mixup 增強（ACL 2021 Findings） |
| 10 | [TransformerAug_Note.md](TransformerAug_Note.md) | Data Augmentation using Pre-trained Transformer Models | 2020 | GPT-2 / BERT / BART 條件式增強（AACL 2020） |
| 11 | [UDA_Note.md](UDA_Note.md) | Unsupervised Data Augmentation for Consistency Training | 2020 | 半監督一致性訓練 + 回譯 + TF-IDF 詞替換 + TSA（NeurIPS 2020） |
| 12 | [EDA_Note.md](EDA_Note.md) | EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks | 2019 | 同義詞替換 / 隨機插入 / 交換 / 刪除（EMNLP 2019） |

> 注意：清單包含第 12 篇 EDA 作為經典基礎論文（共 12 篇筆記，前 11 篇為主要推薦）。

---

## 建議應用順序（針對本專案）

```
靜態增強（離線）:
  EDA → TransformerAug (BART) → AugGPT → CoDa → CoTAM
  ↓
增強資料品質篩選:
  RankAug (BERTScore + 多樣性過濾)
  ↓
動態增強（訓練時）:
  SSMix (顯著性 Mixup)
  ↓
半監督利用未標籤資料:
  UDA (一致性訓練 + TSA) → SCR (LLM 語義一致性) → RobustSentimentAug (偽標籤)
  ↓
嵌入空間增強:
  MAGE (LiDA + 注意力引導)
```

## 涵蓋主題範圍

| 主題 | 相關論文 |
|------|----------|
| 詞彙級增強（Word-level） | EDA, RobustSentimentAug |
| 句子級改寫（Paraphrase） | TransformerAug, AugGPT |
| 條件生成（Conditional Generation） | CoDa, CoTAM |
| 嵌入空間增強（Embedding-space） | SSMix, MAGE |
| 增強資料品質控制 | RankAug |
| 半監督 / 偽標籤 | UDA, SCR, RobustSentimentAug |
| 少樣本情緒分類策略 | LimitedDataSentiment |
