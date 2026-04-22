# Data Augmentation using Pre-trained Transformer Models

**Authors**: Varun Kumar, Ashutosh Choudhary, Eunah Cho  
**Venue / Journal**: Proceedings of the 2nd Workshop on Life-long Learning for Spoken Language Systems @ AACL 2020  
**Year**: 2020  
**arXiv ID**: 2003.02245  
**PDF**: https://arxiv.org/pdf/2003.02245.pdf

---

## 1. 目標解決問題 (Problem)

在低資源（low-resource）文本分類場景中，如何利用預訓練語言模型進行高品質條件式資料增強，並確保生成樣本的標籤一致性與多樣性，是本文探討的核心問題。對於本專案僅有 2,000 筆標籤樣本的情緒分類任務，需要能夠保留情緒極性（正面/負面）的生成式增強策略。

## 2. 方法 (Approach)

本文比較三類預訓練 Transformer 模型用於條件式文本增強：

1. **自回歸模型（GPT-2）**：將類別標籤前綴（prepend）至輸入序列，引導生成對應標籤的新文本。
2. **自編碼模型（BERT）**：利用 MLM（Masked Language Modeling）替換詞彙生成多樣化表面形式。
3. **Seq2Seq 模型（BART）**：以條件生成方式對輸入句子進行改寫（paraphrase），保留語義並改變表面形式。

關鍵技巧：在輸入序列前加入類別標籤文字（如 "positive: [text]"），無需修改模型結構即可實現條件控制。

## 3. 結果 (Results)

在三個文本分類基準（SST-2、TREC、Yelp Reviews）上評估：

- **BART（Seq2Seq）在低資源設定下效果最佳**，在 SST-2 上以 100 個訓練樣本達到 **83.2% 準確率**（vs. 無增強基線 78.5%）。
- GPT-2 生成的文本多樣性最高，但標籤準確率略低。
- BERT 增強樣本的語義保留度最佳，但多樣性相對有限。

## 4. 與本專案的關聯 (Relevance to Our Project)

本文方法對本專案高度適用：可使用 HuggingFace 的 `facebook/bart-base` 對 2,000 筆樣本進行條件改寫增強，透過在文字前加入 "positive: " 或 "negative: " 前綴確保生成內容與標籤一致。BART 生成的增強樣本多樣性強且語義準確，特別適合擴充正面/負面各 1,000 筆的平衡資料集。建議每筆樣本生成 3–5 個改寫版本後，搭配相似度篩選去除低品質增強樣本。
