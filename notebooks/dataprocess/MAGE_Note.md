# MAGE: Multi-Head Attention Guided Embeddings for Low Resource Sentiment Classification

**Authors**: Varun Vashisht, Samar Singh, Mihir Konduskar, Jaskaran Singh Walia, Vukosi Marivate  
**Venue / Journal**: arXiv preprint  
**Year**: 2025  
**arXiv ID**: 2502.17987  
**PDF**: https://arxiv.org/pdf/2502.17987.pdf

---

## 1. 目標解決問題 (Problem)

低資源語言缺乏高品質的情緒標籤資料，傳統資料增強方法往往無法捕捉語言特定的句法與語義特徵，導致增強樣本品質不佳。本文提出一種將**語言無關資料增強（Language-Independent Data Augmentation, LiDA）**與**多頭注意力加權嵌入（Multi-Head Attention weighted embeddings）**整合的框架，在資料稀缺場景下提升情緒分類性能。本文的技術路線對本專案的低樣本量（2,000 筆）情緒分類任務具有參考價值。

## 2. 方法 (Approach)

MAGE 框架的兩大核心元件：

1. **Language-Independent Data Augmentation (LiDA)**：一種語言無關的增強策略，透過詞向量空間中的鄰域採樣（neighborhood sampling in embedding space）生成新樣本，不依賴特定語言的語法規則或詞典資源，適用於任何語言（包括英文）。
2. **Multi-Head Attention Guided Embeddings**：在 Transformer 編碼器上加入多頭注意力權重引導機制，對增強後的關鍵資料點（critical data points）賦予更高的表示權重，選擇性地強化對分類最有貢獻的特徵維度。

兩者的結合使模型既能從擴充的資料中學習，又能聚焦於語義最具辨別力的嵌入方向。

## 3. 結果 (Results)

在 Bantu 語言低資源情緒分類資料集上評估（與 BERT baseline 和其他增強方法比較）：

- MAGE 在所有低資源場景下均**優於**僅使用 BERT-base 微調的基線。
- 相比標準 BERT 微調，F1-score 提升約 **+3%–+6%**（依語言與資料量而異）。
- LiDA 增強策略即使在 **100–500 樣本**的極低資源設定下仍保持穩定效能提升。
- 多頭注意力引導機制對少樣本情境下的特徵穩定性有顯著改善。

## 4. 與本專案的關聯 (Relevance to Our Project)

MAGE 中的 LiDA 嵌入空間增強策略可直接應用於英文情緒分類：在 `bert-base-uncased` 或 `roberta-base` 的嵌入空間中，對每個訓練樣本的嵌入向量加入高斯噪聲（Gaussian noise）或進行插值（interpolation），在不改變語義的前提下增加訓練多樣性。此方法實作成本低（無需外部 API）、計算效率高，適合在本地 GPU 環境對 2,000 筆樣本進行批次增強。多頭注意力引導機制可作為模型微調時的額外注意力監督損失項。
