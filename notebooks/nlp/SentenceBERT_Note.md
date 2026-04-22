# Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

**Authors**: Nils Reimers, Iryna Gurevych
**Venue / Journal**: EMNLP 2019
**Year**: 2019
**arXiv ID**: 1908.10084
**PDF**: https://arxiv.org/pdf/1908.10084.pdf

---

## 1. 目標解決問題 (Problem)

原始 BERT/RoBERTa 在計算句子相似度時需將兩句話同時輸入網路，對於 10,000 筆句子的相似度搜尋需執行約 5,000 萬次推理（約 65 小時），計算成本極高且不適合語義搜尋或分群任務。本論文提出 SBERT，使 BERT 能產生語義豐富的固定維度句子嵌入，以餘弦相似度直接比較。

## 2. 方法 (Approach)

SBERT 修改預訓練 BERT，加入 **Siamese 與 Triplet 網路結構**：兩個句子分別輸入共用同一組權重的 BERT encoder，透過 mean pooling（或 CLS token）取得句子嵌入。使用三種訓練目標：(1) **Classification objective**：將兩句子嵌入拼接後加分類頭，以 cross-entropy 訓練；(2) **Regression objective**：最小化嵌入餘弦相似度與標籤差；(3) **Triplet objective**：確保 anchor 嵌入比負例更接近正例。在 NLI 資料集（SNLI + MultiNLI）上 fine-tune，再遷移至 STS 任務。

## 3. 結果 (Results)

SBERT 在 STS benchmark 上達到與 BERT 互饋方式相當的準確度，但速度提升了 **65 小時 → 約 5 秒**（10,000 筆句子配對）。在 STS-B 上 Spearman 相關係數 0.8803（BERT cross-encoder: 0.8785），在多個 transfer learning 任務超越 InferSent 等句子嵌入基線。SRoBERTa 進一步提升性能。

## 4. 與本專案的關聯 (Relevance to Our Project)

SBERT 的 sentence embedding 可直接用於情感分類：(1) 使用預訓練 SBERT 模型（如 `all-MiniLM-L6-v2`）提取句子嵌入，再接輕量級分類器（logistic regression / MLP）；(2) 結合 SetFit 框架以極少樣本（8–64 筆）fine-tune SBERT 進行二元分類，非常適合本專案僅 2,000 筆訓練資料的低資源場景。SBERT 是 sentence embedding 領域的奠基論文，理解它有助於設計 contrastive 訓練策略。
