# SimCSE: Simple Contrastive Learning of Sentence Embeddings

**Authors**: Tianyu Gao, Xingcheng Yao, Danqi Chen
**Venue / Journal**: EMNLP 2021
**Year**: 2021
**arXiv ID**: 2104.08821
**PDF**: https://arxiv.org/pdf/2104.08821.pdf

---

## 1. 目標解決問題 (Problem)

預訓練語言模型（如 BERT）的句子嵌入空間具有各向異性（anisotropic），高維空間中向量分布不均勻，導致餘弦相似度計算效果差。SimCSE 提出極簡的對比學習框架，無需複雜資料增強即可顯著提升句子嵌入品質，為下游情感分類等任務提供更具判別力的表示。

## 2. 方法 (Approach)

SimCSE 提供兩種訓練方式：(1) **無監督（Unsupervised）**：對同一句子進行兩次前向傳播，利用不同的 dropout mask 作為資料增強，以 in-batch 負例進行對比學習（NTXent loss），使相同句子的兩個嵌入相互靠近，不同句子的嵌入分離；(2) **有監督（Supervised）**：利用 NLI 資料集，將 entailment 對作為正例、contradiction 對作為 hard negative，以 supervised contrastive loss 訓練。理論分析證明對比學習可使嵌入空間更均勻（uniform）並更好對齊正例（aligned）。

## 3. 結果 (Results)

以 BERT-base 為骨幹：無監督 SimCSE 在 STS 任務平均 Spearman 相關係數 **76.3%**（較前最佳提升 4.2%）；有監督 SimCSE 達 **81.6%**（較前最佳提升 2.2%）。以 RoBERTa-large 為骨幹時有監督 SimCSE 達 **86.2%**。在 SentEval transfer tasks（含情感分析 MR、SUBJ 等）亦達 SOTA。

## 4. 與本專案的關聯 (Relevance to Our Project)

SimCSE 的有監督對比學習框架可直接應用於本專案：在 2,000 筆訓練資料中，構建（正面句子, 正面句子增強版）正例對與（正面句子, 負面句子）難負例對，以對比學習預訓練 BERT encoder，再接 linear 分類頭 fine-tune。無監督版本更可利用未標注的 11,000 筆測試資料進行預訓練，改善表示空間後再做分類，是處理小資料集的有效策略。
