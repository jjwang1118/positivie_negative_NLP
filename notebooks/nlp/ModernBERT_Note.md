# Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference

**Authors**: Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, et al.
**Venue / Journal**: arXiv preprint
**Year**: 2024
**arXiv ID**: 2412.13663
**PDF**: https://arxiv.org/pdf/2412.13663.pdf

---

## 1. 目標解決問題 (Problem)

現有 BERT-type encoder 自 2018 年以來架構幾乎未有重大改進，在推理速度與記憶體效率上落後於現代 decoder 模型的工程進展。本論文旨在將近年深度學習工程優化（如 Flash Attention、Rotary Positional Embedding、modern training recipes）引入 encoder-only 架構，打造新一代 BERT 繼承者。對於我們僅有 2,000 筆訓練樣本的情感分析任務，ModernBERT 的高效推理與強大分類能力特別有吸引力。

## 2. 方法 (Approach)

ModernBERT 採用純 encoder-only bidirectional Transformer，融入以下現代化設計：使用 Rotary Positional Embeddings (RoPE) 取代固定式位置編碼，支援 8,192 tokens 的原生長序列；引入 Flash Attention 2 以降低記憶體使用量；採用改良的 pre-norm 層歸一化與 GeGLU activation；以 2 trillion tokens 超大語料進行預訓練。模型提供 Base（149M）與 Large（395M）兩種規格，下游任務 fine-tuning 使用標準 [CLS] token 接分類頭。

## 3. 結果 (Results)

ModernBERT 在涵蓋多樣化分類任務（GLUE、SuperGLUE）及單/多向量 retrieval（BEIR、MTEB）的大規模評測中達到 SOTA，全面優於 DeBERTa-v3 Large 與 RoBERTa Large。在推理速度上為現有 encoder 中最快，GPU 記憶體佔用最低，同時在程式碼理解 benchmark 亦具競爭力。GLUE benchmark 整體分數較 DeBERTaV3 Large 更高。

## 4. 與本專案的關聯 (Relevance to Our Project)

ModernBERT 是截至 2024 年底最先進的 BERT-type encoder，在分類與 retrieval 雙任務上均優於前代模型，且推理速度快、記憶體效率高。對於僅 2,000 筆訓練樣本的二元情感分析，可直接 fine-tune ModernBERT-base，利用其強大的預訓練表示獲得高準確率；亦可結合 SetFit 框架進行 few-shot 訓練，充分發揮小資料集場景下的優勢。
