# ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators

**Authors**: Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning
**Venue / Journal**: ICLR 2020
**Year**: 2020
**arXiv ID**: 2003.10555
**PDF**: https://arxiv.org/pdf/2003.10555.pdf

---

## 1. 目標解決問題 (Problem)

BERT 的 Masked Language Modeling (MLM) 預訓練任務只在 15% 的 masked token 上計算損失，樣本效率偏低，需要大量算力才能達到高性能。ELECTRA 提出更高效的預訓練任務——Replaced Token Detection (RTD)，在所有 token 上定義學習信號，大幅提升樣本效率。

## 2. 方法 (Approach)

ELECTRA 架構包含兩個模型：(1) **小型 Generator**（類似 BERT MLM）：對輸入中部分 token 進行遮罩，預測合理的替換 token；(2) **大型 Discriminator**（即 ELECTRA 主體）：接收 generator 輸出的已替換序列，以二元分類預測每個 token 是否為原始 token 或被替換的假 token。Discriminator 對全序列每個 token 都計算損失，不只是 masked 位置。預訓練結束後只保留 discriminator 用於下游任務 fine-tuning。

## 3. 結果 (Results)

ELECTRA-Small 在 **1 個 GPU 訓練 4 天**即可在 GLUE benchmark 超越 GPT（後者使用 30 倍算力）。ELECTRA-Large 與 RoBERTa Large 和 XLNet 性能相當，但僅用其 **1/4 的計算量**。在相同算力下 ELECTRA 表現優於 RoBERTa 與 XLNet。ELECTRA-small 在 GLUE 達 73.1（BERT-base 為 79.6，但 ELECTRA-small 遠比 BERT-base 小）。

## 4. 與本專案的關聯 (Relevance to Our Project)

ELECTRA-small 或 ELECTRA-base 是本專案小資料情感分類的高效選擇：其以少量算力達到接近 RoBERTa 的性能，fine-tuning 時更節省 GPU 記憶體。DeBERTaV3 正是以 ELECTRA 的 RTD 機制結合 DeBERTa 的 Disentangled Attention 打造，理解 ELECTRA 有助於理解 DeBERTaV3 的優勢來源。本專案可以 `google/electra-base-discriminator` 為 backbone 進行二元情感分類 fine-tuning。
