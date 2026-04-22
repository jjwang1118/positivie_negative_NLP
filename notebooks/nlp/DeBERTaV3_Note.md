# DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing

**Authors**: Pengcheng He, Jianfeng Gao, Weizhu Chen
**Venue / Journal**: ICLR 2023
**Year**: 2023
**arXiv ID**: 2111.09543
**PDF**: https://arxiv.org/pdf/2111.09543.pdf

---

## 1. 目標解決問題 (Problem)

原始 DeBERTa 以 Masked Language Modeling (MLM) 預訓練，樣本效率有限。DeBERTaV3 結合 DeBERTa 的 Disentangled Attention 機制與 ELECTRA 的 Replaced Token Detection (RTD) 預訓練任務，但發現 ELECTRA 原版的 embedding sharing 在 discriminator 與 generator 之間造成「tug-of-war」梯度衝突，降低訓練效率。本論文解決此問題，同時為下游分類（含情感分析）提供更強的預訓練模型。

## 2. 方法 (Approach)

DeBERTaV3 採用三大核心設計：(1) **Disentangled Attention**：將詞義與位置表示分離，以相對位置偏置計算注意力分數；(2) **RTD（Replaced Token Detection）預訓練**：由小型 generator 生成候選 token，discriminator（即 DeBERTaV3）判斷每個 token 是否被替換，定義在全部 token 上的信號比 MLM 的 15% masked tokens 更豐富；(3) **Gradient-Disentangled Embedding Sharing**：在 generator 與 discriminator 共享 embedding 時解耦梯度，避免兩者的學習目標互相干擾。

## 3. 結果 (Results)

DeBERTaV3 Large 在 GLUE benchmark（8 tasks）達到 **91.37% 平均分**，較 DeBERTa Large 提升 1.37%，較 ELECTRA Large 提升 1.91%，創下同等規模模型的 SOTA。XSmall 版本僅 22M 參數，仍顯著優於 RoBERTa-base 與 XLNet-base。多語言版 mDeBERTa-base 在 XNLI zero-shot 跨語言任務達 79.8%，超越 XLM-R Base 3.6%。

## 4. 與本專案的關聯 (Relevance to Our Project)

DeBERTaV3 是目前在 NLU 分類任務上性能最強的 BERT-type encoder 之一，特別適合 fine-tuning 至二元情感分類任務（SST-2、IMDb 等）。其 XSmall/Small 版本參數少，在 2,000 筆訓練樣本的低資源場景下不易過擬合，可作為本專案的強力基線模型。建議使用 `microsoft/deberta-v3-base` 或 `deberta-v3-small` 直接 fine-tune 分類頭。
