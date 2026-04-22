# LoRA: Low-Rank Adaptation of Large Language Models

**Authors**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
**Venue / Journal**: ICLR 2022
**Year**: 2021 (published 2022)
**arXiv ID**: 2106.09685
**PDF**: https://arxiv.org/pdf/2106.09685.pdf

---

## 1. 目標解決問題 (Problem)

對大型預訓練模型進行完整 full fine-tuning 需要更新所有參數，計算與儲存成本隨模型規模線性增長。對每個下游任務都保存一份完整的 fine-tuned 模型既昂貴又不靈活。LoRA 提出一種訓練效率極高的參數高效 fine-tuning 方法，凍結預訓練模型並在 Transformer 層中注入可訓練的低秩矩陣，大幅降低可訓練參數數量，同時維持性能。

## 2. 方法 (Approach)

LoRA 的核心假設：預訓練模型的權重更新具有低 intrinsic rank。對 Transformer 中的 query/value projection 矩陣 $W \in \mathbb{R}^{d \times k}$，LoRA 注入兩個可訓練低秩矩陣 $A \in \mathbb{R}^{d \times r}$ 與 $B \in \mathbb{R}^{r \times k}$（$r \ll \min(d,k)$），以 $\Delta W = BA$ 近似參數更新，原始權重 $W$ 完全凍結。在推理時直接合併 $W + \Delta W$，無額外推理延遲。論文在 RoBERTa、DeBERTa（BERT-type encoder）及 GPT-2、GPT-3 上均做了實驗。

## 3. 結果 (Results)

在 GLUE benchmark 上，LoRA fine-tuned RoBERTa-base 達到與 full fine-tuning 相當的性能（平均差距 < 0.5%），可訓練參數僅為原來的 **0.3%**（從 125M 降至約 0.3M）。LoRA fine-tuned DeBERTaXXL 達到 90.7 GLUE 分，不遜於 full fine-tuning 的 91.0。相較 Adapter 方法，LoRA 無額外推理延遲，訓練吞吐量更高。

## 4. 與本專案的關聯 (Relevance to Our Project)

LoRA 可顯著降低 fine-tuning 成本：在 2,000 筆訓練樣本的場景下，使用 LoRA fine-tuning DeBERTaV3-base 或 RoBERTa-base，僅需更新少量低秩矩陣，(1) 大幅縮短訓練時間；(2) 降低過擬合風險（更少自由度）；(3) 節省 GPU 記憶體，可於消費級 GPU 訓練。可透過 HuggingFace `peft` 套件一行呼叫啟用 LoRA，`r=8` 或 `r=16` 是常用設定。
