# TinyBERT: Distilling BERT for Natural Language Understanding

**Authors**: Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu
**Venue / Journal**: Findings of EMNLP 2020
**Year**: 2020
**arXiv ID**: 1909.10351
**PDF**: https://arxiv.org/pdf/1909.10351.pdf

---

## 1. 目標解決問題 (Problem)

BERT 等大型預訓練模型在推理時計算成本高、延遲大，難以部署於資源受限的裝置。知識蒸餾（Knowledge Distillation, KD）可將大型 teacher 模型的知識壓縮至小型 student 模型，但現有方法僅蒸餾最終輸出，忽略 Transformer 中間層的豐富知識。TinyBERT 提出針對 Transformer 架構設計的多層蒸餾方法。

## 2. approach (Approach)

TinyBERT 引入 **Transformer-specific Knowledge Distillation**，在以下四個層級蒸餾教師 BERT 的知識：(1) **Embedding layer 輸出**；(2) **每個 Transformer layer 的 attention matrix**（注意力頭的分佈）；(3) **每個 Transformer layer 的 hidden state 輸出**；(4) **最終預測層（logits）的 soft labels**。採用兩階段學習框架：**General Distillation**（在通用語料預訓練階段蒸餾）與 **Task-specific Distillation**（在特定任務 fine-tuning 階段再次蒸餾），並搭配資料增強（word substitution with BERT MLM）擴充任務訓練資料。

## 3. 結果 (Results)

TinyBERT-4L 在 GLUE benchmark 達到 BERT-base 性能的 **96.8%**，模型尺寸縮小 **7.5 倍**，推理速度提升 **9.4 倍**。TinyBERT-6L 性能與 BERT-base 相當，但比 4-layer 基線（DistilBERT）在 GLUE 多項任務上表現更佳，尤其在資料增強配合下效果顯著。在 SST-2（情感分析）任務，TinyBERT-4L 達到 **93.0%**（BERT-base: 93.5%）。

## 4. 與本專案的關聯 (Relevance to Our Project)

TinyBERT 代表 BERT 知識蒸餾的標準方法。本專案可應用 TinyBERT 概念的兩種方式：(1) 直接使用預訓練 TinyBERT（`huawei-noah/TinyBERT_General_4L_312D`）對 2,000 筆資料 fine-tune，獲得輕量高速的情感分類器；(2) 以 DeBERTaV3 或 RoBERTa-large 為 teacher，對自訂小型 BERT 進行 task-specific distillation，壓縮模型以降低預測延遲。對小資料集而言，蒸餾的資料增強步驟尤其有價值。
