# BERT-Type NLP Model Papers

> Collected: 2026-04-22  
> Task: Binary Sentiment Classification (Positive=1 / Negative=0)  
> Dataset: 2,000 balanced training samples, 11,000 unlabeled test samples

## Overview

本目錄收錄 10 篇 **BERT-type encoder 模型**相關論文，涵蓋：
- 核心預訓練架構（RoBERTa、ELECTRA、DeBERTaV3、ModernBERT）
- 句子嵌入（Sentence-BERT、SimCSE）
- 少樣本分類（SetFit）
- 參數高效微調（LoRA、Adapter）
- 知識蒸餾（TinyBERT）

---

## Paper Index

| # | File | Title | Year | Model / Technique |
|---|------|-------|------|-------------------|
| 1 | [ModernBERT_Note.md](ModernBERT_Note.md) | Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder | 2024 | ModernBERT (BERT-type encoder, 2T tokens) |
| 2 | [DeBERTaV3_Note.md](DeBERTaV3_Note.md) | DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training | 2023 | DeBERTaV3 + RTD + Gradient-Disentangled Embedding Sharing |
| 3 | [SetFit_Note.md](SetFit_Note.md) | Efficient Few-Shot Learning Without Prompts (SetFit) | 2022 | Sentence Transformer + Contrastive Few-shot Fine-tuning |
| 4 | [SimCSE_Note.md](SimCSE_Note.md) | SimCSE: Simple Contrastive Learning of Sentence Embeddings | 2021 | BERT + Contrastive Learning (dropout / NLI pairs) |
| 5 | [LoRA_BERT_Note.md](LoRA_BERT_Note.md) | LoRA: Low-Rank Adaptation of Large Language Models | 2021 | LoRA PEFT applied to RoBERTa / DeBERTa |
| 6 | [TinyBERT_Note.md](TinyBERT_Note.md) | TinyBERT: Distilling BERT for Natural Language Understanding | 2020 | Transformer Knowledge Distillation (attention + hidden states) |
| 7 | [ELECTRA_Note.md](ELECTRA_Note.md) | ELECTRA: Pre-training Text Encoders as Discriminators | 2020 | Replaced Token Detection pre-training |
| 8 | [RoBERTa_Note.md](RoBERTa_Note.md) | RoBERTa: A Robustly Optimized BERT Pretraining Approach | 2019 | RoBERTa (optimized BERT training) |
| 9 | [SentenceBERT_Note.md](SentenceBERT_Note.md) | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | 2019 | SBERT (siamese / triplet BERT) |
| 10 | [AdapterBERT_Note.md](AdapterBERT_Note.md) | Parameter-Efficient Transfer Learning for NLP | 2019 | Adapter modules for BERT fine-tuning |

---

## Recommended Reading Order for This Project

1. **RoBERTa** → 理解 BERT fine-tuning 基準
2. **ELECTRA** → 理解更高效預訓練任務（DeBERTaV3 的基礎）
3. **DeBERTaV3** → 目前最強 BERT-type encoder 之一
4. **ModernBERT** → 最新 2024 BERT-type encoder，推理最快
5. **Sentence-BERT** → 句子嵌入的奠基方法
6. **SimCSE** → 對比學習提升嵌入品質
7. **SetFit** → 結合 SBERT 的高效 few-shot 分類
8. **LoRA** → 參數高效 fine-tuning 降低過擬合風險
9. **AdapterBERT** → Adapter 模組 PEFT 方法
10. **TinyBERT** → 知識蒸餾壓縮模型

---

## Quick Strategy for 2,000-Sample Sentiment Classification

| Strategy | Model | Expected Accuracy |
|----------|-------|-------------------|
| Baseline fine-tune | `roberta-base` | ~93–95% |
| Strong fine-tune | `deberta-v3-base` | ~94–96% |
| Few-shot SBERT | SetFit + `all-mpnet-base-v2` | ~91–94% |
| LoRA fine-tune | `deberta-v3-base` + LoRA r=16 | ~93–95% |
| Semi-supervised | SimCSE unsup. pretraining → fine-tune | ~94–96% |
| 2024 SOTA | `ModernBERT-base` fine-tune | ~95–97% |
