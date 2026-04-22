# Papers PDF Download List

Generated: 2026-04-22

---

## NLP Model Papers (BERT-type Encoders)

Collected for the binary sentiment classification task (2,000 training samples).  
Notes directory: `notebooks/nlp/`

| # | Save As | Title | Year | arXiv | PDF Link |
|---|---------|-------|------|-------|----------|
| 1 | ModernBERT.pdf | Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference | 2024 | 2412.13663 | [Download](https://arxiv.org/pdf/2412.13663.pdf) |
| 2 | DeBERTaV3.pdf | DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing | 2023 | 2111.09543 | [Download](https://arxiv.org/pdf/2111.09543.pdf) |
| 3 | SetFit.pdf | Efficient Few-Shot Learning Without Prompts | 2022 | 2209.11055 | [Download](https://arxiv.org/pdf/2209.11055.pdf) |
| 4 | SimCSE.pdf | SimCSE: Simple Contrastive Learning of Sentence Embeddings | 2021 | 2104.08821 | [Download](https://arxiv.org/pdf/2104.08821.pdf) |
| 5 | LoRA.pdf | LoRA: Low-Rank Adaptation of Large Language Models | 2021 | 2106.09685 | [Download](https://arxiv.org/pdf/2106.09685.pdf) |
| 6 | TinyBERT.pdf | TinyBERT: Distilling BERT for Natural Language Understanding | 2020 | 1909.10351 | [Download](https://arxiv.org/pdf/1909.10351.pdf) |
| 7 | ELECTRA.pdf | ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators | 2020 | 2003.10555 | [Download](https://arxiv.org/pdf/2003.10555.pdf) |
| 8 | RoBERTa.pdf | RoBERTa: A Robustly Optimized BERT Pretraining Approach | 2019 | 1907.11692 | [Download](https://arxiv.org/pdf/1907.11692.pdf) |
| 9 | SentenceBERT.pdf | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | 2019 | 1908.10084 | [Download](https://arxiv.org/pdf/1908.10084.pdf) |
| 10 | AdapterBERT.pdf | Parameter-Efficient Transfer Learning for NLP | 2019 | 1902.00751 | [Download](https://arxiv.org/pdf/1902.00751.pdf) |

---

## Note on Data Processing Papers

See [notebooks/dataprocess/README.md](../dataprocess/README.md) for the data augmentation papers collected separately.

The dataprocess papers cover techniques such as:
- **EDA** (Easy Data Augmentation)
- **AugGPT** (LLM-based augmentation)
- **CoDa / SSMix / TransformerAug** (various text augmentation methods)
- **RobustSentimentAug / RankAug** (sentiment-specific augmentation)
- **SCR Semi-Supervised Sentiment**
- **CoTAM / MAGE / LimitedDataSentiment**

---

## Download Instructions

To download all NLP model papers at once, run in terminal:
```bash
# Create download directory
mkdir -p data/papers/nlp

# Download using curl (Windows PowerShell)
$papers = @(
    "2412.13663", "2111.09543", "2209.11055", "2104.08821",
    "2106.09685", "1909.10351", "2003.10555", "1907.11692",
    "1908.10084", "1902.00751"
)
foreach ($id in $papers) {
    Invoke-WebRequest -Uri "https://arxiv.org/pdf/$id.pdf" -OutFile "data/papers/nlp/$id.pdf"
}
```
