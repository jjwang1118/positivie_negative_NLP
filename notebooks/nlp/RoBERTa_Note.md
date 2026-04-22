# RoBERTa: A Robustly Optimized BERT Pretraining Approach

**Authors**: Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov
**Venue / Journal**: arXiv preprint (Meta AI Research)
**Year**: 2019
**arXiv ID**: 1907.11692
**PDF**: https://arxiv.org/pdf/1907.11692.pdf

---

## 1. 目標解決問題 (Problem)

原始 BERT 被嚴重低估訓練（undertrained）。本論文透過系統性消融實驗，重新檢視 BERT 預訓練中各超參數選擇（訓練步數、batch size、序列長度、dynamic masking、NSP 任務移除）的影響，發現調整這些設定後的 RoBERTa 可匹配甚至超越後繼發表的所有模型。

## 2. 方法 (Approach)

RoBERTa 在 BERT 架構基礎上做出以下優化：(1) **移除 Next Sentence Prediction (NSP) 任務**，改以更長的連續文本片段訓練；(2) **Dynamic Masking**：每次輸入時重新生成 masking pattern，而非靜態預先生成；(3) **更大 batch size 與更多訓練步數**：從 BERT 的 1M steps 擴展至 500K 步配合更大 batch；(4) **更多預訓練資料**（160GB，含 CC-News、OpenWebText 等）；(5) **Byte-level BPE 詞彙表**（50K tokens），取代 character-level BPE。下游任務以標準 [CLS] token + linear head fine-tuning。

## 3. 結果 (Results)

RoBERTa 在 GLUE benchmark 9 tasks 平均 **88.5 分**（BERT-large：80.5），在 SST-2（情感分析）達 **96.4% 準確率**。在 SQuAD 2.0（89.4 F1）與 RACE（86.4% 準確率）均達 SOTA。RoBERTa 成為此後數年 NLP fine-tuning 任務的最強基線之一。

## 4. 與本專案的關聯 (Relevance to Our Project)

RoBERTa 在 SST-2 達 96.4% 的準確率，代表其 fine-tune 後在二元情感分類的高品質上限。本專案可直接使用 `roberta-base` 或 `roberta-large` 作為 encoder，加接 linear classification head，以 2,000 筆訓練資料 fine-tune。`roberta-base`（125M 參數）在 2,000 筆小資料上通常能獲得良好結果而不嚴重過擬合，是強力的基線選擇。
