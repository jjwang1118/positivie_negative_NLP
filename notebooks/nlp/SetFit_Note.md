# Efficient Few-Shot Learning Without Prompts (SetFit)

**Authors**: Lewis Tunstall, Nils Reimers, Unso Eun Seo Jo, Luke Bates, Daniel Korat, Moshe Wasserblat, Oren Pereg
**Venue / Journal**: arXiv preprint
**Year**: 2022
**arXiv ID**: 2209.11055
**PDF**: https://arxiv.org/pdf/2209.11055.pdf

---

## 1. 目標解決問題 (Problem)

現有少樣本（few-shot）方法如 PEFT（parameter-efficient fine-tuning）與 PET（pattern-exploiting training）雖效果良好，但嚴重依賴人工設計的 prompt，且通常需要數十億參數的語言模型。本論文提出 SetFit（Sentence Transformer Fine-tuning），一個無需 prompt 的高效 few-shot 分類框架，以 Sentence Transformers（BERT-type encoder）為骨幹，在極少樣本下實現高準確率。

## 2. 方法 (Approach)

SetFit 分兩階段訓練：(1) **Contrastive Fine-tuning**：從訓練集中採樣少量文本對（相同類別 = 正例對，不同類別 = 負例對），以 Siamese 網路結構對預訓練 Sentence Transformer（如 `all-mpnet-base-v2`）進行對比學習 fine-tuning，使同類樣本的嵌入靠近、異類分離；(2) **Classification Head 訓練**：使用 fine-tuned ST 為全部訓練樣本生成嵌入，訓練一個輕量分類頭（logistic regression 或多層感知機）。整個流程不需要任何 prompt 或 verbalizer。

## 3. 結果 (Results)

在 8-shot 設定下，SetFit 以 **110M 參數的 SBERT 模型**達到與 PEFT 方法（使用 10B 參數 T0-3B）相當的性能。訓練速度比 PEFT 方法快一個數量級。在 SST-2（情感分類）等 9 個文本分類 benchmark 中，SetFit 在極低資料量下（8~64 shot）均達到 SOTA 或接近 SOTA 性能，多語言場景同樣有效。

## 4. 與本專案的關聯 (Relevance to Our Project)

SetFit 完美契合本專案場景：2,000 筆標注訓練資料在 few-shot 分類框架下屬於豐富資源，理論上可輕易超越 8-shot 基線。實作流程簡單：安裝 `setfit` 套件，選用 `sentence-transformers/all-mpnet-base-v2` 或 `BAAI/bge-base-en-v1.5` 作為骨幹，以本專案資料直接訓練即可。此外，11,000 筆無標注測試資料可用於無監督 SimCSE 預熱 ST，進一步提升嵌入品質。
