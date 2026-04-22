# Robust Sentiment Analysis for Low Resource Languages Using Data Augmentation Approaches: A Case Study in Marathi

**Authors**: Aabha Pingle, Aditya Vyawahare, Isha Joshi, Rahul Tangsali, Geetanjali Kale, Raviraj Joshi  
**Venue / Journal**: arXiv preprint  
**Year**: 2023  
**arXiv ID**: 2310.00734  
**PDF**: https://arxiv.org/pdf/2310.00734.pdf

---

## 1. 目標解決問題 (Problem)

低資源語言的情緒分析資料集通常規模較小且領域受限，難以訓練出具跨域泛化能力的分類模型。本文系統性地研究在低資源情緒分析場景下，多種資料增強方法對**域內（in-domain）**和**跨域（cross-domain）**準確率的影響。雖以 Marathi 語為案例，但其增強框架與結論對本專案的英文二元情緒分類同樣適用。

## 2. 方法 (Approach)

本文提出並比較四大類資料增強技術，涵蓋從傳統到基於大型模型的方法：

1. **回譯（Back-Translation）**：將原文翻譯至中間語言（如英語）再譯回，生成表面形式不同但語義保留的新樣本。
2. **BERT-based 增強**：
   - **隨機詞替換（Random Token Replacement）**：以 BERT 的 masked LM 預測替換隨機選取的詞彙。
   - **命名實體替換（Named Entity Replacement）**：以同類型實體替換原文中的命名實體（如人名、地名），保留情緒結構。
3. **GPT-based 增強**：
   - **文本生成（Text Generation）**：以 GPT 模型生成同樣情緒標籤的全新樣本。
   - **偽標籤生成（Pseudo-label Generation）**：對未標籤樣本生成預測標籤，擴充訓練集。

## 3. 結果 (Results)

在 Marathi 情緒分析資料集（域內與跨域）上驗證，使用 BERT-based 模型評估：

- **回譯**帶來最穩定的跨域泛化提升，跨域準確率提升 **+4.2%–+8.7%**。
- **GPT-based 文本生成**在域內評估表現最佳，F1 提升達 **+5.8%**。
- **偽標籤生成**策略有效利用未標籤資料，跨域準確率額外提升 **+2.3%**。
- 命名實體替換在含豐富實體的文本上效果顯著，但對短評論效果有限。

## 4. 與本專案的關聯 (Relevance to Our Project)

本文的多方法比較框架為本專案提供了完整的增強策略選擇指南。建議按以下優先順序應用至 2,000 筆英文訓練樣本：（1）**回譯**（英→法→英 或 英→德→英）作為基礎增強；（2）**BERT masked LM 替換**（使用 `bert-base-uncased` + `nlpaug`）快速生成大量變體；（3）**GPT-based 文本生成**為低信心預測類別補充高品質樣本。偽標籤策略可在第一輪模型訓練完成後，對 11,000 筆測試集文本進行標籤預測並篩選高信心樣本加入訓練。
