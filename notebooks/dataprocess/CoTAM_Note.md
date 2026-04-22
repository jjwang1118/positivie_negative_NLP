# Controllable Data Augmentation for Few-Shot Text Mining with Chain-of-Thought Attribute Manipulation

**Authors**: Letian Peng, Yuwei Zhang, Jingbo Shang  
**Venue / Journal**: arXiv preprint (v3 updated May 2024)  
**Year**: 2024  
**arXiv ID**: 2307.07099  
**PDF**: https://arxiv.org/pdf/2307.07099.pdf

---

## 1. 目標解決問題 (Problem)

現有 LLM-based 資料增強方法往往僅靠 prompt 描述任務，無法精確控制生成樣本的特定屬性（如情緒極性、主題類別），導致增強樣本標籤污染或語義漂移。本文提出透過「屬性操控（attribute manipulation）」實現可控增強，對本專案二元情緒分類（正面/負面）的精確標籤一致性尤為重要。

## 2. 方法 (Approach)

CoTAM（Chain-of-Thought Attribute Manipulation）透過三步驟鏈式思維（Chain-of-Thought）流程對現有樣本進行受控屬性翻轉：

1. **屬性分解（Attribute Decomposition）**：將任務屬性（如情緒極性）分解為可操作的子屬性（如情感詞、否定詞、強調表達）。
2. **操控提案（Manipulation Proposal）**：讓 LLM 識別並提出如何修改原文以改變目標屬性（如將正面情緒改為負面），同時保留其他非目標屬性不變。
3. **句子重建（Sentence Reconstruction）**：依據操控提案重新生成完整句子，確保流暢性。

這種方法特別適合從現有標籤樣本生成反向情緒（flipped polarity）的增強資料。

## 3. 結果 (Results)

在文本分類、情緒分析（SST-2、IMDB）、條件文本生成等多個任務上驗證：

- 在 **SST-2 二元情緒分類**任務中，few-shot 設定（16 樣本/類）下，CoTAM 達到 **91.8% 準確率**，優於 GPT3Mix（88.3%）和直接提示增強（89.1%）。
- PCA 可視化顯示 CoTAM 增強資料集具有清晰、人類可識別的決策邊界。
- 在 aspect-based 情緒分析任務上，micro-F1 相比最佳基線提升 **+2.3%**。

## 4. 與本專案的關聯 (Relevance to Our Project)

CoTAM 方法最適合本專案的應用場景：從正面（label=1）樣本生成對應的負面（label=0）擴充樣本，或反之，透過情緒極性翻轉增強邊界樣本的多樣性。具體實作：以 GPT-4o mini 或 Claude Haiku 為後端，對每筆訓練樣本執行屬性翻轉增強，每類各生成 500–1,000 筆翻轉版本，使訓練集擴充至約 4,000 筆，同時保持正負類完美平衡（1:1 比例）。
