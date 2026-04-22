# "Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME)

**Authors**: Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin  
**Venue / Journal**: KDD 2016  
**Year**: 2016  
**arXiv ID**: 1602.04938  
**PDF**: https://arxiv.org/pdf/1602.04938.pdf  

---

## 1. 目標解決問題 (Problem)

深度學習模型在 NLP 分類任務（如情緒分析）中通常是黑箱，無法解釋為何做出某個預測。
LIME 旨在為任意分類器提供「局部可解釋性」——針對單一預測，找出哪些 token 最關鍵。
這與我們想透過「逐步加入 token 觀察分類變化」的想法直接對應。

## 2. 方法 (Approach)

LIME（Local Interpretable Model-Agnostic Explanations）的核心流程：

1. **擾動輸入**：對原始文字隨機遮蔽（mask）部分 token，生成大量擾動版本
2. **查詢黑箱**：將每個擾動版本餵入分類器，得到預測機率
3. **局部線性近似**：在「原始輸入的鄰域」中訓練一個可解釋的線性模型（Lasso）
4. **輸出重要性**：線性模型的係數代表每個 token 對該預測的貢獻分數

對 NLP 任務，重要 token 即為影響情緒分類的 **key strings**。

## 3. 結果 (Results)

- 在文字分類（20 Newsgroups、sentiment）上，LIME 的解釋讓人類能識別模型依賴不可靠特徵（如停用詞）
- 用戶實驗顯示 LIME 解釋顯著提升了人類對模型預測的信任判斷準確率
- 線性近似在局部區域保真度（faithfulness）高，但全局解釋能力有限

## 4. 與本專案的關聯 (Relevance to Our Project)

LIME 可直接應用於我們的 BERT 分類器：對每個測試樣本找出最重要的 token，
等同於自動找出代表正/負面情緒的 **key substring**。
相比我們的 Greedy Token Selection，LIME 效率更高（不需逐一測試所有子集），
但犧牲了精確的 Shapley 保證，適合用作快速可視化工具。
