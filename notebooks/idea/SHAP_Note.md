# A Unified Approach to Interpreting Model Predictions (SHAP)

**Authors**: Scott Lundberg, Su-In Lee  
**Venue / Journal**: NeurIPS 2017  
**Year**: 2017  
**arXiv ID**: 1705.07874  
**PDF**: https://arxiv.org/pdf/1705.07874.pdf  

---

## 1. 目標解決問題 (Problem)

現有的模型解釋方法（LIME、梯度法等）各自有不同的設計假設，難以比較且缺乏理論保證。
SHAP 提出一個統一框架，基於賽局理論的 Shapley value，給予每個 token 在預測中的「公平貢獻量」。
對情緒分析而言，SHAP 能精確量化每個詞對「正面/負面」機率的貢獻，找出真正的 key strings。

## 2. 方法 (Approach)

SHAP（SHapley Additive exPlanations）核心概念：

- 將每個 feature（token）的重要性定義為其在所有可能子集組合中的**平均邊際貢獻**
- 公式：$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$
- 對 BERT 等深度模型可使用 **KernelSHAP**（基於 LIME 框架近似 Shapley value）或 **DeepSHAP**（利用反向傳播）
- 輸出：每個 token 的 SHAP 值，正值 → 推向正面，負值 → 推向負面

這直接對應使用者想法中的「逐一加入 token 觀察分類結果變化」。

## 3. 結果 (Results)

- SHAP 統一了 LIME、DeepLIFT、整合梯度等 6 種現有方法，證明其為唯一同時滿足局部準確性、缺失性、一致性三個公理的解法
- 在情緒分析任務上，SHAP 識別出的重要詞與人類標注的情緒詞高度吻合
- 比 LIME 更具理論保證，但計算成本更高

## 4. 與本專案的關聯 (Relevance to Our Project)

可在我們的 BERT 分類器訓練完成後，用 `shap` 套件（`shap.Explainer`）對每個預測生成 token-level 重要性分數。
正 SHAP 值的 token 即為「正面情緒 key strings」，負值為「負面情緒 key strings」。
可從 2000 筆訓練資料中統計出高頻重要 token，作為特徵詞典。
