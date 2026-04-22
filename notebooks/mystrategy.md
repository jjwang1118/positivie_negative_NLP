# Key String 擷取與驗證工作流程

> **專案路徑**：`c:\Users\USER\Documents\資科\P_N`  
> **任務**：情緒分析二元分類（正面 / 負面），從訓練好的 BERT 分類器中擷取 key strings

---

## 依據論文

| # | 論文 | Venue | 筆記路徑 | 用途 |
|---|------|-------|----------|------|
| 1 | "Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME) | KDD 2016 | [notebooks/idea/LIME_Note.md](idea/LIME_Note.md) | 快速生成 token 重要性分數 |
| 2 | A Unified Approach to Interpreting Model Predictions (SHAP) | NeurIPS 2017 | [notebooks/idea/SHAP_Note.md](idea/SHAP_Note.md) | 具理論保證的 Shapley token 貢獻量 |
| 3 | Pathologies of Neural Models Make Interpretations Difficult (Input Reduction) | EMNLP 2018 | [notebooks/idea/InputReduction_Note.md](idea/InputReduction_Note.md) | 偵測模型捷徑（shortcut）與病態行為 |
| 4 | ERASER: A Benchmark to Evaluate Rationalized NLP Models | ACL 2020 | [notebooks/idea/ERASER_Note.md](idea/ERASER_Note.md) | 以 Sufficiency / Comprehensiveness 客觀評估 key strings |

---


