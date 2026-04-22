# ERASER: A Benchmark to Evaluate Rationalized NLP Models

**Authors**: Jay DeYoung, Sarthak Jain, Nazneen Fatema Rajani, Eric Lehman, Caiming Xiong, Richard Socher, Byron C. Wallace  
**Venue / Journal**: ACL 2020  
**Year**: 2020  
**arXiv ID**: 1911.03429  
**PDF**: https://arxiv.org/pdf/1911.03429.pdf  
**Website**: https://www.eraserbenchmark.com/  

---

## 1. 目標解決問題 (Problem)

NLP 解釋性研究缺乏統一的評估標準：各論文用不同資料集、不同指標評估模型找到的「rationale」（理由/關鍵片段），
導致進展難以追蹤與比較。ERASER 建立了一個標準化的 benchmark，包含人工標注的 rationale，
讓「找出 key string 的方法」能被客觀評估。

## 2. 方法 (Approach)

ERASER 提供兩類 rationale 評估指標：

**1. 與人類標注的對齊程度（Plausibility）**
- Token-level F1：預測重要 token 與人工標注的重疊率
- IOU F1：片段層級的重疊

**2. 忠實度（Faithfulness）**—更關鍵
- **Comprehensiveness**：只保留 rationale token 時，分類準確率應維持高水準
  $\Delta = f(\text{full}) - f(\text{full} \setminus \text{rationale})$
- **Sufficiency**：只用 rationale token 就能達到完整輸入的同等效果
  $\Delta = f(\text{full}) - f(\text{rationale only})$

資料集涵蓋：情緒分析（Movie Reviews）、問答（MultiRC）、NLI（e-SNLI）等 7 個任務。

## 3. 結果 (Results)

- 現有的解釋方法（梯度法、LIME）在 Plausibility 上表現一般，在 Faithfulness 上普遍不佳
- Extractive Rationale 模型（直接預測 rationale mask）在 Sufficiency 上明顯優於事後解釋方法
- 情緒分析任務（Movie Reviews）的人工 rationale 標注精確率約 80%，是最適合驗證 key string 的資料集

## 4. 與本專案的關聯 (Relevance to Our Project)

ERASER 的兩個指標（Comprehensiveness + Sufficiency）正是評估我們 **key string 品質**的標準方法：
- **Sufficiency 測試**：只用找到的 key strings 重新分類，看準確率是否維持
- **Comprehensiveness 測試**：移除 key strings 後，分類準確率應大幅下降

建議在找到 key strings 後，用這兩個指標驗證它們真的是情緒分類的充分且必要條件。
可直接使用 ERASER 的 Movie Reviews 子集驗證我們方法的泛化能力。
