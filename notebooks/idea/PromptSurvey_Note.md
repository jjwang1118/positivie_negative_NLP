# Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in NLP

**Authors**: Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, Graham Neubig  
**Venue / Journal**: ACM Computing Surveys (CSUR) 2022 / arXiv 2021  
**Year**: 2021  
**arXiv ID**: 2107.13586  
**PDF**: https://arxiv.org/pdf/2107.13586.pdf  

---

## 1. 目標解決問題 (Problem)

傳統 NLP 流程需要針對每個任務收集大量標注資料並微調模型。
Prompting 範式提供另一條路：透過設計文字模板（prompt），
讓預訓練模型在不更新參數的情況下執行分類任務（zero-shot / few-shot）。
這與我們「不微調」的需求直接相關，也是「key string 作為判斷依據」想法的理論根基。

## 2. 方法 (Approach)

Prompting 的核心框架：

```
原始輸入 x → 加入模板 → 提示字串 x'  
x' = "The movie was [MASK]. It was [positive/negative]."  
→ 模型填充 [MASK]，映射到 label  
```

主要類型：
- **Cloze Prompt**（完形填空）：`[X]. The sentiment is [MASK].`，適合 BERT 類 MLM 模型
- **Prefix Prompt**（前綴）：`Classify: [X]`，適合 GPT 類模型
- **Discrete vs. Continuous Prompt**：手工設計 vs. 訓練出的 soft prompt token

對情緒分析：直接用 BERT + Cloze Prompt，無需任何標注資料即可分類。

## 3. 結果 (Results)

- Zero-shot prompting 在 SST-2 上達到 ~85–88%（BERT-large），接近早期微調基線
- Few-shot prompting（8–32 examples）可達到 ~91–93%，逼近 full fine-tuning
- Prompt-based few-shot 在資料量極少（<100）時明顯優於傳統微調

## 4. 與本專案的關聯 (Relevance to Our Project)

Prompting 是我們「不使用微調/LoRA」情況下的最強替代方案之一：
1. **零樣本版**：直接用 `"[text]. The sentiment is [MASK]."` → BERT 填充 "positive"/"negative"
2. **Key string 整合**：將我們找到的 key strings 嵌入 prompt：
   `"Words like 'excellent' and 'terrible' appear. The overall sentiment is [MASK]."`
3. 搭配 SHAP 分析：先用 SHAP 找 key strings，再設計包含這些詞的 prompt，
   形成「可解釋 + 無微調」的完整 pipeline。
