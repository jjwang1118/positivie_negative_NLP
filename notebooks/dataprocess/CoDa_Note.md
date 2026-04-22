# CoDa: Constrained Generation based Data Augmentation for Low-Resource NLP

**Authors**: Chandra Kiran Reddy Evuru, Sreyan Ghosh, Sonal Kumar, Ramaneswaran S, Utkarsh Tyagi, Dinesh Manocha  
**Venue / Journal**: NAACL 2024 Findings  
**Year**: 2024  
**arXiv ID**: 2404.00415  
**PDF**: https://arxiv.org/pdf/2404.00415.pdf

---

## 1. 目標解決問題 (Problem)

低資源 NLP 任務中，如何在不微調 LLM、不使用複雜解碼演算法的前提下，生成高品質且多樣化的增強資料？現有方法要麼需要模型微調（導致小資料集過擬合），要麼生成的樣本與訓練資料分布差異過大。本文針對資料稀缺場景提出了一種訓練無關（training-free）且使用者可顯式控制的增強框架，與本專案的 2,000 樣本訓練場景高度吻合。

## 2. 方法 (Approach)

CoDa（Constrained Generation based Data Augmentation）的核心流程分三步驟：

1. **約束提取（Constraint Extraction）**：從每筆訓練樣本中自動提取一組簡單約束（如關鍵詞、主題詞、實體等）。
2. **約束語言化（Verbalization）**：將提取的約束轉換成自然語言 prompt 格式，結合類別標籤一起輸入 instruction-following LLM（如 GPT-4 或 Flan-T5）。
3. **約束引導生成（Constrained Generation）**：LLM 依據 prompt 中的約束生成新穎且多樣化的訓練樣本，確保生成樣本滿足任務相關的關鍵語義約束。

此框架提供使用者對增強過程的**顯式控制**（explicit control），且無需任何模型微調。

## 3. 結果 (Results)

在 11 個資料集、3 種任務（情緒分析、自然語言推理、主題分類）、3 種低資源設定下驗證：

- CoDa 相較於所有基線方法（EDA、CBERT、LAMBADA、GPT3Mix 等）均有提升。
- **整體改善幅度：0.12%–7.19%**（依任務與資料集規模而異）。
- 在 SST-2 情緒分類任務的低資源設定下，F1 提升達 **+3.4%**。
- 在各項自動化與人工評估中，CoDa 生成樣本的品質與多樣性均優於基線。

## 4. 與本專案的關聯 (Relevance to Our Project)

CoDa 對本專案的核心優勢在於其**顯式約束控制**與**無需微調**：可從 2,000 筆訓練樣本中提取情緒關鍵詞（如 "excellent"、"terrible"）作為約束，引導 LLM 生成同一情緒極性的新樣本。框架開源（GitHub: Sreyan88/CoDa），可直接套用於 GPT-4o 或 Gemini API。建議對正面/負面各 1,000 筆樣本各生成 2–3 倍增強資料，以低成本大幅擴充訓練集。
