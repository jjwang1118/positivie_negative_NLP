# Idea Notes — Key String Extraction for Sentiment Classification

## 核心想法

```
原始文字
  → BERT frozen 向量化
  → 逐一加入 token（Greedy Token Addition）
  → 觀察分類機率變化 → 找出 key strings
  → 以 key strings 作為分類判斷依據
```

## 相關論文

| # | 檔案 | 論文 | 與本想法的關係 |
|---|------|------|--------------|
| 1 | [LIME_Note.md](LIME_Note.md) | LIME (KDD 2016) | 最接近的現有方法，擾動 token 找重要性 |
| 2 | [SHAP_Note.md](SHAP_Note.md) | SHAP (NeurIPS 2017) | 理論最嚴格的 token 重要性量化，有 Python 套件 |
| 3 | [InputReduction_Note.md](InputReduction_Note.md) | Input Reduction (EMNLP 2018) | 揭示貪婪移除/加入的病態問題，需注意 |
| 4 | [ERASER_Note.md](ERASER_Note.md) | ERASER (ACL 2020) | 提供評估 key string 品質的標準指標 |
| 5 | [PromptSurvey_Note.md](PromptSurvey_Note.md) | Prompting Survey (CSUR 2022) | 不微調的替代分類方案，可整合 key strings |

## 實作建議順序

1. **SHAP** → 快速驗證想法可行性（安裝 `shap`，對 BERT 輸出做 KernelSHAP）
2. **LIME** → 視覺化每個樣本的重要 token（`lime.lime_text`）
3. **ERASER 指標** → 用 Sufficiency / Comprehensiveness 評估找到的 key strings 品質
4. **Input Reduction 警告** → 人工審查 key strings 是否有語意意義
5. **Prompting** → 將 key strings 嵌入 prompt，做無微調的最終分類器
