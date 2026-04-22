# New Directions in Text Classification Research: Maximizing The Performance of Sentiment Classification from Limited Data

**Authors**: Surya Agustian, Muhammad Irfan Syah, Nurul Fatiara, Rahmad Abdillah  
**Venue / Journal**: arXiv preprint  
**Year**: 2024  
**arXiv ID**: 2407.05627  
**PDF**: https://arxiv.org/pdf/2407.05627.pdf

---

## 1. 目標解決問題 (Problem)

在訓練資料極度有限（300–600 樣本）的場景下，情緒分類模型往往因過擬合而在測試集表現不佳。本文探討如何透過外部資料聚合（aggregation）與資料增強（augmentation）等策略，從有限資料中最大化情緒分類性能。此問題設定與本專案（2,000 筆訓練、11,000 筆未標籤測試）的核心挑戰直接吻合。

## 2. 方法 (Approach)

本文採用競賽式研究框架，提供基線與優化方法的對比：

1. **外部資料聚合（External Data Aggregation）**：引入相關主題的外部情緒資料集（如 COVID 疫苗接種情緒資料、開放主題資料集）作為額外訓練資料。
2. **資料增強（Data Augmentation）**：對有限訓練集應用多種文本增強技術，包括同義詞替換、回譯（back-translation）等。
3. **基線評估（SVM Baseline）**：使用 SVM 作為廣泛報告的傳統機器學習最佳方法，提供公平比較基準。

本文重點在量化分析增強策略帶來的提升，為少樣本情緒分類建立基準分數系統（benchmark scoring）。

## 3. 結果 (Results)

在印尼語情緒分類競賽資料集（正面/負面/中立三分類）上：

- **基線方法（SVM，無優化）**：F1-score = **40.83%**。
- **優化方法（SVM + 資料增強 + 外部資料聚合）**：F1-score = **51.28%**。
- 增強與聚合策略帶來約 **+10.45 個百分點**的 F1 提升（相對提升約 25.6%）。
- 結果驗證即使是傳統 ML 模型，透過系統性資料增強也能顯著改善少樣本場景性能。

## 4. 與本專案的關聯 (Relevance to Our Project)

本文直接展示了在少樣本情緒分類中，資料策略（而非模型結構）才是關鍵瓶頸。對本專案的啟示：（1）系統性比較 EDA、回譯、外部資料聚合等增強策略對二元分類 F1 的貢獻；（2）聚合 SST-2、IMDB 等公開正/負情緒資料集作為額外訓練資料；（3）建立增強前後 F1 比較的消融實驗（ablation study）框架，量化各增強策略的實際貢獻。
