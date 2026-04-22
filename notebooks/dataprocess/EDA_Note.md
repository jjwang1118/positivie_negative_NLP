# EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks

**Authors**: Jason Wei, Kai Zou  
**Venue / Journal**: EMNLP-IJCNLP 2019  
**Year**: 2019  
**arXiv ID**: 1901.11196  
**PDF**: https://arxiv.org/pdf/1901.11196.pdf

---

## 1. 目標解決問題 (Problem)

文本分類任務常因訓練資料不足而導致模型表現不佳，尤其在小型資料集場景下尤為突出。本文探討如何以極低成本的文字操作方式擴充訓練集，無需任何外部資料或複雜模型。對於我們僅有 2,000 筆標籤樣本的二元情緒分類任務，資料稀缺問題正是核心挑戰，EDA 提供了一種即插即用的解決方案。

## 2. 方法 (Approach)

EDA 提出四種簡單但有效的文本增強操作，統稱為「簡易資料增強」：

1. **同義詞替換（Synonym Replacement, SR）**：隨機選取 *n* 個非停用詞，以 WordNet 同義詞替換。
2. **隨機插入（Random Insertion, RI）**：隨機選取一個非停用詞，找其同義詞插入句中任意位置。
3. **隨機交換（Random Swap, RS）**：隨機選兩個詞交換位置，重複 *n* 次。
4. **隨機刪除（Random Deletion, RD）**：以概率 *p* 隨機刪除每個詞。

每筆原始訓練樣本可擴充為數筆不同表面形式但語義相同的新樣本，並保留原始標籤。

## 3. 結果 (Results)

在五個文本分類資料集（SST-2、CR、SUBJ、TREC、PC）上進行評估：

- 使用 EDA 以 **50% 訓練資料** 達到與使用 **100% 原始資料** 相當的準確率。
- 在小資料集（500 樣本）上平均提升約 **+0.8%～+3.0% 準確率**。
- 搭配 CNN 與 RNN 模型均有穩定提升，對小資料集效果尤為顯著。

## 4. 與本專案的關聯 (Relevance to Our Project)

EDA 的四種操作可直接套用至本專案的 2,000 筆英文情緒訓練樣本，每筆樣本可生成 4–8 筆增強版本，使訓練集擴充至 8,000–16,000 筆。其中同義詞替換與隨機刪除對情緒分類的語義保留效果最佳；建議使用 `nlpaug` 或 `textaugment` 套件快速實作，並在訓練前對增強樣本進行去重過濾。
