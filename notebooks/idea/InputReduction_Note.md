# Pathologies of Neural Models Make Interpretations Difficult (Input Reduction)

**Authors**: Shi Feng, Eric Wallace, Alvin Grissom II, Mohit Iyyer, Pedro Rodriguez, Jordan Boyd-Graber  
**Venue / Journal**: EMNLP 2018  
**Year**: 2018  
**arXiv ID**: 1804.07781  
**PDF**: https://arxiv.org/pdf/1804.07781.pdf  

---

## 1. 目標解決問題 (Problem)

此論文研究 NLP 神經模型的一個病態行為：透過「Input Reduction」（迭代移除最不重要的 token），
發現模型在只剩下幾個對人類毫無意義的 token 時仍能以高信心做出正確預測。
這揭示了梯度/擾動法找出的「重要 token」不一定真正有語意意義，
與使用者想法中「找出 key string」的合理性問題直接相關。

## 2. 方法 (Approach)

**Input Reduction 演算法**（貪婪迭代移除）：

1. 計算每個 token 的重要性分數（使用梯度 × 輸入大小）
2. 移除**最不重要**的 token
3. 重複直到只剩 1 個 token，記錄每步的模型信心
4. 觀察：移除大量 token 後模型信心幾乎不變，說明模型依賴少數 token 即可分類

反向操作即為使用者的**Greedy Token Addition**（逐步加入最重要 token）。

論文同時提出 **Fine-tune with Entropy Regularization** 來緩解此病態現象，
讓模型在 reduced input 上輸出高 entropy（不確定），使解釋更可靠。

## 3. 結果 (Results)

- 在 VQA、閱讀理解、文字分類任務上，平均只需保留原句 20% 以下的 token，模型信心仍 >90%
- 加入 entropy regularization 後，減少了病態行為，解釋方法的人類評分顯著提升
- 揭示了「最重要 token」可能是停用詞或位置 token，而非真正的語意關鍵詞

## 4. 與本專案的關聯 (Relevance to Our Project)

使用者想法的核心風險就是此論文所指出的病態：找到的「key string」可能是模型捷徑（shortcut），
而非真正的情緒語意特徵。建議在我們的 Greedy Token Selection 後：
1. 用人類評估驗證找到的 key string 是否有語意意義
2. 加入 entropy regularization 訓練，讓模型更依賴真正的語意 token
3. 與 SHAP 結果交叉比對，排除統計偽相關
