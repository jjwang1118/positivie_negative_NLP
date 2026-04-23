# Unsupervised Data Augmentation for Consistency Training

**Authors**: Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, Quoc V. Le  
**Venue / Journal**: NeurIPS 2020  
**Year**: 2019 (arXiv) / 2020 (conference)  
**arXiv ID**: 1904.12848  
**PDF**: https://arxiv.org/pdf/1904.12848.pdf  
**Code**: https://github.com/google-research/uda

---

## 1. 目標解決問題 (Problem)

半監督學習（SSL）在標注資料稀缺時展現出巨大潛力，其核心假設為：模型對同一輸入的不同帶噪版本，預測結果應保持一致（一致性訓練，Consistency Training）。然而，過去的 SSL 方法多採用簡單的隨機噪聲（如 Gaussian noise、dropout），對 NLP 任務效果有限。本文的核心主張是：**噪聲的品質**（而非數量）才是一致性訓練成敗的關鍵，應以高品質的資料增強替代簡單噪聲，從而在無需額外標注的情況下大幅提升半監督性能。

本專案同樣面臨類似挑戰：2,000 筆有標籤樣本 + 11,000 筆無標籤測試集，UDA 框架可直接對應此設定。

## 2. 方法 (Approach)

### 2.1 整體框架：一致性訓練

UDA 的訓練目標分為兩部分：

1. **監督損失**（Supervised Loss）：對有標籤樣本 $(x, y)$ 計算標準交叉熵損失。
2. **一致性損失**（Unsupervised Consistency Loss）：對無標籤樣本 $\hat{x}$，使用資料增強生成其噪聲版本 $\tilde{x}$，要求模型對 $\hat{x}$ 和 $\tilde{x}$ 的預測分布盡量接近，以 KL 散度衡量：

$$\mathcal{L}_{unsup} = \mathbb{E}[D_{KL}(p_\theta(\hat{y}|\hat{x}) \| p_\theta(\hat{y}|\tilde{x}))]$$

總損失為兩者加權和，無標籤一致性損失的梯度僅反向傳播至增強版本 $\tilde{x}$ 側（原始版本 $\hat{x}$ 的預測作為「軟目標」使用，停止梯度）。

### 2.2 NLP 噪聲策略

**（a）回譯（Back-Translation）**
- 將英文句子翻譯至另一語言（如法語）後再譯回英文，生成語義等價但表面形式不同的增強樣本。
- 使用 beam search 或隨機取樣（temperature sampling）控制多樣性與品質的平衡。
- 高 temperature（如 0.9）產生更多樣的改寫；低 temperature 接近確定性翻譯，品質更高。

**（b）TF-IDF 詞替換（TF-IDF Word Replacement）**
- 計算訓練集中每個詞的 TF-IDF 分數，低 TF-IDF 詞（即通用詞、功能詞）被視為對分類標籤影響較小的詞。
- 以一定機率隨機替換低 TF-IDF 詞，保留高 TF-IDF 的關鍵情緒詞不被替換，從而在不破壞情緒語義的前提下引入表面噪聲。

### 2.3 訓練穩定技術

**（a）Training Signal Annealing（TSA，訓練信號退火）**
- 問題動機：有標籤資料遠少於無標籤資料時，模型容易對有標籤樣本過擬合（over-fitting）。
- 解法：動態調整監督損失的閾值 $\eta_t$。訓練初期 $\eta_t$ 較低，僅使用模型尚未確定的有標籤樣本計算監督損失；隨訓練進行，$\eta_t$ 逐漸增大，讓更多有標籤樣本參與訓練。
- 退火策略分為三種：linear-schedule、log-schedule、exp-schedule，分別以不同速率從 $\frac{1}{K}$ 增長至 1（$K$ 為類別數）。

**（b）預測銳化（Sharpening / Confidence-based Masking）**
- 對無標籤樣本，僅在模型對原始預測有足夠信心時（最大預測機率超過閾值）才計算一致性損失，濾除模型尚不確定的樣本，避免強化錯誤的偽標籤。

## 3. 結果 (Results)

### NLP 任務（文本分類）

| 資料集 | 標注樣本數 | UDA 錯誤率 | 前 SOTA | 備註 |
|--------|-----------|-----------|---------|------|
| IMDb（情緒分析） | 20 | **4.20%** | 4.32%（25,000 標注） | 極少標注超越全量監督學習 |
| IMDb | 200 | 3.54% | — | 持續改善 |
| Amazon Review | 20 | — | — | 多領域情緒分析 |
| Yelp Review | 20 | — | — | — |
| DBpedia | 20 | — | — | — |
| AG News | 20 | — | — | 新聞分類 |

關鍵結論：
- 在 IMDb 上，**僅用 20 筆標注**的 UDA 模型，效果超越使用 **25,000 筆全量標注**的監督學習 SOTA（錯誤率 4.20% vs 4.32%）。
- 回譯（Back-Translation）在 NLP 任務中是最有效的噪聲策略，顯著優於隨機刪除或替換等傳統做法。
- 與 BERT 微調結合後效果進一步提升，展示出強大的遷移學習協同性。

### 消融實驗（Ablation）

- 使用回譯 vs. 不使用增強（僅 dropout）：錯誤率從 11.8% → 4.20%（IMDb，20 標注）。
- TSA vs. 無 TSA：TSA 在低標注設定下對防止過擬合至關重要。
- TF-IDF 詞替換 vs. 隨機詞替換：TF-IDF 策略保留情緒關鍵詞，效果優於隨機替換。

## 4. 與本專案的關聯 (Relevance to Our Project)

UDA 是目前最契合本專案設定的半監督框架之一，可直接套用於「2,000 筆有標籤 + 11,000 筆無標籤測試集」的場景：

1. **框架直接適用**：UDA 的核心設計就是利用無標籤資料做一致性訓練，本專案的 11,000 筆測試集文本即可作為無標籤來源。

2. **回答「偽標籤正確性驗證」問題**：
   - UDA 並不直接給無標籤樣本賦予硬標籤（hard pseudo-label），而是以模型在原始樣本上的**軟預測分布**（soft target）作為一致性目標，從根本上規避偽標籤噪聲問題。
   - 信心閾值（confidence masking）進一步確保只有模型高確信的無標籤樣本才貢獻一致性損失，是一種內建的偽標籤品質控制機制。
   - 這直接解答了「如何驗證偽標籤正確性」的問題：UDA 的答案是**不用硬偽標籤**，改用軟目標 + 信心篩選。

3. **TF-IDF 詞替換實作成本低**：不需要外部翻譯 API，可用 `sklearn.feature_extraction.text.TfidfVectorizer` 計算詞重要性後，搭配 `nlpaug` 或自行實作詞替換邏輯。

4. **與 BERT 微調結合**：本專案已在使用 BERT 系列模型，UDA 的一致性損失可作為額外訓練信號加入現有 fine-tuning 流程，不需更換模型架構。

5. **建議實作優先順序**：
   - 優先試驗 TF-IDF 詞替換（實作快、無 API 成本）。
   - 再加入回譯（英→法→英，使用 `Helsinki-NLP/opus-mt` 或 Google Translate API）。
   - 最後加入 TSA 機制，防止 2,000 筆有標籤樣本被過度擬合。
