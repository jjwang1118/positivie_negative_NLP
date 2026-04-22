# AugGPT: Leveraging ChatGPT for Text Data Augmentation

**Authors**: Haixing Dai, Zhengliang Liu, Wenxiong Liao, Xiaoke Huang, Yihan Cao, Zihao Wu, Lin Zhao, Shaochen Xu, Wei Liu, et al.  
**Venue / Journal**: arXiv preprint  
**Year**: 2023  
**arXiv ID**: 2302.13007  
**PDF**: https://arxiv.org/pdf/2302.13007.pdf

---

## 1. 目標解決問題 (Problem)

當前文本資料增強方法存在兩大缺陷：一是生成樣本的標籤正確性（faithfulness）不足，二是生成樣本多樣性（diversity/compactness）有限，難以兼顧兩者。在少樣本學習場景（few-shot learning）中，目標域資料極度稀缺，如何利用大型語言模型生成高品質增強資料成為關鍵。這直接對應本專案以 2,000 筆資料訓練二元情緒分類器的需求。

## 2. 方法 (Approach)

AugGPT 基於 ChatGPT（GPT-3.5）提出以下增強流程：

1. 對訓練集中每個樣本，設計 prompt 要求 ChatGPT 將其改寫為多個「概念上相似但語義措辭不同」的版本。
2. 在 prompt 中明確指定類別標籤，確保生成樣本與原始標籤一致（faithfulness）。
3. 透過語義相似度指標控制生成多樣性（diversity），避免重複度過高的增強樣本。
4. 將增強樣本混入原始訓練集進行下游模型微調。

整個流程不需要微調 ChatGPT，是一種訓練無關（training-free）的增強方案。

## 3. 結果 (Results)

在多個少樣本文本分類任務（包含 SST-2、AGNews、DBPedia 等）上驗證：

- AugGPT 在 **Few-shot 設定**（每類 8–16 樣本）下，測試準確率普遍優於 EDA、CBERT、LAMBADA 等基線方法。
- 在 SST-2 二元情緒分類任務中，以 16 樣本訓練，AugGPT 達到 **85.6% 準確率**（vs. 無增強的 78.2%）。
- 增強樣本的標籤一致性超過 95%，且 t-SNE 可視化顯示增強分布覆蓋度更廣。

## 4. 與本專案的關聯 (Relevance to Our Project)

本文方法對本專案具有直接應用價值：可利用 ChatGPT API 或本地部署的開源 LLM（如 LLaMA-3）對 2,000 筆訓練樣本進行 prompt-based 改寫增強。每筆樣本建議生成 3–5 個版本，重點確保正面（label=1）與負面（label=0）各生成等比例增強樣本以維持類別平衡。此外，AugGPT 的多樣性控制策略可防止過度相似的增強樣本降低模型泛化性能。
