# Parameter-Efficient Transfer Learning for NLP

**Authors**: Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly
**Venue / Journal**: ICML 2019
**Year**: 2019
**arXiv ID**: 1902.00751
**PDF**: https://arxiv.org/pdf/1902.00751.pdf

---

## 1. 目標解決問題 (Problem)

對每個下游任務都 full fine-tuning 整個 BERT 模型並儲存一份獨立副本，在多任務場景下既昂貴又難以擴展。本論文提出 Adapter 模組：在 BERT 的每個 Transformer 層中插入小型可訓練模組，fine-tuning 時只訓練 adapter 參數，原始 BERT 權重完全凍結，實現高度的參數共享與任務隔離。

## 2. 方法 (Approach)

Adapter 模組由兩個 feedforward 投影層（down-projection → non-linearity → up-projection）與一個殘差連接組成：輸入先投影至低維瓶頸空間（bottleneck size *d_b*，通常 8–256），再投回原始維度，最後加上殘差。每個 Transformer 層插入兩個 Adapter：一個在 multi-head attention 之後，另一個在 FFN 之後。只訓練 Adapter 參數與 layer normalization 參數（約佔 BERT-base 總參數的 0.5%–8.6%），其餘權重凍結。在 GLUE benchmark 的 26 個文本分類任務上系統性評估。

## 3. 結果 (Results)

Adapter 微調在 GLUE 多任務 benchmark 上達到 full fine-tuning **0.4% 以內**的性能，每個任務僅新增約 **3.6% 的額外參數**（相較於每任務訓練 100% 參數的 full fine-tuning）。在 BERT-based 的 26 個文本分類任務（含 SST-2 情感分析）中幾乎一致達到接近 full fine-tuning 的準確率，且多任務間的參數完全可複用。

## 4. 與本專案的關聯 (Relevance to Our Project)

Adapter fine-tuning 是本專案應對小資料的另一策略：固定絕大多數 BERT 參數，僅訓練極少的 adapter 參數，有效降低 2,000 筆資料場景下的過擬合風險。可透過 HuggingFace `adapters` 套件（原 AdapterHub）輕鬆整合，支援 RoBERTa、DeBERTa 等多種 backbone。Adapter 與 LoRA 的比較研究也是超參數搜索的有趣方向：adapter 有推理額外延遲但設計更靈活，LoRA 無延遲但需在層級選擇上更謹慎。
