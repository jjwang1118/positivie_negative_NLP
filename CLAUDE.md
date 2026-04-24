# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NLP 情緒分析實驗，判斷文本為正面或負面情緒的二元分類任務。
(**Note: 此專案不能使用LLM模型，除了BERT類型的模型以外，其他如GPT、Claude等生成式模型不適用)

- Label `1` = positive
- Label `0` = negative

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Commands

| 用途 | 指令 |
|------|------|
| 執行所有測試 | `pytest` |
| 執行單一測試 | `pytest tests/test_example.py::test_name` |
| 測試覆蓋率 | `pytest --cov=src` |
| 啟動 Jupyter | `jupyter notebook` |

## Structure

```
src/          # 核心模組（資料處理、模型、評估）
tests/        # pytest 測試
notebooks/    # 論文相關資訊和整理
data/         # 資料集（raw/ processed/）
results/      # 實驗輸出、圖表、metrics
exp/          # 以下會存放各實驗編號資料夾，檔案包含實驗流程設計與實驗腳本
```

## Architecture

- `src/` 放可重用的 pipeline 元件（preprocess、model、evaluate）
- `notebooks/` 做探索性分析，成熟後的邏輯移入 `src/`
- `results/` 如果要推上github，個個實驗只需要推最外層和內層的空資料夾以及```yaml```、```json```、```md```、```csv```等文字檔和```png```、```jpg```等照片檔，避免推上大檔案（如模型 checkpoint）
## Agents

各 agent 定義在對應目錄的 `SKILL.md`，建議依下列順序使用：

```
structure-manager → data-analyzer → paper-collector
      → model-establisher → trainer → predictor
```

| Agent | 目錄 | 職責 |
|-------|------|------|
| `structure-manager` | [.claude/agents/structure-manager/](.claude/agents/structure-manager/SKILL.md) | 建立與管理專案目錄、生成 `src/config.py` 路徑設定 |
| `data-analyzer` | [.claude/agents/data-analyzer/](.claude/agents/data-analyzer/SKILL.md) | 分析 train/test 標籤分布，輸出長條圖至 `results/figures/` |
| `paper-collector` | [.claude/agents/paper-collector/](.claude/agents/paper-collector/SKILL.md) | 搜尋並評估相關論文，整合 deep-research + fact-checker，輸出 Tier 分級推薦表 |
| `model-establisher` | [.claude/agents/model-establisher/](.claude/agents/model-establisher/SKILL.md) | 依 paper-collector 推薦建構 `src/model.py` 與 MODEL\_REGISTRY |
| `trainer` | [.claude/agents/trainer/](.claude/agents/trainer/SKILL.md) | 詢問超參數、建立完整訓練流程，儲存 checkpoint 與 log |
| `predictor` | [.claude/agents/predictor/](.claude/agents/predictor/SKILL.md) | 載入訓練好的模型，輸出 CSV/JSON 預測結果與 confidence score |

### 輔助 Skills（由 paper-collector 呼叫）

| Skill | 目錄 | 用途 |
|-------|------|------|
| `deep-research` | [.claude/agents/deep-research/](.claude/agents/deep-research/SKILL.md) | Phase 1：廣泛搜集論文與綜合分析 |
| `fact-checker` | [.claude/agents/fact-checker/](.claude/agents/fact-checker/SKILL.md) | Phase 2：驗證論文數據與聲明可信度 |
