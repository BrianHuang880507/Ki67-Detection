# Exp1 封存說明

- 封存日期：2026-07-28
- 原始位置：`immunity/`
- 實驗名稱：B4 p6 morphology-based IDO response prediction
- 封存內容：程式、設定、文件、模型、表格、圖與執行報告
- 既有輸出：22 個檔案，約 3.73 MB
- 原始報告：`outputs/b4_p6/REPORT.md`

## 重跑

在專案根目錄執行：

```powershell
conda run --no-capture-output -n ki67dtc python -m immunity.exp1.run_experiment --config immunity/exp1/configs/b4_p6.yaml
```

重跑結果只會寫入 `immunity/exp1/outputs/b4_p6/`，不會覆蓋新版
`immunity/outputs/`。

既有 `final_models.csv` 與 `run_metadata.json` 內的絕對路徑保留第一次執行
當時的原始紀錄，因此仍可能顯示舊的 `immunity/outputs/` 位置；實際封存
模型位於 `exp1/outputs/b4_p6/models/`。

## 解讀邊界

Exp1 以每張影像的背景校正 `IDO_score` 為連續預測目標，並將三條
IDO dose-response AUC 當成下游摘要。它沒有 Well 對應資訊，也沒有
PBMC／T-cell functional assay，因此結果不能稱為已驗證的免疫抑制能力。
