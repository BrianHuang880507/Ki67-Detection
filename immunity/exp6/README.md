# Exp6：劑量關聯性 ＋ IDO 亮暗細胞影像庫

回答生醫所同仁的兩個問題：

1. **IDO 對 IFN-γ／TNF-α 的關聯性，以及外觀特徵對 IFN-γ／TNF-α 的關聯性**（長條圖取前 10 名）。
2. **把 IDO 亮的細胞與不亮的細胞分別切出來看**。

## 執行

在專案根目錄：

```powershell
conda run --no-capture-output -n ki67dtc python -m immunity.exp6.run_experiment
```

約 40 秒跑完。常用參數：

| 參數 | 預設 | 說明 |
|---|---|---|
| `--data-root` | 目前工作目錄 | Exp3／Exp4 產物所在的專案根目錄 |
| `--output-root` | `<data-root>/immunity/outputs/exp6` | 輸出目錄 |
| `--top-n` | 10 | 長條圖取前幾名 |
| `--decile-pct` | 10 | 亮／暗各取上下多少百分比 |
| `--tiles-per-group` | 30 | 影像庫每組放幾顆細胞 |
| `--skip-gallery` | 關 | 只跑關聯分析，不讀原始影像（約 3 秒） |
| `--skip-cell-crops` | 關 | 不輸出逐顆細胞的四格 PNG |

測試：

```powershell
conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp6.py -q
```

## 輸入（唯讀，不覆寫）

| 檔案 | 內容 |
|---|---|
| `immunity/outputs/exp4/cell_level_rui49.csv` | 49 個 Rui2025 風格特徵，全部由**相位差（PC）通道**計算 |
| `immunity/outputs/exp3/flatfield_2026-08-19/ido_flatfield.csv` | flat-field 校正後的 `IDO_score_ff` 與校正前的 `IDO_score_scalar` |
| `immunity/outputs/exp3/current_inputs_2026-08-19/data_manifest.csv` | 693 張影像的 donor／passage／條件／劑量／影像路徑 |
| `immunity/outputs/exp3/feature_cache/masks/<group>/<image_key>.npz` | whole-cell 與 nucleus label mask（只有影像庫會用到） |

規模：23,012 顆細胞、693 張影像、3 donor × 3 passage × 8 條件。

## 方法上的三個關鍵決定

### 1. 分析單位是「一張影像」，不是「一顆細胞」

劑量施加在整個孔上，同一張影像內的細胞共享同一個劑量。若把 23,012 顆細胞
當成獨立觀測，p 值會被灌到沒有意義。所有相關係數都先把每張影像的細胞取中位數，
`n` 因此是影像數（350／521），不是細胞數。

### 2. 劑量軸只切兩條，因為資料不是完整 factorial design

| 軸 | 切片 | 劑量 | n |
|---|---|---|---:|
| IFN | TNF-α = 0 | IFN-γ 0／25／50／100 | 350 張 |
| TNF | IFN-γ = 0 或 25 | TNF-α 0／25／50 | 521 張 |

TNF 軸的相關係數在 IFN 區塊內計算（partial Spearman：先在區塊內取 rank、
置中，再求 Pearson r），避免拿 IFN 的差異冒充 TNF 的效果。TNF 軸另外拆成
`IFN-γ = 0` 與 `IFN-γ = 25` 兩列單獨報告。

### 3. 排除的欄位

49 個特徵中排除 8 個，剩 41 個進入排名：

- **6 個純位置欄位**（`Center_X/Y`、`BoundingBox` 四個座標）。IDO 通道原本有
  左右照明梯度，位置欄位描述的是細胞在視野中的座標而不是外觀。
- **`MinIntensity`**：rolling-ball 背景扣除後恆為 0。
- **`Orientation`**：角度是循環量，線性相關係數不可解釋。

（`MinIntensityEdge` 在影像中位數層級也是常數，會自動得到 `n/a`，實際可用 40 個。）

## 輸出

全部寫進 `immunity/outputs/exp6/`。

### 表格

| 檔案 | 內容 |
|---|---|
| `ido_dose_correlations.csv` | IDO（校正前後）對兩條劑量軸的 rho、p、灰階變化量、逐區塊同號數 |
| `ido_dose_response.csv` | 三條描述性劑量曲線的影像層級摘要 |
| `feature_dose_correlations.csv` | 41 個外觀特徵 × 2 條劑量軸的完整排名（含 BH FDR） |
| `feature_ido_correlations.csv` | 外觀特徵對 IDO 的關聯性，含「同一條件內」與「同一影像內」版本 |
| `image_level_dataset.csv` | 影像層級資料，一張影像一列 |
| `bright_dim_features_global.csv`、`bright_dim_features_matched.csv` | 亮／暗細胞的外觀特徵比較（Mann–Whitney U ＋ rank-biserial） |
| `bright_dim_selected_cells_*.csv` | 影像庫實際用到的細胞清單，含原始影像路徑 |

### 圖

| 檔案 | 內容 |
|---|---|
| `figures/fig01_ido_vs_dose.png` | IDO 三條劑量曲線 |
| `figures/fig02_top10_features_ifn.png` | 外觀特徵 vs. IFN-γ 前 10 名 |
| `figures/fig03_top10_features_tnf.png` | 外觀特徵 vs. TNF-α 前 10 名 |
| `figures/fig04_top10_features_combined.png` | 上面兩張並排，簡報用 |
| `figures/fig05_top10_features_vs_ido.png` | 外觀特徵 vs. IDO（全部影像 vs. 同一條件內） |
| `figures/fig06_top10_bright_vs_dim_matched.png` | 同一條件下亮／暗細胞的外觀差異 |
| `figures/fig_gallery_{global,matched}_{ido,phase,ido_segmented}.png` | 亮／暗細胞影像庫，共 6 張 |
| `cell_crops_global/*.png` | 逐顆細胞四格圖：phase 原圖／phase 分割／IDO 原圖／IDO 分割 |

### 兩種亮暗切法

| 切法 | 定義 | 回答什麼 |
|---|---|---|
| `global` | 全部細胞取 `IDO_score_ff` 上／下 10% | 最直觀的「亮 vs 不亮」，但亮的那群幾乎必然來自 IFN-γ 刺激組 |
| `matched` | 在每個 donor×passage×條件內取上／下 10% | 劑量被固定住，剩下的差異才是「同樣刺激下，為什麼有些細胞亮」 |

## 結果與限制

見執行後產生的 `immunity/outputs/exp6/REPORT.md`。

要點：本實驗**只做關聯性排名，沒有做 out-of-fold 預測**。要回答「形態能不能
預測 IDO」請看 Exp3／Exp4 的交叉驗證結果。只有 3 個 donor，693 張影像是技術
重複而非生物重複，所有結果定位為探索性描述。
