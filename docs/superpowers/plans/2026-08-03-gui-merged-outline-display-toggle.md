# GUI Merged Outlines Display Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓沒有 `*_seg.npy`、但保留有效 `*_merged_cp_outlines.txt` 的舊資料，也能在 GUI 切換 Paired Only 與全部 segmentation 物件。

**Architecture:** 安全解析 merged outlines 的兩行記錄，同時取得 paired polygons 與 all polygons。`MainWindow` 分開快取兩組 polygons：既有 paired cache 繼續負責 Paired Only、Cell_ID 與 highlight；新增 all cache 只供取消勾選後的 GUI 預覽。NPY／記憶體 regions 維持最高優先權，所有輸出路徑繼續固定使用 Paired Only。

**Tech Stack:** Python 3.10、NumPy、OpenCV、PyQt6、`unittest`

## Global Constraints

- 舊資料只要有有效 `*_merged_cp_outlines.txt`，即使沒有任何 `*_seg.npy`，開關仍須啟用。
- paired mode 只顯示同一筆記錄中 nucleus 與 cytoplasm 都存在的物件。
- all mode 顯示 merged 檔中所有有效 nucleus 與 cytoplasm，包括單側未配對物件。
- merged 檔以 nucleus／cytoplasm 兩行為一筆；`-1,-1` 代表該側不存在。
- NPY 或 pipeline 記憶體 `PairedOverlayData` 存在時，必須維持目前優先權與 renderer。
- all-polygons cache 只供 GUI 預覽，不得接入 CSV、XLSX、NPY、面積圖或自動 overlay PNG。
- `save_pipeline_fill_overlays()` 的 NPY 與 outline fallback 都必須維持 Paired Only。
- 不修改 merged outlines 格式，不寫回來源檔，不批次轉換舊資料，也不重新執行 pipeline。
- 空檔、無有效 polygon、奇數座標、非整數座標或讀檔錯誤不得讓 GUI 當機。
- 重要 public function 與 class 使用繁體中文 Google style docstring。
- Python 測試使用 `D:\Anaconda3\envs\ki67dtc\python.exe`。
- 只提交本功能列出的檔案；不得 stage 或 commit 工作樹既有的 `immunity/`、`CODEX_PLUGIN_SETUP.md`、`操作手冊.pptx` 等修改。

---

## File Structure

- `ki67dtc/app_pipeline.py`：集中 merged outline 行讀取與 polygon 安全解析；保留 paired loader，讓 all loader 回傳全部有效 polygons。
- `ki67dtc/gui/main_window.py`：快取 all polygons、判斷 outline-only availability，並依 GUI preference 選擇 paired／all polygons。
- `tests/test_paired_overlay.py`：驗證 merged loader 的 paired／all 差異、損壞資料降級與自動 PNG 輸出隔離。
- `tests/test_main_window_layout_contract.py`：驗證舊資料無 NPY 時可切換、Run／failed／stop 狀態，以及無效檔案降級。
- `Changelog.md`：記錄舊資料可直接使用 merged outlines 切換顯示。

### Task 1: 安全解析 paired 與 all merged outlines

**Files:**
- Modify: `ki67dtc/app_pipeline.py:226-300`
- Test: `tests/test_paired_overlay.py:1-215`

**Interfaces:**
- Produces: `_load_merged_outline_lines(merged_path: Path) -> list[str]`。
- Produces: `_parse_merged_outline_polygon(line: str) -> np.ndarray | None`。
- Preserves: `load_paired_merged_outlines(merged_path: Path) -> OverlayPolygons`，只回傳兩側都有效的記錄。
- Preserves and strengthens: `load_merged_outlines(merged_path: Path) -> OverlayPolygons`，回傳每筆記錄中所有有效單側 polygons，損壞行不拋例外。
- Preserves: `save_pipeline_fill_overlays()` outline fallback 只呼叫 `load_paired_merged_outlines()`。

- [ ] **Step 1: 新增 paired／all 與損壞記錄的失敗測試**

在 `tests/test_paired_overlay.py` 的 app pipeline imports 加入 `load_merged_outlines`，並新增：

```python
def test_merged_outline_loaders_separate_paired_and_all_records(self) -> None:
    with TemporaryDirectory() as temporary_directory:
        merged_path = Path(temporary_directory) / "sample_merged_cp_outlines.txt"
        merged_path.write_text(
            "1,1,5,1,5,5,1,5\n"
            "0,0,6,0,6,6,0,6\n"
            "8,8,12,8,12,12,8,12\n"
            "-1,-1\n"
            "-1,-1\n"
            "8,1,13,1,13,6,8,6\n"
            "not,a,polygon\n"
            "14,1,18,1,18,5,14,5\n"
            "20,20,24,20,24,24,20,24\n",
            encoding="utf-8",
        )

        paired = load_paired_merged_outlines(merged_path)
        all_polygons = load_merged_outlines(merged_path)

        self.assertEqual(len(paired.nuc_polygons), 1)
        self.assertEqual(len(paired.cyto_polygons), 1)
        self.assertEqual(len(all_polygons.nuc_polygons), 2)
        self.assertEqual(len(all_polygons.cyto_polygons), 3)
        np.testing.assert_array_equal(
            all_polygons.nuc_polygons[1],
            np.array([[8, 8], [12, 8], [12, 12], [8, 12]], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            all_polygons.cyto_polygons[-1],
            np.array([[14, 1], [18, 1], [18, 5], [14, 5]], dtype=np.int32),
        )


def test_merged_outline_loaders_return_empty_for_unreadable_file(self) -> None:
    with TemporaryDirectory() as temporary_directory:
        missing_path = Path(temporary_directory) / "missing_merged_cp_outlines.txt"

        paired = load_paired_merged_outlines(missing_path)
        all_polygons = load_merged_outlines(missing_path)

        self.assertEqual(paired.nuc_polygons, [])
        self.assertEqual(paired.cyto_polygons, [])
        self.assertEqual(all_polygons.nuc_polygons, [])
        self.assertEqual(all_polygons.cyto_polygons, [])
```

最後一個 dangling nucleus 行沒有對應 cytoplasm 行，必須忽略。第四筆的 nucleus
格式損壞，但有效 cytoplasm 必須保留在 all polygons，不能進入 paired polygons。

- [ ] **Step 2: 執行測試並確認現有 all loader 因損壞行失敗**

Run:

```powershell
& 'D:\Anaconda3\envs\ki67dtc\python.exe' -m unittest tests.test_paired_overlay.PairedOverlayTest.test_merged_outline_loaders_separate_paired_and_all_records tests.test_paired_overlay.PairedOverlayTest.test_merged_outline_loaders_return_empty_for_unreadable_file -v
```

Expected: 兩項都 FAIL；現有 loader 分別在損壞座標拋出 `ValueError`，以及在缺檔時
拋出 `FileNotFoundError`。

- [ ] **Step 3: 抽出安全行讀取與 polygon parser**

在 `ki67dtc/app_pipeline.py` 的 `load_merged_outlines()` 前新增：

```python
def _load_merged_outline_lines(merged_path: Path) -> list[str]:
    """安全讀取 merged outlines 的非空行。

    Args:
        merged_path: ``*_merged_cp_outlines.txt`` 路徑。

    Returns:
        保留原始兩行記錄順序的非空文字；讀取失敗時回傳空列表。
    """
    try:
        with merged_path.open("r", encoding="utf-8", errors="ignore") as file:
            return [line.strip() for line in file if line.strip()]
    except OSError:
        return []


def _parse_merged_outline_polygon(line: str) -> np.ndarray | None:
    """解析 merged outline 單行 polygon。

    Args:
        line: 逗號分隔的 x、y 座標，或代表缺值的 ``-1,-1``。

    Returns:
        至少三點的 ``int32`` polygon；缺值或格式錯誤時回傳 ``None``。
    """
    if line == "-1,-1":
        return None
    try:
        coords = list(map(int, line.split(",")))
    except ValueError:
        return None
    if len(coords) < 6 or len(coords) % 2:
        return None
    return np.asarray(coords, dtype=np.int32).reshape(-1, 2)
```

- [ ] **Step 4: 讓 paired 與 all loader 共用安全 parser**

將 `load_merged_outlines()` 改為逐筆保留每一側有效 polygon：

```python
def load_merged_outlines(merged_path: Path) -> OverlayPolygons:
    """讀取 merged outlines 中全部有效 nucleus 與 cytoplasm polygons。

    Args:
        merged_path: ``*_merged_cp_outlines.txt`` 路徑。

    Returns:
        每筆兩行記錄中所有可解析的單側 polygons；讀檔失敗時回傳空集合。
    """
    lines = _load_merged_outline_lines(merged_path)
    nucleus_polygons: list[np.ndarray] = []
    cytoplasm_polygons: list[np.ndarray] = []
    for index in range(0, len(lines) - 1, 2):
        nucleus = _parse_merged_outline_polygon(lines[index])
        cytoplasm = _parse_merged_outline_polygon(lines[index + 1])
        if nucleus is not None:
            nucleus_polygons.append(nucleus)
        if cytoplasm is not None:
            cytoplasm_polygons.append(cytoplasm)
    return OverlayPolygons(nucleus_polygons, cytoplasm_polygons)
```

將 `load_paired_merged_outlines()` 改為相同的行讀取與 parser，但只在兩側都有效時
append：

```python
def load_paired_merged_outlines(merged_path: Path) -> OverlayPolygons:
    """讀取 merged outlines，只保留核與質兩側都存在的配對。"""
    lines = _load_merged_outline_lines(merged_path)
    nucleus_polygons: list[np.ndarray] = []
    cytoplasm_polygons: list[np.ndarray] = []
    for index in range(0, len(lines) - 1, 2):
        nucleus = _parse_merged_outline_polygon(lines[index])
        cytoplasm = _parse_merged_outline_polygon(lines[index + 1])
        if nucleus is None or cytoplasm is None:
            continue
        nucleus_polygons.append(nucleus)
        cytoplasm_polygons.append(cytoplasm)
    return OverlayPolygons(nucleus_polygons, cytoplasm_polygons)
```

完整實作須保留目前 `load_paired_merged_outlines()` 的繁體中文 Google style
`Args`／`Returns` 說明。

- [ ] **Step 5: 執行 loader 測試並確認通過**

Run:

```powershell
& 'D:\Anaconda3\envs\ki67dtc\python.exe' -m unittest tests.test_paired_overlay.PairedOverlayTest.test_merged_outline_loaders_separate_paired_and_all_records tests.test_paired_overlay.PairedOverlayTest.test_merged_outline_loaders_return_empty_for_unreadable_file tests.test_paired_overlay.PairedOverlayTest.test_merged_outline_fallback_keeps_only_complete_pairs -v
```

Expected: 3 tests PASS。

- [ ] **Step 6: 新增 outline fallback 自動 PNG 輸出隔離 regression**

在 `tests/test_paired_overlay.py` 新增：

```python
def test_saved_outline_fallback_remains_paired_only(self) -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        image_dir = root / "data" / "input" / "demo" / "PC"
        outline_dir = root / "data" / "output" / "outline" / "demo"
        results_dir = root / "data" / "output" / "results" / "demo"
        image_dir.mkdir(parents=True)
        outline_dir.mkdir(parents=True)
        image_path = image_dir / "sample.png"
        base = np.full((18, 18, 3), 100, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(image_path), base))
        (outline_dir / "sample_merged_cp_outlines.txt").write_text(
            "5,5,9,5,9,9,5,9\n"
            "2,2,11,2,11,11,2,11\n"
            "12,10,16,10,16,15,12,15\n"
            "-1,-1\n"
            "-1,-1\n"
            "12,1,16,1,16,6,12,6\n",
            encoding="utf-8",
        )
        result = PipelineResult(
            data_folder=root / "data" / "input" / "demo",
            image_files=[image_path],
            results_dir=results_dir,
        )

        saved_path = save_pipeline_fill_overlays(result, alpha=0.5)[0]
        saved = cv2.imread(str(saved_path), cv2.IMREAD_COLOR)

        self.assertIsNotNone(saved)
        assert saved is not None
        self.assertFalse(np.array_equal(saved[3, 3], base[3, 3]))
        np.testing.assert_array_equal(saved[12, 14], base[12, 14])
        np.testing.assert_array_equal(saved[3, 14], base[3, 14])
```

- [ ] **Step 7: 執行 Task 1 完整測試並提交**

Run:

```powershell
& 'D:\Anaconda3\envs\ki67dtc\python.exe' -m unittest tests.test_paired_overlay -v
git diff --check -- ki67dtc/app_pipeline.py tests/test_paired_overlay.py
git diff -- ki67dtc/app_pipeline.py tests/test_paired_overlay.py
git add -- ki67dtc/app_pipeline.py tests/test_paired_overlay.py
git commit -m "feat(overlay): 解析完整 merged outlines 顯示資料"
```

Expected: 所有 paired overlay tests PASS；commit 只包含 loader、對應測試與輸出隔離 regression。

### Task 2: 讓 outline-only 舊資料切換 paired／all 預覽

**Files:**
- Modify: `ki67dtc/gui/main_window.py:28-41, 316-346, 1018-1025, 1130-1195, 1366-1476, 1645-1687, 1738-1746`
- Test: `tests/test_main_window_layout_contract.py:35-640`
- Modify: `Changelog.md:3-8`

**Interfaces:**
- Consumes: `load_merged_outlines(merged_path: Path) -> OverlayPolygons` 與 `load_paired_merged_outlines(merged_path: Path) -> OverlayPolygons`。
- Produces: `MainWindow._current_all_overlay_polygons: dict[Path, tuple[list[np.ndarray], list[np.ndarray]]]`。
- Produces: `MainWindow._has_all_outline_polygons(image_path: Path | None) -> bool`。
- Extends: `_create_overlay_bgr(..., all_nuc_polys: list[np.ndarray] | None = None, all_cyto_polys: list[np.ndarray] | None = None) -> np.ndarray`。
- Preserves: `_current_overlay_polygons` 只保存 paired polygons，供 Cell_ID、Ki67 與 highlight 對齊。
- Preserves: NPY／`PairedOverlayData` renderer 優先權與所有輸出路徑。

- [ ] **Step 1: 新增 outline-only 舊資料 fixture helper 與切換失敗測試**

在 `tests/test_main_window_layout_contract.py` 的 `MainWindowLayoutContractTest` 新增：

```python
def _load_outline_only_old_dataset(
    self,
    root: Path,
) -> tuple[Path, Path, np.ndarray]:
    """載入只有 merged outlines 的舊資料 fixture。"""
    image_dir = root / "data" / "input" / "demo" / "PC"
    outline_dir = root / "data" / "output" / "outline" / "demo"
    image_dir.mkdir(parents=True)
    outline_dir.mkdir(parents=True)
    image_path = image_dir / "sample.png"
    outline_path = outline_dir / "sample_merged_cp_outlines.txt"
    base = np.full((18, 18, 3), 100, dtype=np.uint8)
    self.assertTrue(cv2.imwrite(str(image_path), base))
    outline_path.write_text(
        "5,5,9,5,9,9,5,9\n"
        "2,2,11,2,11,11,2,11\n"
        "12,10,16,10,16,15,12,15\n"
        "-1,-1\n"
        "-1,-1\n"
        "12,1,16,1,16,6,12,6\n",
        encoding="utf-8",
    )
    original_cwd = os.getcwd()
    try:
        os.chdir(root)
        self.window._pipeline_result = PipelineResult(
            data_folder=root / "data" / "input" / "demo",
            image_files=[image_path],
        )
        self.window._current_image_index = 0
        self.window._load_image_and_overlays(image_path)
        self.window._update_display_pixmap()
    finally:
        os.chdir(original_cwd)
    return image_path, outline_path, base


def test_outline_only_old_dataset_switches_paired_and_all_display(self) -> None:
    with TemporaryDirectory() as temporary_directory:
        image_path, outline_path, base = self._load_outline_only_old_dataset(
            Path(temporary_directory)
        )
        original_outline = outline_path.read_bytes()
        paired = self.window._current_overlay_image_array.copy()
        self.window._selected_cell_id = "sample_1"
        self.window._highlight_enabled = True

        self.window.chk_paired_only.setChecked(False)
        QApplication.processEvents()
        all_regions = self.window._current_overlay_image_array

        self.assertTrue(self.window.chk_paired_only.isEnabled())
        self.assertFalse(self.window._paired_only_preference)
        np.testing.assert_array_equal(paired[12, 14], base[12, 14])
        np.testing.assert_array_equal(paired[3, 14], base[3, 14])
        self.assertFalse(np.array_equal(all_regions[12, 14], base[12, 14]))
        self.assertFalse(np.array_equal(all_regions[3, 14], base[3, 14]))
        self.assertEqual(outline_path.read_bytes(), original_outline)
        self.assertTrue(self.window._highlight_enabled)
        self.assertEqual(self.window._selected_cell_id, "sample_1")
        self.assertIsNone(self.window._pipeline_thread)
        self.assertIn(image_path, self.window._current_all_overlay_polygons)
```

- [ ] **Step 2: 執行切換測試並確認開關仍停用**

Run:

```powershell
& 'D:\Anaconda3\envs\ki67dtc\python.exe' -m unittest tests.test_main_window_layout_contract.MainWindowLayoutContractTest.test_outline_only_old_dataset_switches_paired_and_all_display -v
```

Expected: FAIL；`chk_paired_only.isEnabled()` 為 `False`，且 `MainWindow` 尚無 `_current_all_overlay_polygons`。

- [ ] **Step 3: 新增 all-polygons cache 並在載入 merged 檔時填入**

在 `ki67dtc/gui/main_window.py` import `load_merged_outlines`，並於 `__init__()` 新增：

```python
self._current_all_overlay_polygons: dict[
    Path, tuple[list[np.ndarray], list[np.ndarray]]
] = {}
```

在 `_load_image_and_overlays()` 的 merged outline 分支同時載入 paired 與 all：

```python
all_nuc_polys: list[np.ndarray] = []
all_cyto_polys: list[np.ndarray] = []
if merged_path is None:
    self._current_overlay_polygons.pop(img_path, None)
    self._current_all_overlay_polygons.pop(img_path, None)
else:
    paired_polygons = load_paired_merged_outlines(merged_path)
    all_polygons = load_merged_outlines(merged_path)
    nuc_polys = paired_polygons.nuc_polygons
    cyto_polys = paired_polygons.cyto_polygons
    all_nuc_polys = all_polygons.nuc_polygons
    all_cyto_polys = all_polygons.cyto_polygons
    if all_nuc_polys or all_cyto_polys:
        self._current_overlay_polygons[img_path] = (nuc_polys, cyto_polys)
        self._current_all_overlay_polygons[img_path] = (
            all_nuc_polys,
            all_cyto_polys,
        )
    else:
        self._current_overlay_polygons.pop(img_path, None)
        self._current_all_overlay_polygons.pop(img_path, None)
```

加入集中 availability helper：

```python
def _has_all_outline_polygons(self, image_path: Path | None) -> bool:
    """判斷目前影像是否保留可供雙模式顯示的 merged polygons。"""
    if image_path is None:
        return False
    polygons = self._current_all_overlay_polygons.get(image_path)
    return polygons is not None and bool(polygons[0] or polygons[1])
```

`_load_image_and_overlays()` 的兩次 `_set_paired_only_control_availability()` 都改用：

```python
display_modes_available = (
    paired_data is not None or self._has_all_outline_polygons(img_path)
)
self._set_paired_only_control_availability(
    display_modes_available,
    notify=not display_modes_available,
)
```

- [ ] **Step 4: 依 preference 選擇 paired 或 all outline polygons**

延伸 `_create_overlay_bgr()`：

```python
def _create_overlay_bgr(
    self,
    base_bgr: np.ndarray,
    nuc_polys: list[np.ndarray] | None,
    cyto_polys: list[np.ndarray] | None,
    nuc_mask: np.ndarray | None = None,
    cyto_mask: np.ndarray | None = None,
    paired_data: PairedOverlayData | None = None,
    all_nuc_polys: list[np.ndarray] | None = None,
    all_cyto_polys: list[np.ndarray] | None = None,
) -> np.ndarray:
```

outline fallback 分支選擇顯示 polygons，但 Ki67 與 highlight 繼續使用原本的 paired
`nuc_polys`／`cyto_polys`：

```python
display_nuc_polys = nuc_polys
display_cyto_polys = cyto_polys
if not self._paired_only_preference:
    if all_nuc_polys is not None:
        display_nuc_polys = all_nuc_polys
    if all_cyto_polys is not None:
        display_cyto_polys = all_cyto_polys
blended = self._create_outline_overlay_bgr(
    base_bgr,
    nuc_polys=display_nuc_polys,
    cyto_polys=display_cyto_polys,
)
```

`_load_image_and_overlays()` 與 `_update_display_pixmap()` 取得並傳入 all polygons。
`_update_display_pixmap()` 的 early return 必須把 `all_polys` 納入判斷：

```python
all_polys = self._current_all_overlay_polygons.get(img_path)
if polys is None and all_polys is None and masks is None and paired_data is None:
    self._set_pixmap_in_view(self._pixmap_from_bgr(self._current_image_array))
    return

nuc_polys, cyto_polys = polys if polys is not None else ([], [])
all_nuc_polys, all_cyto_polys = all_polys if all_polys is not None else (None, None)
display_bgr = self._create_overlay_bgr(
    self._current_image_array,
    nuc_polys,
    cyto_polys,
    nuc_mask=nuc_mask,
    cyto_mask=cyto_mask,
    paired_data=paired_data,
    all_nuc_polys=all_nuc_polys,
    all_cyto_polys=all_cyto_polys,
)
```

- [ ] **Step 5: 清理與恢復流程納入 all-polygons cache**

在 `_on_reset_clicked()` 與 `_populate_image_list([])` 清除：

```python
self._current_all_overlay_polygons.clear()
```

擴充 `_restore_paired_only_control_availability()`：

```python
self._set_paired_only_control_availability(
    paired_data is not None
    or has_complete_masks
    or self._has_all_outline_polygons(image_path)
)
```

Run 開始仍先將 preference 設為 `True` 並重繪；對 outline-only 舊資料，這會從
all mode 立即恢復 paired polygons。failed／stop 後則重新啟用 checkbox。

- [ ] **Step 6: 執行舊資料切換測試並確認通過**

Run:

```powershell
& 'D:\Anaconda3\envs\ki67dtc\python.exe' -m unittest tests.test_main_window_layout_contract.MainWindowLayoutContractTest.test_outline_only_old_dataset_switches_paired_and_all_display -v
```

Expected: PASS；未配對兩側只在 all mode 出現，來源檔 bytes 不變。

- [ ] **Step 7: 新增 Run／failed／stop 與無效 merged 檔 regression**

新增測試；先讓 outline-only fixture 顯示 all mode，再驗證 Run 立即回到 paired mode，
failed 與 stop 都恢復 availability：

```python
def test_outline_only_availability_recovers_after_failure_and_stop(self) -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        _, _, base = self._load_outline_only_old_dataset(root)
        self.window.chk_paired_only.setChecked(False)
        self.window.input_dir_edit.setText(str(root))

        with patch("ki67dtc.gui.main_window.PipelineThread"):
            self.window._on_run_clicked()
        self.assertFalse(self.window.chk_paired_only.isEnabled())
        self.assertTrue(self.window.chk_paired_only.isChecked())
        np.testing.assert_array_equal(
            self.window._current_overlay_image_array[12, 14], base[12, 14]
        )

        self.window._on_pipeline_failed("測試失敗")
        self.assertTrue(self.window.chk_paired_only.isEnabled())

        self.window.chk_paired_only.setChecked(False)
        with patch("ki67dtc.gui.main_window.PipelineThread"):
            self.window._on_run_clicked()
            self.window._on_stop_clicked()
        self.assertTrue(self.window.chk_paired_only.isEnabled())
        self.assertTrue(self.window.chk_paired_only.isChecked())
        np.testing.assert_array_equal(
            self.window._current_overlay_image_array[12, 14], base[12, 14]
        )
```

新增空檔與完全損壞檔案的降級測試：

```python
def test_invalid_outline_only_dataset_disables_paired_toggle(self) -> None:
    for content in ("", "not,a,polygon\n-1,-1\n"):
        with self.subTest(content=content), TemporaryDirectory() as directory:
            root = Path(directory)
            _, outline_path, _ = self._load_outline_only_old_dataset(root)
            outline_path.write_text(content, encoding="utf-8")
            original_cwd = os.getcwd()
            try:
                os.chdir(root)
                self.window._load_image_and_overlays(
                    self.window._pipeline_result.image_files[0]
                )
            finally:
                os.chdir(original_cwd)

            self.assertFalse(self.window.chk_paired_only.isEnabled())
            self.assertTrue(self.window.chk_paired_only.isChecked())
            self.assertIn("僅能顯示核質配對", self.window._last_status_message)
```

新增 NPY 與 merged outlines 同時存在時的來源優先權測試；merged 檔刻意加入 NPY
沒有的右上角未配對 cytoplasm，切到 all mode 後該像素仍不得出現：

```python
def test_segmentation_masks_remain_preferred_over_merged_outlines(self) -> None:
    original_cwd = os.getcwd()
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        image_dir = root / "data" / "input" / "demo" / "PC"
        segment_dir = root / "data" / "output" / "segment" / "demo"
        outline_dir = root / "data" / "output" / "outline" / "demo"
        image_dir.mkdir(parents=True)
        segment_dir.mkdir(parents=True)
        outline_dir.mkdir(parents=True)
        image_path = image_dir / "sample.png"
        base = np.full((18, 18, 3), 100, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(image_path), base))
        cyto_mask = np.zeros((18, 18), dtype=np.int32)
        cyto_mask[2:11, 2:11] = 1
        nuc_mask = np.zeros_like(cyto_mask)
        nuc_mask[5:9, 5:9] = 1
        np.save(segment_dir / "sample_cyto_seg.npy", {"masks": cyto_mask})
        np.save(segment_dir / "sample_nuc_seg.npy", {"masks": nuc_mask})
        (outline_dir / "sample_merged_cp_outlines.txt").write_text(
            "5,5,9,5,9,9,5,9\n"
            "2,2,11,2,11,11,2,11\n"
            "-1,-1\n"
            "12,1,16,1,16,6,12,6\n",
            encoding="utf-8",
        )
        try:
            os.chdir(root)
            self.window._pipeline_result = PipelineResult(
                data_folder=root / "data" / "input" / "demo",
                image_files=[image_path],
            )
            self.window._current_image_index = 0
            self.window._load_image_and_overlays(image_path)
            self.window.chk_paired_only.setChecked(False)
            QApplication.processEvents()
        finally:
            os.chdir(original_cwd)

        np.testing.assert_array_equal(
            self.window._current_overlay_image_array[3, 14], base[3, 14]
        )
        self.assertIn(image_path.stem, self.window._pipeline_result.paired_overlays)
```

- [ ] **Step 8: 執行 GUI tests 並修正任何回歸**

Run:

```powershell
& 'D:\Anaconda3\envs\ki67dtc\python.exe' -m unittest tests.test_main_window_layout_contract -v
```

Expected: 全部 GUI tests PASS，包含 NPY 優先、缺檔降級、Run／failed／stop、Reset 與新資料集預設。

- [ ] **Step 9: 更新 Changelog**

在 `Changelog.md` 的 `2026.08.03` 章節加入：

```markdown
  * 舊資料即使已清除 `*_seg.npy`，只要保留 merged outlines，GUI 仍可切換
    Paired Only 與全部 segmentation 物件；來源檔與自動輸出維持不變。
```

- [ ] **Step 10: 執行完整回歸、檢查差異並提交**

Run:

```powershell
& 'D:\Anaconda3\envs\ki67dtc\python.exe' -m unittest tests.test_main_window_layout_contract -v
& 'D:\Anaconda3\envs\ki67dtc\python.exe' -m unittest tests.test_paired_overlay -v
& 'D:\Anaconda3\envs\ki67dtc\python.exe' -m unittest tests.test_pipeline_device_option -v
& 'D:\Anaconda3\envs\ki67dtc\python.exe' -m unittest discover -s tests -v
git diff --check -- ki67dtc/gui/main_window.py tests/test_main_window_layout_contract.py Changelog.md
git diff -- ki67dtc/gui/main_window.py tests/test_main_window_layout_contract.py Changelog.md
git add -- ki67dtc/gui/main_window.py tests/test_main_window_layout_contract.py Changelog.md
git commit -m "feat(gui): 支援舊資料切換完整輪廓顯示"
```

Expected: 全部測試 PASS；commit 只包含 GUI outline-only 切換、對應 tests 與 Changelog。

## Final Verification

- [ ] 執行 `git status --short`，確認只剩執行前既有的使用者修改。
- [ ] 執行 `git show --name-only --format=oneline c4addb5..HEAD`，確認 feature commits 不含 `immunity/`、`CODEX_PLUGIN_SETUP.md` 或 `操作手冊.pptx`。
- [ ] 執行 `& 'D:\Anaconda3\envs\ki67dtc\python.exe' -m unittest tests.test_main_window_layout_contract.MainWindowLayoutContractTest.test_outline_only_old_dataset_switches_paired_and_all_display -v`，確認舊資料切換 PASS。
- [ ] 執行 `& 'D:\Anaconda3\envs\ki67dtc\python.exe' -m unittest tests.test_paired_overlay.PairedOverlayTest.test_saved_outline_fallback_remains_paired_only -v`，確認自動 PNG 仍為 Paired Only。
- [ ] 執行 `git diff --check c4addb5..HEAD`，確認提交範圍無空白錯誤。
