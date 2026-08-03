# GUI Paired Only Display Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 GUI 加入預設開啟的「僅顯示核質配對」開關，允許使用者只切換畫面上的 Paired Only／全部 segmentation 物件，且不改變任何輸出。

**Architecture:** 延伸既有 `PairedOverlayData`，在 segmentation masks 尚存在時快取全部唯一 label 的裁切區域，並新增全部物件 renderer。`MainWindow` 保存使用者偏好，依當前影像是否有完整快取或成對 masks 啟用 checkbox；所有檔案輸出繼續明確呼叫既有 Paired Only renderer。

**Tech Stack:** Python 3.10–3.12、NumPy、OpenCV、PyQt6、`unittest`

## Global Constraints

- checkbox 顯示文字必須是 `僅顯示核質配對`，`objectName` 必須是 `pairedOnlyCheck`。
- GUI 每次啟動、載入新資料集或重設後，顯示模式預設為 Paired Only。
- 取消勾選只顯示更多 segmentation 物件；不得重新執行 pipeline 或修改 masks。
- CSV、XLSX、NPY、面積圖與 `save_pipeline_fill_overlays()` 的 overlay PNG 不得讀取 GUI 顯示偏好。
- 缺少完整 nucleus／cytoplasm 顯示資料時，checkbox 必須勾選、停用並顯示非阻斷狀態訊息。
- 重要 public function 與 class 使用繁體中文 Google style docstring；不要加入描述明顯程式碼的註解。
- 不調整 Cellpose、配對感知小面積過濾、核質配對規則、`cell_status` 或 cleaned CSV 篩選。

---

## File Structure

- `ki67dtc/paired_overlay.py`：保存完整／配對顯示資料，提供 Paired Only 與全部物件 renderer。
- `ki67dtc/gui/main_window.py`：建立 checkbox、保存使用者偏好、判斷資料可用性並選擇 renderer。
- `tests/test_paired_overlay.py`：驗證精簡區域快取、全部物件 renderer 與輸出隔離。
- `tests/test_main_window_layout_contract.py`：驗證控制項、切換、缺檔降級、偏好恢復與 reset。
- `Changelog.md`：記錄新的 GUI 純顯示功能與輸出不變保證。

### Task 1: 快取全部 label regions 並新增全部物件 renderer

**Files:**
- Modify: `ki67dtc/paired_overlay.py:31-235`
- Test: `tests/test_paired_overlay.py:13-91`

**Interfaces:**
- Consumes: `LabelRegion`、`PairedLabelRegions`、`_extract_label_region()` 與 `_paint_region()`。
- Produces: `PairedOverlayData.all_cytoplasm_regions: tuple[LabelRegion, ...]`。
- Produces: `PairedOverlayData.all_nucleus_regions: tuple[LabelRegion, ...]`。
- Produces: `render_all_overlay_bgr(base_bgr: np.ndarray, data: PairedOverlayData, *, show_nucleus: bool = True, show_cytoplasm: bool = True, alpha: float = 0.5) -> np.ndarray`。
- Preserves: `render_paired_overlay_bgr()` 的簽章與 Paired Only 預設行為。

- [ ] **Step 1: 新增全部區域快取與 renderer 的失敗測試**

在 `tests/test_paired_overlay.py` 的 import 加入 `render_all_overlay_bgr`，並新增：

```python
def test_all_regions_are_cached_and_rendered_without_mutating_inputs(self) -> None:
    base = np.full((12, 14, 3), 120, dtype=np.uint8)
    cytoplasm_mask = np.zeros((12, 14), dtype=np.int32)
    cytoplasm_mask[1:10, 1:9] = 1
    cytoplasm_mask[1:4, 10:13] = 2
    nucleus_mask = np.zeros_like(cytoplasm_mask)
    nucleus_mask[4:6, 4:6] = 1
    nucleus_mask[8:10, 10:12] = 2
    base_before = base.copy()
    cytoplasm_before = cytoplasm_mask.copy()
    nucleus_before = nucleus_mask.copy()

    data = build_paired_overlay_data(cytoplasm_mask, nucleus_mask)
    paired = render_paired_overlay_bgr(base, data, alpha=0.5)
    all_regions = render_all_overlay_bgr(base, data, alpha=0.5)

    self.assertEqual(
        [region.label for region in data.all_cytoplasm_regions],
        [1, 2],
    )
    self.assertEqual(
        [region.label for region in data.all_nucleus_regions],
        [1, 2],
    )
    np.testing.assert_array_equal(paired[2, 11], base[2, 11])
    np.testing.assert_array_equal(paired[8, 10], base[8, 10])
    self.assertFalse(np.array_equal(all_regions[2, 11], base[2, 11]))
    self.assertFalse(np.array_equal(all_regions[8, 10], base[8, 10]))
    hidden = render_all_overlay_bgr(
        base,
        data,
        show_nucleus=False,
        show_cytoplasm=False,
        alpha=0.5,
    )
    lower_alpha = render_all_overlay_bgr(base, data, alpha=0.25)
    np.testing.assert_array_equal(hidden, base)
    full_distance = np.linalg.norm(
        all_regions[2, 11].astype(int) - base[2, 11].astype(int)
    )
    lower_distance = np.linalg.norm(
        lower_alpha[2, 11].astype(int) - base[2, 11].astype(int)
    )
    self.assertLess(lower_distance, full_distance)
    np.testing.assert_array_equal(base, base_before)
    np.testing.assert_array_equal(cytoplasm_mask, cytoplasm_before)
    np.testing.assert_array_equal(nucleus_mask, nucleus_before)
```

- [ ] **Step 2: 執行測試並確認因新介面不存在而失敗**

Run:

```powershell
python -m unittest tests.test_paired_overlay.PairedOverlayTest.test_all_regions_are_cached_and_rendered_without_mutating_inputs -v
```

Expected: FAIL，import 階段顯示 `cannot import name 'render_all_overlay_bgr'`。

- [ ] **Step 3: 延伸 `PairedOverlayData` 並讓配對與全部模式共用 regions**

在 `ki67dtc/paired_overlay.py` 修改 dataclass：

```python
@dataclass(frozen=True)
class PairedOverlayData:
    """保存單張影像的配對與完整 segmentation 顯示資料。

    Attributes:
        pairs: 依 nucleus label 排列的核質配對 regions。
        all_cytoplasm_regions: 依 label 排列的全部細胞質 regions。
        all_nucleus_regions: 依 label 排列的全部細胞核 regions。
        raw_nucleus_count: 原始細胞核 label 數量。
        raw_cytoplasm_count: 原始細胞質 label 數量。
        paired_nucleus_count: 成功配對的唯一細胞核數量。
        paired_cytoplasm_count: 成功配對的唯一細胞質數量。
    """

    pairs: tuple[PairedLabelRegions, ...]
    all_cytoplasm_regions: tuple[LabelRegion, ...]
    all_nucleus_regions: tuple[LabelRegion, ...]
    raw_nucleus_count: int
    raw_cytoplasm_count: int
    paired_nucleus_count: int
    paired_cytoplasm_count: int
```

在 `build_paired_overlay_data()` 中先建立全部唯一 label regions，再用同一批物件建立配對：

```python
cytoplasm_labels = [
    int(label) for label in np.unique(cytoplasm_mask) if label != 0
]
nucleus_labels = [
    int(label) for label in np.unique(nucleus_mask) if label != 0
]
cytoplasm_regions = {
    label: _extract_label_region(cytoplasm_mask, label)
    for label in cytoplasm_labels
}
nucleus_regions = {
    label: _extract_label_region(nucleus_mask, label)
    for label in nucleus_labels
}

paired_regions = [
    PairedLabelRegions(
        cytoplasm=cytoplasm_regions[cytoplasm_label],
        nucleus=nucleus_regions[nucleus_label],
    )
    for cytoplasm_label, nucleus_label in label_pairs
]

return PairedOverlayData(
    pairs=tuple(paired_regions),
    all_cytoplasm_regions=tuple(cytoplasm_regions.values()),
    all_nucleus_regions=tuple(nucleus_regions.values()),
    raw_nucleus_count=len(nucleus_regions),
    raw_cytoplasm_count=len(cytoplasm_regions),
    paired_nucleus_count=len({nucleus for _, nucleus in label_pairs}),
    paired_cytoplasm_count=len({cytoplasm for cytoplasm, _ in label_pairs}),
)
```

- [ ] **Step 4: 抽出共用 region renderer 並新增全部物件 public renderer**

以私有 helper 集中輸入驗證、填色、透明混合與輪廓繪製；傳入配對序列時保留目前依 pair 順序著色的行為：

```python
def _render_regions_overlay_bgr(
    base_bgr: np.ndarray,
    cytoplasm_regions: Sequence[LabelRegion],
    nucleus_regions: Sequence[LabelRegion],
    *,
    show_nucleus: bool,
    show_cytoplasm: bool,
    alpha: float,
) -> np.ndarray:
    """將指定核質 regions 繪製在 BGR 原圖上。"""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必須介於 0 與 1 之間")
    if base_bgr.ndim != 3 or base_bgr.shape[2] != 3:
        raise ValueError("base_bgr 必須是三通道 BGR 影像")

    overlay = base_bgr.copy()
    color_layer = np.zeros_like(overlay)
    paint_mask = np.zeros(overlay.shape[:2], dtype=bool)
    palette_bgr = tuple((blue, green, red) for red, green, blue in _PALETTE_RGB)
    nucleus_fill_bgr = (240, 0, 0)

    if show_cytoplasm:
        for index, region in enumerate(cytoplasm_regions):
            _paint_region(
                color_layer,
                paint_mask,
                region,
                palette_bgr[index % len(palette_bgr)],
            )
    if show_nucleus:
        for region in nucleus_regions:
            _paint_region(color_layer, paint_mask, region, nucleus_fill_bgr)

    overlay[paint_mask] = (
        alpha * color_layer[paint_mask] + (1.0 - alpha) * overlay[paint_mask]
    ).astype(np.uint8)

    if show_cytoplasm:
        for region in cytoplasm_regions:
            cv2.drawContours(overlay, region.contours, -1, (255, 190, 0), 1)
    if show_nucleus:
        for region in nucleus_regions:
            cv2.drawContours(overlay, region.contours, -1, (255, 0, 0), 1)
    return overlay
```

將既有 `render_paired_overlay_bgr()` 的內容改為呼叫 helper：

```python
return _render_regions_overlay_bgr(
    base_bgr,
    [pair.cytoplasm for pair in data.pairs],
    [pair.nucleus for pair in data.pairs],
    show_nucleus=show_nucleus,
    show_cytoplasm=show_cytoplasm,
    alpha=alpha,
)
```

新增 public function：

```python
def render_all_overlay_bgr(
    base_bgr: np.ndarray,
    data: PairedOverlayData,
    *,
    show_nucleus: bool = True,
    show_cytoplasm: bool = True,
    alpha: float = 0.5,
) -> np.ndarray:
    """將全部 segmentation 核質區域直接繪製在 BGR 原圖上。

    Args:
        base_bgr: OpenCV BGR 原圖。
        data: 同時包含配對與全部 label regions 的顯示資料。
        show_nucleus: 是否顯示全部細胞核。
        show_cytoplasm: 是否顯示全部細胞質。
        alpha: 填色透明度，範圍為 0 到 1。

    Returns:
        套用全部 segmentation 半透明填色與輪廓後的 BGR 影像。

    Raises:
        ValueError: 原圖不是三通道 BGR，或透明度不在 0 到 1 時拋出。
    """
    return _render_regions_overlay_bgr(
        base_bgr,
        data.all_cytoplasm_regions,
        data.all_nucleus_regions,
        show_nucleus=show_nucleus,
        show_cytoplasm=show_cytoplasm,
        alpha=alpha,
    )
```

- [ ] **Step 5: 執行 overlay 測試並確認通過**

Run:

```powershell
python -m unittest tests.test_paired_overlay -v
```

Expected: 既有 Paired Only、輸出 PNG、outline fallback 與新增全部模式測試全部 PASS。

- [ ] **Step 6: 檢查 Task 1 差異並提交**

Run:

```powershell
git diff --check
git diff -- ki67dtc/paired_overlay.py tests/test_paired_overlay.py
git add ki67dtc/paired_overlay.py tests/test_paired_overlay.py
git commit -m "feat(overlay): 新增全部分割物件顯示資料"
```

Expected: commit 只包含 overlay 資料、renderer 與對應測試。

### Task 2: 接上 GUI checkbox、可用性降級與偏好恢復

**Files:**
- Modify: `ki67dtc/gui/main_window.py:35-40, 327-351, 829-883, 986-1018, 1035-1080, 1112-1165, 1246-1357, 1409-1460, 1586-1595, 1640-1655`
- Test: `tests/test_main_window_layout_contract.py:19-27, 201-207, 396-435`
- Modify: `Changelog.md:1-6`

**Interfaces:**
- Consumes: `PairedOverlayData.all_cytoplasm_regions`、`PairedOverlayData.all_nucleus_regions` 與 `render_all_overlay_bgr()`（Task 1）。
- Produces: `MainWindow.chk_paired_only: QCheckBox`，`objectName == "pairedOnlyCheck"`。
- Produces: `MainWindow._paired_only_preference: bool`，只代表 GUI 顯示偏好，預設 `True`。
- Produces: `_set_paired_only_control_availability(available: bool, *, notify: bool = False) -> None`。
- Produces: `_on_paired_only_changed(checked: bool) -> None`。
- Preserves: `save_pipeline_fill_overlays()` 固定呼叫 `render_paired_overlay_bgr()`。

- [ ] **Step 1: 新增 checkbox layout contract 失敗測試**

擴充 `test_overlay_controls_share_image_file_header_row()` 並加入預設狀態斷言：

```python
self.assertIs(
    self.window.chk_paired_only.parent(),
    self.window.image_header_widget,
)
self.assertEqual(self.window.chk_paired_only.objectName(), "pairedOnlyCheck")
self.assertEqual(self.window.chk_paired_only.text(), "僅顯示核質配對")
self.assertTrue(self.window.chk_paired_only.isChecked())
self.assertFalse(self.window.chk_paired_only.isEnabled())
self.assertTrue(self.window._paired_only_preference)
```

- [ ] **Step 2: 執行 layout test 並確認缺少控制項而失敗**

Run:

```powershell
python -m unittest tests.test_main_window_layout_contract.MainWindowLayoutContractTest.test_overlay_controls_share_image_file_header_row -v
```

Expected: FAIL with `MainWindow` 沒有 `chk_paired_only`。

- [ ] **Step 3: 建立 checkbox、預設偏好與獨立 signal handler**

在 `MainWindow.__init__()` 初始化：

```python
self._paired_only_preference: bool = True
```

在影像 header 的核、質 checkbox 後建立控制項：

```python
self.chk_paired_only = QtWidgets.QCheckBox(
    "僅顯示核質配對",
    self.image_header_widget,
)
self.chk_paired_only.setObjectName("pairedOnlyCheck")
self.chk_paired_only.setChecked(True)
self.chk_paired_only.setEnabled(False)
```

把它加在 `chk_show_cyto` 後、`chk_show_ki67` 前，並連接獨立 handler：

```python
overlay_controls.addWidget(self.chk_show_nuc)
overlay_controls.addWidget(self.chk_show_cyto)
overlay_controls.addWidget(self.chk_paired_only)
overlay_controls.addWidget(self.chk_show_ki67)

self.chk_paired_only.toggled.connect(self._on_paired_only_changed)
```

新增 handler；不要把偏好寫入 `_on_overlay_controls_changed()`，否則使用者調透明度時可能覆蓋停用影像的既有偏好：

```python
def _on_paired_only_changed(self, checked: bool) -> None:
    """保存核質配對顯示偏好並重繪目前影像。"""
    if not self.chk_paired_only.isEnabled():
        return
    self._paired_only_preference = checked
    self._update_display_pixmap()
```

- [ ] **Step 4: 執行 layout test 並確認通過**

Run:

```powershell
python -m unittest tests.test_main_window_layout_contract.MainWindowLayoutContractTest.test_overlay_controls_share_image_file_header_row -v
```

Expected: PASS。

- [ ] **Step 5: 新增顯示切換、缺檔降級與偏好恢復失敗測試**

在 `tests/test_main_window_layout_contract.py` import `build_paired_overlay_data`，新增：

```python
def test_paired_only_checkbox_switches_display_without_starting_pipeline(self) -> None:
    with TemporaryDirectory() as tmp_dir:
        image_path = Path(tmp_dir) / "sample.png"
        base = np.full((12, 14, 3), 120, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(image_path), base))

        cyto_mask = np.zeros((12, 14), dtype=np.int32)
        cyto_mask[1:10, 1:9] = 1
        cyto_mask[1:4, 10:13] = 2
        nuc_mask = np.zeros_like(cyto_mask)
        nuc_mask[4:6, 4:6] = 1
        data = build_paired_overlay_data(cyto_mask, nuc_mask)
        self.window._pipeline_result = PipelineResult(
            data_folder=Path(tmp_dir),
            image_files=[image_path],
            paired_overlays={image_path.stem: data},
        )
        self.window._current_image_index = 0

        self.window._load_image_and_overlays(image_path)
        self.window._update_display_pixmap()
        paired = self.window._current_overlay_image_array.copy()
        self.window._selected_cell_id = "sample_1"
        self.window._highlight_enabled = True

        self.window.chk_paired_only.setChecked(False)
        QApplication.processEvents()
        all_regions = self.window._current_overlay_image_array

        self.assertTrue(self.window.chk_paired_only.isEnabled())
        self.assertFalse(self.window._paired_only_preference)
        np.testing.assert_array_equal(paired[2, 11], base[2, 11])
        self.assertFalse(np.array_equal(all_regions[2, 11], base[2, 11]))
        self.assertTrue(self.window._highlight_enabled)
        self.assertEqual(self.window._selected_cell_id, "sample_1")
        self.assertIsNone(self.window._pipeline_thread)


def test_missing_full_data_forces_paired_only_and_restores_preference(self) -> None:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        available_path = root / "available.png"
        missing_path = root / "missing.png"
        base = np.full((12, 14, 3), 120, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(available_path), base))
        self.assertTrue(cv2.imwrite(str(missing_path), base))

        cyto_mask = np.zeros((12, 14), dtype=np.int32)
        cyto_mask[1:10, 1:9] = 1
        nuc_mask = np.zeros_like(cyto_mask)
        nuc_mask[4:6, 4:6] = 1
        data = build_paired_overlay_data(cyto_mask, nuc_mask)
        self.window._pipeline_result = PipelineResult(
            data_folder=root,
            image_files=[available_path, missing_path],
            paired_overlays={available_path.stem: data},
        )

        self.window._current_image_index = 0
        self.window._load_image_and_overlays(available_path)
        self.window.chk_paired_only.setChecked(False)
        self.window._current_image_index = 1
        self.window._load_image_and_overlays(missing_path)

        self.assertFalse(self.window.chk_paired_only.isEnabled())
        self.assertTrue(self.window.chk_paired_only.isChecked())
        self.assertFalse(self.window._paired_only_preference)
        self.assertIn("僅能顯示核質配對", self.window._last_status_message)

        self.window._current_image_index = 0
        self.window._load_image_and_overlays(available_path)

        self.assertTrue(self.window.chk_paired_only.isEnabled())
        self.assertFalse(self.window.chk_paired_only.isChecked())
        self.assertFalse(self.window._paired_only_preference)


def test_reset_restores_paired_only_default(self) -> None:
    self.window.chk_paired_only.setEnabled(True)
    self.window.chk_paired_only.setChecked(False)

    self.window._on_reset_clicked()

    self.assertTrue(self.window._paired_only_preference)
    self.assertTrue(self.window.chk_paired_only.isChecked())
    self.assertFalse(self.window.chk_paired_only.isEnabled())


def test_loading_new_dataset_restores_paired_only_default(self) -> None:
    with TemporaryDirectory() as tmp_dir:
        dataset = Path(tmp_dir) / "data" / "input" / "demo"
        pc_dir = dataset / "PC"
        pc_dir.mkdir(parents=True)
        image_path = pc_dir / "sample.png"
        image = np.full((8, 8, 3), 120, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(image_path), image))
        self.window._paired_only_preference = False

        self.window._load_images_from_folder(dataset)

        self.assertTrue(self.window._paired_only_preference)
        self.assertTrue(self.window.chk_paired_only.isChecked())
```

擴充既有 `test_segmentation_masks_are_loaded_and_rendered_as_colored_overlay()`：

```python
self.assertTrue(self.window.chk_paired_only.isEnabled())
self.assertIn(image_path.stem, self.window._pipeline_result.paired_overlays)
```

- [ ] **Step 6: 執行新增 GUI tests 並確認行為尚未建置而失敗**

Run:

```powershell
python -m unittest tests.test_main_window_layout_contract.MainWindowLayoutContractTest.test_paired_only_checkbox_switches_display_without_starting_pipeline tests.test_main_window_layout_contract.MainWindowLayoutContractTest.test_missing_full_data_forces_paired_only_and_restores_preference tests.test_main_window_layout_contract.MainWindowLayoutContractTest.test_reset_restores_paired_only_default tests.test_main_window_layout_contract.MainWindowLayoutContractTest.test_loading_new_dataset_restores_paired_only_default -v
```

Expected: FAIL；Paired Only preference 尚未控制 renderer，可用性也尚未同步。

- [ ] **Step 7: 實作控制項可用性同步且保留使用者偏好**

在 `MainWindow` 新增：

```python
def _set_paired_only_control_availability(
    self,
    available: bool,
    *,
    notify: bool = False,
) -> None:
    """依完整 segmentation 顯示資料同步 Paired Only 控制項。

    Args:
        available: 當前影像是否具備完整 nucleus 與 cytoplasm regions。
        notify: 資料不足時是否更新非阻斷狀態訊息。
    """
    was_blocked = self.chk_paired_only.blockSignals(True)
    try:
        self.chk_paired_only.setEnabled(available)
        self.chk_paired_only.setChecked(
            self._paired_only_preference if available else True
        )
    finally:
        self.chk_paired_only.blockSignals(was_blocked)

    if notify and not available:
        self._show_status_message(
            "缺少完整 segmentation 資料，僅能顯示核質配對"
        )
```

在 `_load_image_and_overlays()` 讀取 masks 後，若 pipeline cache 尚無資料但兩張 masks 都存在，就建立一次並存入 `PipelineResult.paired_overlays`：

```python
paired_data = self._paired_overlay_data_for_image(img_path)
if paired_data is None and nuc_mask is not None and cyto_mask is not None:
    paired_data = build_paired_overlay_data(cyto_mask, nuc_mask)
    if self._pipeline_result is not None:
        self._pipeline_result.paired_overlays[img_path.stem] = paired_data

self._set_paired_only_control_availability(
    paired_data is not None,
)
```

完成 outlines 載入判斷後再通知資料不足，避免既有「沒有找到 outlines」訊息覆蓋規格要求的狀態；完全沒有 masks、cache 與 outlines 的 early-return 分支也必須先執行：

```python
self._set_paired_only_control_availability(
    paired_data is not None,
    notify=paired_data is None,
)
```

載入影像失敗、`_on_reset_clicked()` 與 `_populate_image_list([])` 時呼叫：

```python
self._set_paired_only_control_availability(False)
```

`_on_reset_clicked()` 呼叫前先恢復偏好：

```python
self._paired_only_preference = True
self._set_paired_only_control_availability(False)
```

`_load_images_from_folder()` 解析出有效的新資料集後，也必須先恢復預設再建立影像清單：

```python
self._paired_only_preference = True
self._set_paired_only_control_availability(False)
```

- [ ] **Step 8: 依偏好選擇 renderer，並讓記憶體快取在 NPY 清除後仍可重繪**

在 imports 加入 `render_all_overlay_bgr`。在 `_create_overlay_bgr()` 的 `paired_data` 分支，以及 `_apply_segmentation_mask_overlay()` 建立資料後，使用相同選擇：

```python
renderer = (
    render_paired_overlay_bgr
    if self._paired_only_preference
    else render_all_overlay_bgr
)
blended = renderer(
    base_bgr,
    paired_data,
    show_nucleus=self._show_nuc,
    show_cytoplasm=self._show_cyto,
    alpha=self._overlay_alpha,
)
```

在 `_update_display_pixmap()` 先取得 `paired_data`，並修正 early return，避免 NPY 與 outlines 已清除時忽略 pipeline 的記憶體快取：

```python
polys = self._current_overlay_polygons.get(img_path)
masks = self._current_overlay_masks.get(img_path)
paired_data = self._paired_overlay_data_for_image(img_path)
if polys is None and masks is None and paired_data is None:
    self._set_pixmap_in_view(self._pixmap_from_bgr(self._current_image_array))
    return
```

`save_pipeline_fill_overlays()` 不得改用 `render_all_overlay_bgr()`，也不得新增顯示偏好參數。將 pipeline 完成訊息由強制模式改為中性統計：

```python
"核質配對統計："
f"cytoplasm {paired_cytoplasm_count}/{raw_cytoplasm_count}，"
f"nucleus {paired_nucleus_count}/{raw_nucleus_count}"
```

- [ ] **Step 9: 執行 GUI 與輸出隔離測試並確認通過**

Run:

```powershell
python -m unittest tests.test_main_window_layout_contract -v
python -m unittest tests.test_paired_overlay -v
python -m unittest tests.test_pipeline_device_option -v
```

Expected: 全部 PASS；尤其 `test_saved_overlay_uses_same_paired_only_data` 仍確認未配對物件沒有寫入 overlay PNG。

- [ ] **Step 10: 更新 Changelog 並執行完整回歸測試**

在 `Changelog.md` 頂端加入：

```markdown
## 2026.08.03

  * GUI overlay 控制列新增「僅顯示核質配對」開關，可即時切換
    Paired Only 與全部 segmentation 物件；此設定只影響畫面，不修改
    segmentation、CSV、XLSX 或自動輸出的 overlay PNG。
```

Run:

```powershell
python -m unittest discover -s tests -v
git diff --check
```

Expected: 全部測試 PASS，`git diff --check` 無輸出。

- [ ] **Step 11: 檢查 Task 2 差異並提交**

Run:

```powershell
git diff -- ki67dtc/gui/main_window.py tests/test_main_window_layout_contract.py Changelog.md
git add ki67dtc/gui/main_window.py tests/test_main_window_layout_contract.py Changelog.md
git commit -m "feat(gui): 新增核質配對顯示開關"
```

Expected: commit 只包含 GUI 開關、顯示模式整合、測試與 Changelog；不包含工作區既有的 immunity 變更或其他未追蹤檔案。

## Final Verification

- [ ] 執行 `git status --short`，確認只顯示執行前就存在的使用者變更。
- [ ] 執行 `git show --name-only --format=oneline HEAD~2..HEAD`，確認兩個 feature commits 不包含 `immunity/`、`CODEX_PLUGIN_SETUP.md` 或 `操作手冊.pptx`。
- [ ] 執行 `python -m unittest tests.test_main_window_layout_contract.MainWindowLayoutContractTest.test_paired_only_checkbox_switches_display_without_starting_pipeline -v`，確認 GUI 切換測試 PASS。
- [ ] 執行 `python -m unittest tests.test_paired_overlay.PairedOverlayTest.test_saved_overlay_uses_same_paired_only_data -v`，確認自動輸出的 overlay PNG 仍為 Paired Only。
