# Main UI SegmentationUI Style Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 將目前 PyQt6 main UI 重整成接近 `SegmentationUI` 的深色 cyan 工具介面，同時保留主線 `ki67dtc.app_pipeline` 與最新分析流程。

**Architecture:** 保留 `app.py -> ki67dtc.gui.main_window.MainWindow -> ki67dtc.app_pipeline.run_pipeline()` 的主線架構，不移植 `GUI_v2/controller.py`。UI 會拆出 theme 與 icon helpers，`MainWindow` 只負責組裝畫面、連接 action、更新 view state。

**Tech Stack:** Python, PyQt6, unittest, Qt offscreen UI tests, current `ki67dtc` analysis backend.

---

## Confirmed Design

- 視覺風格對齊 `SegmentationUI`：
  - 深色 navy / charcoal background。
  - cyan 作為主要 accent。
  - 字體使用 `Segoe UI`。
  - 表格、清單、console 使用 compact scientific tool density。
- 上方選單列：
  - `檔案`
    - `開啟`
  - `分析選項`
    - `核來源`
    - `Ki67 Backend`
    - `分析方法`
    - `螢光分析`
    - `Ki67 分析`
    - `清理暫存檔案`
- 主畫面比例：
  - 左側 2/3：主圖片顯示區。
  - 右側 1/3：垂直 4 等份。
- 右側四區：
  - 第一區：上半部終端輸出，下半部三個圖示按鈕。
  - 第二區：資料夾圖片 list。
  - 第三區：分析後特徵參數 table。
  - 第四區：沿用 `SegmentationUI` 的細胞面積分析圖，不改成 Ki67 強度分布。
- 三個圖示按鈕：
  - 置中排列。
  - 預設藍底黑色 icon。
  - 開始按鈕進入 running 後 icon 變白色。

## PyQt Skill Research

- 本機目前沒有已安裝的 PyQt/PySide skill。
- 網路搜尋到兩個可參考來源：
  - `CodeAtCode/oss-ai-skills` 的 `frameworks/pyqt/SKILL.md`：較接近 PyQt/PySide coding guidance。
  - `TheQtCompanyRnD/agent-skills`：Qt 官方 AI skills，偏 Qt/QML/C++ 與 Qt 設計原則，可當設計與 Qt pattern 參考。
- 建議：
  - 實作本計畫時，不需要先安裝第三方 skill。
  - 若後續要正式引入 PyQt/PySide 專用 skill，再先審閱其 `SKILL.md`，確認不會和本專案 PyQt6 實作習慣衝突。

## File Structure

### Create

- `tests/test_main_window_layout_contract.py`
  - 使用 Qt offscreen 建立 `MainWindow`。
  - 驗證 menu、action、right panel、button state contract。
- `ki67dtc/gui/theme.py`
  - 集中管理 SegmentationUI-style palette 與 QSS。
- `ki67dtc/gui/icons.py`
  - 用 Qt built-in standard icons 產生 start/stop/restart icon。
  - 支援 idle/running icon color。

### Modify

- `ki67dtc/gui/main_window.py`
  - 重整 `_build_ui()`。
  - 拆分 menu、image area、right panels、control buttons。
  - 連接 menu actions 到既有 handler / state。
  - 新增 terminal output widget。
  - 保留既有 image display、overlay、table refresh 邏輯。
- `app.py`
  - 如需要，只補上 app-level stylesheet 載入或 high-DPI 設定。
- `README.md`
  - 補充新版 GUI 操作入口與選單說明。

### Do Not Modify In This Plan

- 不改 `ki67dtc.app_pipeline.run_pipeline()` 的分析行為。
- 不改 `ki67dtc.cell_anal.py` 的特徵計算。
- 不改 `ki67dtc.img_prep.py` 的 segmentation 流程。
- 不保留或使用 root 的 `img_prep 1.py` 作為主線來源。

---

### Task 1: Add UI Layout Contract Tests

**Files:**
- Create: `tests/test_main_window_layout_contract.py`
- Test: `tests/test_main_window_layout_contract.py`

- [ ] **Step 1: Write failing layout contract tests**

Create `tests/test_main_window_layout_contract.py`:

```python
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QToolButton

from ki67dtc.gui.main_window import MainWindow


class MainWindowLayoutContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()

    def tearDown(self) -> None:
        self.window.close()

    def test_menu_bar_contains_file_and_analysis_options(self) -> None:
        menu_titles = [action.text() for action in self.window.menuBar().actions()]

        self.assertIn("檔案", menu_titles)
        self.assertIn("分析選項", menu_titles)

    def test_file_menu_exposes_open_action(self) -> None:
        self.assertEqual(self.window.action_open_input.text(), "開啟")

    def test_analysis_menu_exposes_expected_options(self) -> None:
        expected = [
            "核來源",
            "Ki67 Backend",
            "分析方法",
            "螢光分析",
            "Ki67 分析",
            "清理暫存檔案",
        ]
        actual = [action.text() for action in self.window.analysis_option_actions]

        self.assertEqual(actual, expected)

    def test_right_side_has_four_named_panels(self) -> None:
        expected_names = [
            "terminalPanel",
            "imageListPanel",
            "featureTablePanel",
            "areaChartPanel",
        ]

        for object_name in expected_names:
            with self.subTest(object_name=object_name):
                self.assertIsNotNone(self.window.findChild(object, object_name))

    def test_control_buttons_are_icon_only_and_centered_contract(self) -> None:
        buttons = self.window.control_button_row.findChildren(QToolButton)

        self.assertEqual(
            [button.objectName() for button in buttons],
            ["startButton", "stopButton", "restartButton"],
        )
        self.assertTrue(all(button.text() == "" for button in buttons))
        self.assertEqual(
            self.window.control_button_row.property("alignmentRole"),
            "centeredIconControls",
        )

    def test_start_button_icon_state_changes_when_running(self) -> None:
        self.window._set_running_state(False)
        self.assertEqual(self.window.start_button.property("iconTone"), "black")

        self.window._set_running_state(True)
        self.assertEqual(self.window.start_button.property("iconTone"), "white")

    def test_area_chart_panel_keeps_cell_area_analysis_label(self) -> None:
        self.assertEqual(self.window.area_chart_title.text(), "細胞面積分析")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest tests.test_main_window_layout_contract -v
```

Expected:

```text
FAILED
AttributeError: 'MainWindow' object has no attribute 'action_open_input'
```

- [ ] **Step 3: Commit failing test**

```powershell
git add tests/test_main_window_layout_contract.py
git commit -m "test(ui): 新增主畫面版面契約測試"
```

---

### Task 2: Add SegmentationUI Theme Helpers

**Files:**
- Create: `ki67dtc/gui/theme.py`
- Modify: `ki67dtc/gui/main_window.py`
- Test: `tests/test_main_window_layout_contract.py`

- [ ] **Step 1: Create theme helper**

Create `ki67dtc/gui/theme.py`:

```python
"""SegmentationUI 風格的 PyQt6 主題設定。"""

from __future__ import annotations


APP_QSS = """
QMainWindow, QWidget {
    background-color: #1C2030;
    color: #E8EDF5;
    font-family: "Segoe UI";
    font-size: 11pt;
}

QMenuBar {
    background-color: #161A26;
    color: #E8EDF5;
    border-bottom: 1px solid #2E3548;
}

QMenuBar::item {
    padding: 6px 12px;
}

QMenuBar::item:selected {
    background-color: #252A3A;
}

QMenu {
    background-color: #1C2030;
    color: #E8EDF5;
    border: 1px solid #2E3548;
}

QMenu::item {
    padding: 6px 28px 6px 24px;
}

QMenu::item:selected {
    background-color: #00AEEF;
    color: #0A0E18;
}

QFrame#terminalPanel,
QFrame#imageListPanel,
QFrame#featureTablePanel,
QFrame#areaChartPanel {
    background-color: #161A26;
    border: 1px solid #2E3548;
}

QTextEdit#terminalOutput {
    background-color: #0F1218;
    color: #C8D2E0;
    border: none;
    font-family: "Consolas";
    font-size: 10pt;
}

QListWidget#folderImageList,
QTableWidget#featureParameterTable {
    background-color: #0F1218;
    color: #D6DEEA;
    border: none;
    gridline-color: #2E3548;
}

QListWidget#folderImageList::item:selected {
    background-color: #00AEEF;
    color: #0A0E18;
}

QToolButton {
    background-color: #00AEEF;
    border: none;
    border-radius: 4px;
    min-width: 44px;
    min-height: 32px;
}

QToolButton:disabled {
    background-color: #252A3A;
}

QLabel#areaChartLabel {
    background-color: #E8EFF7;
    border: 1px solid #2E3548;
}
"""
```

- [ ] **Step 2: Apply theme in `MainWindow.__init__`**

Modify `ki67dtc/gui/main_window.py`:

```python
from .theme import APP_QSS
```

Inside `MainWindow.__init__`, after `self.resize(1400, 900)`:

```python
self.setStyleSheet(APP_QSS)
```

- [ ] **Step 3: Run tests**

Run:

```powershell
python -m unittest tests.test_main_window_layout_contract -v
```

Expected: still fails on missing UI contract attributes. Theme helper import must not error.

- [ ] **Step 4: Commit theme helper**

```powershell
git add ki67dtc/gui/theme.py ki67dtc/gui/main_window.py
git commit -m "style(ui): 加入 SegmentationUI 深色主題"
```

---

### Task 3: Add Icon Helper And Running State Contract

**Files:**
- Create: `ki67dtc/gui/icons.py`
- Modify: `ki67dtc/gui/main_window.py`
- Test: `tests/test_main_window_layout_contract.py`

- [ ] **Step 1: Create icon helper**

Create `ki67dtc/gui/icons.py`:

```python
"""主畫面 icon-only 控制按鈕。"""

from __future__ import annotations

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QStyle, QWidget


ICON_SIZE = QSize(20, 20)


def standard_icon(widget: QWidget, pixmap: QStyle.StandardPixmap, color: str) -> QIcon:
    """建立套色後的 Qt standard icon。

    Args:
        widget: 用來取得目前 style 的 widget。
        pixmap: Qt standard pixmap 類型。
        color: icon 顏色，例如 `black` 或 `white`。

    Returns:
        套色後的 QIcon。
    """
    base = widget.style().standardIcon(pixmap).pixmap(ICON_SIZE)
    tinted = QPixmap(base.size())
    tinted.fill(QColor("transparent"))

    painter = QPainter(tinted)
    painter.drawPixmap(0, 0, base)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(tinted.rect(), QColor(color))
    painter.end()

    return QIcon(tinted)
```

- [ ] **Step 2: Add `_set_running_state` to `MainWindow`**

Modify `ki67dtc/gui/main_window.py`:

```python
from .icons import standard_icon
```

Add method inside `MainWindow`:

```python
def _set_running_state(self, running: bool) -> None:
    """更新分析執行中的控制按鈕狀態。"""
    self._is_running = running
    tone = "white" if running else "black"
    self.start_button.setProperty("iconTone", tone)
    self.start_button.setIcon(
        standard_icon(self, QtWidgets.QStyle.StandardPixmap.SP_MediaPlay, tone)
    )
    self.stop_button.setEnabled(running)
    self.restart_button.setEnabled(True)
```

- [ ] **Step 3: Run tests**

Run:

```powershell
python -m unittest tests.test_main_window_layout_contract -v
```

Expected: tests still fail until buttons are created in Task 5, but import and method definition must not error after Task 5 wiring.

- [ ] **Step 4: Commit icon helper**

```powershell
git add ki67dtc/gui/icons.py ki67dtc/gui/main_window.py
git commit -m "style(ui): 新增主畫面圖示按鈕狀態"
```

---

### Task 4: Replace Top Form Controls With Menu Actions

**Files:**
- Modify: `ki67dtc/gui/main_window.py`
- Test: `tests/test_main_window_layout_contract.py`

- [ ] **Step 1: Add menu builder**

Add this method inside `MainWindow`:

```python
def _build_menu_bar(self) -> None:
    """建立檔案與分析選項選單。"""
    file_menu = self.menuBar().addMenu("檔案")
    self.action_open_input = file_menu.addAction("開啟")
    self.action_open_input.triggered.connect(self._on_browse_input)

    analysis_menu = self.menuBar().addMenu("分析選項")

    self.action_nuc_source = analysis_menu.addAction("核來源")
    self.action_ki67_backend = analysis_menu.addAction("Ki67 Backend")
    self.action_feature_backend = analysis_menu.addAction("分析方法")

    self.action_fluor_analy = analysis_menu.addAction("螢光分析")
    self.action_fluor_analy.setCheckable(True)
    self.action_fluor_analy.setChecked(True)

    self.action_ki67_analy = analysis_menu.addAction("Ki67 分析")
    self.action_ki67_analy.setCheckable(True)
    self.action_ki67_analy.setChecked(True)

    self.action_clean_temp = analysis_menu.addAction("清理暫存檔案")
    self.action_clean_temp.setCheckable(True)
    self.action_clean_temp.setChecked(True)

    self.analysis_option_actions = [
        self.action_nuc_source,
        self.action_ki67_backend,
        self.action_feature_backend,
        self.action_fluor_analy,
        self.action_ki67_analy,
        self.action_clean_temp,
    ]
```

- [ ] **Step 2: Call menu builder**

In `MainWindow.__init__`, before `self._build_ui()`:

```python
self._build_menu_bar()
```

- [ ] **Step 3: Preserve analysis option reads**

When building the pipeline request in `_on_run_clicked`, replace checkbox reads with action reads:

```python
fluor_analy = self.action_fluor_analy.isChecked()
ki67 = self.action_ki67_analy.isChecked()
clean_temp = self.action_clean_temp.isChecked()
```

Keep existing default strings for `nuc_source`, `ki67_backend`, and `feature_backend` until Task 8 adds submenus or dialogs.

- [ ] **Step 4: Run menu contract tests**

Run:

```powershell
python -m unittest tests.test_main_window_layout_contract.MainWindowLayoutContractTest.test_menu_bar_contains_file_and_analysis_options -v
python -m unittest tests.test_main_window_layout_contract.MainWindowLayoutContractTest.test_file_menu_exposes_open_action -v
python -m unittest tests.test_main_window_layout_contract.MainWindowLayoutContractTest.test_analysis_menu_exposes_expected_options -v
```

Expected: all three tests pass.

- [ ] **Step 5: Commit menu action changes**

```powershell
git add ki67dtc/gui/main_window.py tests/test_main_window_layout_contract.py
git commit -m "feat(ui): 將輸入與分析選項移到選單列"
```

---

### Task 5: Rebuild Main Layout Into 2/3 Image And 1/3 Four Panels

**Files:**
- Modify: `ki67dtc/gui/main_window.py`
- Test: `tests/test_main_window_layout_contract.py`

- [ ] **Step 1: Add panel helper**

Add helper methods inside `MainWindow`:

```python
def _new_panel(self, object_name: str, title: str) -> QtWidgets.QFrame:
    """建立右側等分 panel。"""
    panel = QtWidgets.QFrame(self)
    panel.setObjectName(object_name)
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(10, 8, 10, 8)
    layout.setSpacing(6)

    label = QLabel(title, panel)
    label.setObjectName(f"{object_name}Title")
    layout.addWidget(label)
    return panel
```

- [ ] **Step 2: Replace `_build_ui()` root layout**

Inside `_build_ui()`, use a horizontal `QSplitter`:

```python
central = QWidget(self)
self.setCentralWidget(central)
root = QHBoxLayout(central)
root.setContentsMargins(8, 8, 8, 8)

self.main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
root.addWidget(self.main_splitter)

self.image_panel = QWidget(self)
self.image_panel.setObjectName("imagePanel")
image_layout = QVBoxLayout(self.image_panel)
image_layout.setContentsMargins(0, 0, 0, 0)

self.image_view = ZoomableGraphicsView(self)
self.image_view.setObjectName("mainImageView")
self.image_scene = QGraphicsScene(self)
self.image_view.setScene(self.image_scene)
image_layout.addWidget(self.image_view)

self.image_file_label = QLabel("Image File Name", self)
self.image_file_label.setObjectName("imageFileName")
image_layout.addWidget(self.image_file_label)

self.right_splitter = QSplitter(Qt.Orientation.Vertical, self)
self.right_splitter.setObjectName("rightPanelSplitter")

self.main_splitter.addWidget(self.image_panel)
self.main_splitter.addWidget(self.right_splitter)
self.main_splitter.setSizes([2, 1])
```

Then add four panel builders from Tasks 6 and 7.

- [ ] **Step 3: Run tests**

Run:

```powershell
python -m unittest tests.test_main_window_layout_contract -v
```

Expected: right-side panel tests fail until Task 6 creates panels.

- [ ] **Step 4: Commit layout skeleton**

```powershell
git add ki67dtc/gui/main_window.py
git commit -m "feat(ui): 重整主畫面左右分割版面"
```

---

### Task 6: Build Right-Side Panels And Centered Icon Controls

**Files:**
- Modify: `ki67dtc/gui/main_window.py`
- Test: `tests/test_main_window_layout_contract.py`

- [ ] **Step 1: Build terminal/control panel**

Add method:

```python
def _build_terminal_panel(self) -> QtWidgets.QFrame:
    """建立終端輸出與置中圖示控制列。"""
    panel = self._new_panel("terminalPanel", "1. 輸出主控台")
    layout = panel.layout()

    self.terminal_output = QtWidgets.QTextEdit(panel)
    self.terminal_output.setObjectName("terminalOutput")
    self.terminal_output.setReadOnly(True)
    self.terminal_output.setPlainText("[INFO] 等待開啟資料夾...")
    layout.addWidget(self.terminal_output, stretch=1)

    control_container = QWidget(panel)
    control_container.setObjectName("controlButtonRow")
    control_container.setProperty("alignmentRole", "centeredIconControls")
    self.control_button_row = control_container

    row = QHBoxLayout(control_container)
    row.setContentsMargins(0, 4, 0, 0)
    row.addStretch(1)

    self.start_button = QtWidgets.QToolButton(control_container)
    self.start_button.setObjectName("startButton")
    self.start_button.clicked.connect(self._on_run_clicked)

    self.stop_button = QtWidgets.QToolButton(control_container)
    self.stop_button.setObjectName("stopButton")
    self.stop_button.clicked.connect(self._on_stop_clicked)

    self.restart_button = QtWidgets.QToolButton(control_container)
    self.restart_button.setObjectName("restartButton")
    self.restart_button.clicked.connect(self._on_reset_clicked)

    for button in (self.start_button, self.stop_button, self.restart_button):
        button.setText("")
        row.addWidget(button)

    row.addStretch(1)
    layout.addWidget(control_container)
    self._set_running_state(False)
    return panel
```

- [ ] **Step 2: Build image list panel**

Add method:

```python
def _build_image_list_panel(self) -> QtWidgets.QFrame:
    """建立資料夾圖片清單。"""
    panel = self._new_panel("imageListPanel", "2. 影像清單")
    layout = panel.layout()

    self.image_list = QListWidget(panel)
    self.image_list.setObjectName("folderImageList")
    self.image_list.currentRowChanged.connect(self._on_image_selection_changed)
    layout.addWidget(self.image_list)
    return panel
```

- [ ] **Step 3: Build feature table panel**

Add method:

```python
def _build_feature_table_panel(self) -> QtWidgets.QFrame:
    """建立特徵參數表格。"""
    panel = self._new_panel("featureTablePanel", "3. 分析結果（特徵參數）")
    layout = panel.layout()

    self.results_table = QTableWidget(panel)
    self.results_table.setObjectName("featureParameterTable")
    self.results_table.setColumnCount(5)
    self.results_table.setHorizontalHeaderLabels(
        ["Cell_ID", "Area", "Circularity", "Ki67", "Mean Intensity"]
    )
    layout.addWidget(self.results_table)
    return panel
```

- [ ] **Step 4: Build area chart panel**

Add method:

```python
def _build_area_chart_panel(self) -> QtWidgets.QFrame:
    """建立細胞面積分析圖區。"""
    panel = self._new_panel("areaChartPanel", "4. 細胞面積分析")
    layout = panel.layout()

    self.area_chart_title = panel.findChild(QLabel, "areaChartPanelTitle")
    self.area_chart_label = QLabel(panel)
    self.area_chart_label.setObjectName("areaChartLabel")
    self.area_chart_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    self.area_chart_label.setText("")
    layout.addWidget(self.area_chart_label)
    return panel
```

- [ ] **Step 5: Add panels to right splitter**

At the end of `_build_ui()`:

```python
self.right_splitter.addWidget(self._build_terminal_panel())
self.right_splitter.addWidget(self._build_image_list_panel())
self.right_splitter.addWidget(self._build_feature_table_panel())
self.right_splitter.addWidget(self._build_area_chart_panel())
self.right_splitter.setSizes([1, 1, 1, 1])
```

- [ ] **Step 6: Run panel contract tests**

Run:

```powershell
python -m unittest tests.test_main_window_layout_contract -v
```

Expected: panel, button, and area chart title tests pass.

- [ ] **Step 7: Commit right panel changes**

```powershell
git add ki67dtc/gui/main_window.py tests/test_main_window_layout_contract.py
git commit -m "feat(ui): 新增右側四區資訊面板"
```

---

### Task 7: Wire Existing Image And Result Updates To New Widgets

**Files:**
- Modify: `ki67dtc/gui/main_window.py`
- Test: `tests/test_main_window_layout_contract.py`

- [ ] **Step 1: Update image list population**

Modify `_populate_image_list()` so it writes to `self.image_list`:

```python
def _populate_image_list(self) -> None:
    """更新資料夾圖片清單。"""
    self.image_list.clear()
    if not self._pipeline_result:
        return

    for image_path in self._pipeline_result.image_files:
        self.image_list.addItem(image_path.name)
```

- [ ] **Step 2: Update image selection handler**

Ensure `_on_image_selection_changed()` reads `self.image_list.currentRow()` and updates the main image.

```python
def _on_image_selection_changed(self, row: int) -> None:
    """切換目前影像。"""
    if not self._pipeline_result or row < 0:
        return
    self._current_image_index = row
    self._load_image_and_overlays()
    self._update_display_pixmap()
    self._refresh_results_table_for_current_image()
```

- [ ] **Step 3: Update progress callback to terminal output**

Modify `_on_progress_changed()`:

```python
def _on_progress_changed(self, done: int, total: int, message: str) -> None:
    """同步進度列與終端輸出。"""
    if total > 0:
        percent = int(done / total * 100)
        self.progress_bar.setValue(percent)
    self.statusBar().showMessage(message)
    self.terminal_output.append(f"[INFO] {message}")
```

If `self.progress_bar` is removed from the redesigned UI, delete only the `progress_bar` line and keep status bar + terminal append.

- [ ] **Step 4: Keep feature table behavior**

Ensure `_refresh_results_table_for_current_image()` still writes to `self.results_table`. Do not rename the table outside `self.results_table`, because existing row selection and highlight logic depends on it.

- [ ] **Step 5: Display cell area analysis chart**

Add method:

```python
def _load_area_chart(self) -> None:
    """載入細胞面積分析圖。"""
    chart_path = Path("data/output/figure") / "all_log_cell_area_distribution.png"
    if not chart_path.exists():
        self.area_chart_label.clear()
        return

    pixmap = QPixmap(str(chart_path))
    if pixmap.isNull():
        self.area_chart_label.clear()
        return

    self.area_chart_label.setPixmap(
        pixmap.scaled(
            self.area_chart_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    )
```

Call `_load_area_chart()` after pipeline finishes.

- [ ] **Step 6: Run tests**

Run:

```powershell
python -m unittest tests.test_main_window_layout_contract -v
```

Expected: all contract tests pass.

- [ ] **Step 7: Commit wiring changes**

```powershell
git add ki67dtc/gui/main_window.py
git commit -m "feat(ui): 串接新版主畫面資料更新"
```

---

### Task 8: Add Analysis Option Submenus Without Changing Backend Behavior

**Files:**
- Modify: `ki67dtc/gui/main_window.py`
- Test: `tests/test_main_window_layout_contract.py`

- [ ] **Step 1: Add QActionGroup helpers**

Add:

```python
def _selected_action_value(self, actions: list[QtGui.QAction], default: str) -> str:
    """取得目前選中的 action value。"""
    for action in actions:
        if action.isChecked():
            return str(action.data())
    return default
```

- [ ] **Step 2: Expand `核來源` into submenu**

In `_build_menu_bar()` replace `self.action_nuc_source = analysis_menu.addAction("核來源")` with:

```python
nuc_menu = analysis_menu.addMenu("核來源")
self.nuc_source_actions = []
for label, value, checked in [("DAPI", "dapi", True), ("PC", "pc", False)]:
    action = nuc_menu.addAction(label)
    action.setCheckable(True)
    action.setChecked(checked)
    action.setData(value)
    self.nuc_source_actions.append(action)
self.action_nuc_source = nuc_menu.menuAction()
```

- [ ] **Step 3: Expand backend and analysis method menus**

Use the same pattern:

```python
ki67_backend_menu = analysis_menu.addMenu("Ki67 Backend")
self.ki67_backend_actions = []
for label, value, checked in [("PyImageJ", "pyimagej", True), ("OpenCV", "opencv", False)]:
    action = ki67_backend_menu.addAction(label)
    action.setCheckable(True)
    action.setChecked(checked)
    action.setData(value)
    self.ki67_backend_actions.append(action)
self.action_ki67_backend = ki67_backend_menu.menuAction()

feature_backend_menu = analysis_menu.addMenu("分析方法")
self.feature_backend_actions = []
for label, value, checked in [("PyImageJ", "pyimagej", True), ("Python", "python", False)]:
    action = feature_backend_menu.addAction(label)
    action.setCheckable(True)
    action.setChecked(checked)
    action.setData(value)
    self.feature_backend_actions.append(action)
self.action_feature_backend = feature_backend_menu.menuAction()
```

Use `QActionGroup` if multiple checked actions appear during manual testing.

- [ ] **Step 4: Read selected values in `_on_run_clicked()`**

```python
nuc_source = self._selected_action_value(self.nuc_source_actions, "dapi")
ki67_backend = self._selected_action_value(self.ki67_backend_actions, "pyimagej")
feature_backend = self._selected_action_value(self.feature_backend_actions, "pyimagej")
```

- [ ] **Step 5: Run menu tests**

Run:

```powershell
python -m unittest tests.test_main_window_layout_contract -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit option menu changes**

```powershell
git add ki67dtc/gui/main_window.py
git commit -m "feat(ui): 補上分析選項子選單"
```

---

### Task 9: Manual Visual Verification

**Files:**
- No production file changes unless visual bugs are found.

- [ ] **Step 1: Launch UI**

Run:

```powershell
python app.py
```

Expected:

- Window opens.
- Top menu has `檔案` and `分析選項`.
- Left side occupies about 2/3 width.
- Right side occupies about 1/3 width.
- Right side has four equal-height panels.
- Control buttons are centered.

- [ ] **Step 2: Verify idle button state**

Expected:

- Start button icon is black on cyan background.
- Stop button is disabled or subdued.
- Restart button is available.

- [ ] **Step 3: Start analysis with a small local dataset**

Expected:

- Terminal panel appends log lines.
- Start button icon turns white while running.
- Stop button becomes available while running.

- [ ] **Step 4: Verify chart panel**

Expected:

- Fourth panel is labeled `細胞面積分析`.
- It shows `Cell Area Distribution (log3)` after outputs exist.
- It does not show Ki67 intensity distribution.

- [ ] **Step 5: Capture screenshot**

Save screenshot manually for review:

```text
C:\Users\B30027\Pictures\Screenshots\ki67-main-ui-redesign-after.png
```

- [ ] **Step 6: Commit visual fix if needed**

Only if manual verification requires spacing or style tweaks:

```powershell
git add ki67dtc/gui/main_window.py ki67dtc/gui/theme.py
git commit -m "style(ui): 微調主畫面間距與控制按鈕狀態"
```

---

### Task 10: Documentation Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add GUI operation notes**

In `README.md`, update the GUI section with:

```markdown
### 圖形介面

```bash
python app.py
```

主畫面採用深色工具介面：

- `檔案 > 開啟`：選擇輸入資料夾。
- `分析選項 > 核來源`：選擇 `DAPI` 或 `PC`。
- `分析選項 > Ki67 Backend`：選擇 Ki67 二值化後端。
- `分析選項 > 分析方法`：選擇特徵提取後端。
- `分析選項 > 螢光分析`：切換螢光分析。
- `分析選項 > Ki67 分析`：切換 Ki67 陽性分析。
- `分析選項 > 清理暫存檔案`：分析完成後清理中間檔。

右側四個區塊依序顯示終端輸出與控制按鈕、資料夾影像清單、特徵參數表與細胞面積分析圖。
```
```

- [ ] **Step 2: Run README sanity check**

Run:

```powershell
Select-String -Path README.md -Pattern '檔案 > 開啟','細胞面積分析'
```

Expected:

```text
README.md:...: - `檔案 > 開啟`：選擇輸入資料夾。
README.md:...: 右側四個區塊依序顯示...
```

- [ ] **Step 3: Commit docs**

```powershell
git add README.md
git commit -m "docs(ui): 補充新版圖形介面操作說明"
```

---

## Verification Commands

Run after implementation:

```powershell
python -m unittest tests.test_main_window_layout_contract -v
python -m unittest tests.test_python_feature_backend -v
python app.py
```

Expected:

- UI contract tests pass.
- Existing feature backend tests pass.
- `python app.py` launches the redesigned main UI.

## Execution Notes

- 建議在 isolated worktree 實作，避免影響目前 `main`。
- 若使用 project-local `.worktrees/`，先確認 `.gitignore` 已忽略該資料夾。
- 每個 task 完成後 commit，commit message 使用繁體中文 subject，type/scope 保持 AngularJS convention。

Plan complete and ready for execution choice.
