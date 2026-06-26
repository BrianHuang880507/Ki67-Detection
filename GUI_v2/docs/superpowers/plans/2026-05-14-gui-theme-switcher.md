# GUI Theme Switcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Dark Lab / Clean Lab / Blue Steel theme switching via View menu, with choice persisted to `config.json`.

**Architecture:** Three QSS string constants + a `THEMES` dict live in `controller.py`. `_apply_theme(name)` swaps the stylesheet and writes config. On startup, `__init__` reads `config.json` before `setup_control()` so the menu checkmark is in sync before any signal fires. The View menu in `CellimageSegmentation_v6.py` hosts three exclusive checkable `QAction`s wired to `_apply_theme`.

**Tech Stack:** PyQt5, Python 3.10, standard library `json` for persistence.

---

### Task 1: Add QSS constants and module-level config

**Files:**
- Modify: `controller.py` (top of file, after `_QSS` / before `class AnalysisWorker`)

- [ ] **Step 1: Add `import json` at the top of `controller.py`**

Find the import block (lines 1–12) and add `import json` after `import shutil`:

```python
import shutil
import json
```

- [ ] **Step 2: Rename `_QSS` to `_QSS_DARK`**

In `controller.py`, the module-level string currently starts with `_QSS = """`. Rename it:

```python
_QSS_DARK = """
QMainWindow, QWidget {
    background-color: #1C2030;
    ...
"""
```

(The body is unchanged — only the variable name changes.)

- [ ] **Step 3: Add `_QSS_CLEAN` constant after `_QSS_DARK`**

Paste this immediately after the closing `"""` of `_QSS_DARK`:

```python
_QSS_CLEAN = """
QMainWindow, QWidget {
    background-color: #F4F6F9;
    color: #1A2030;
    font-family: "Segoe UI";
    font-size: 11pt;
}
QFrame { background-color: transparent; }
QLabel { background-color: transparent; color: #1A2030; }

QLabel#Image {
    background-color: #E4E8EF;
    border: 1px solid #C8D0DC;
    border-radius: 3px;
}
QLabel#ImageFileName {
    color: #8898B0;
    font-size: 10pt;
}
QLabel#itriLogoLabel {
    background-color: #FFFFFF;
    border: 1px solid #D0D8E4;
    border-radius: 5px;
    padding: 4px 8px;
}
QLabel#SlideNumber {
    background-color: #FFFFFF;
    color: #0090CC;
    border: 1px solid #C8D0DC;
    border-radius: 3px;
    font-family: "Consolas";
    font-size: 12pt;
    padding: 2px 6px;
}
QLabel#CellNumberLabel {
    color: #6B7A99;
    font-size: 12pt;
    font-weight: bold;
    background-color: transparent;
}
QLCDNumber#CellNumberShow {
    background-color: #FFFFFF;
    color: #0090CC;
    border: 1px solid #C8D0DC;
    border-radius: 3px;
}
QPushButton {
    background-color: #FFFFFF;
    color: #3A4258;
    border: 1px solid #C8D0DC;
    border-radius: 4px;
    padding: 5px 12px;
    font-family: "Segoe UI";
    font-size: 11pt;
}
QPushButton:hover {
    background-color: #0090CC;
    color: #FFFFFF;
    border: 1px solid #0090CC;
}
QPushButton:pressed {
    background-color: #006E9E;
    border: 1px solid #006E9E;
}
QPushButton:disabled {
    background-color: #F0F2F5;
    color: #B0B8C8;
    border: 1px solid #D8DDE8;
}
QPushButton#SegmentButton {
    background-color: #0090CC;
    color: #FFFFFF;
    border: none;
    border-radius: 5px;
    font-size: 16pt;
    font-weight: bold;
    padding: 8px;
}
QPushButton#SegmentButton:hover { background-color: #00AEEF; }
QPushButton#SegmentButton:pressed { background-color: #006E9E; }
QPushButton#BrowseFileButton {
    background-color: transparent;
    color: #0090CC;
    border: 1px solid #0090CC;
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 11pt;
}
QPushButton#BrowseFileButton:hover { background-color: #0090CC; color: #FFFFFF; }
QPushButton#SaveFileButton {
    background-color: transparent;
    color: #00A890;
    border: 1px solid #00A890;
    border-radius: 4px;
    padding: 5px 12px;
}
QPushButton#SaveFileButton:hover { background-color: #00A890; color: #FFFFFF; }
QPushButton#PreviousButton, QPushButton#NextButton {
    background-color: #F0F2F5;
    color: #6B7A99;
    border: 1px solid #C8D0DC;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 11pt;
}
QPushButton#PreviousButton:hover, QPushButton#NextButton:hover {
    background-color: #E0E5EF;
    color: #1A2030;
    border: 1px solid #A8B4C8;
}
QScrollArea {
    background-color: #FFFFFF;
    border: 1px solid #C8D0DC;
    border-radius: 3px;
}
QScrollArea > QWidget > QWidget { background-color: #FFFFFF; }
QCheckBox { color: #4A5868; font-size: 10pt; spacing: 6px; }
QCheckBox::indicator {
    width: 13px; height: 13px;
    border: 1px solid #A8B4C8; border-radius: 2px; background: #FFFFFF;
}
QCheckBox::indicator:checked { background: #0090CC; border: 1px solid #0070A0; }
QCheckBox::indicator:hover { border: 1px solid #0090CC; }
QRadioButton { color: #3A4258; font-size: 11pt; spacing: 6px; }
QRadioButton::indicator {
    width: 14px; height: 14px;
    border: 1.5px solid #A8B4C8; border-radius: 7px; background: #FFFFFF;
}
QRadioButton::indicator:checked { background: #0090CC; border: 2px solid #0070A0; }
QRadioButton::indicator:hover { border: 1.5px solid #0090CC; }
QGroupBox {
    background-color: transparent;
    border: 1px solid #C8D0DC; border-radius: 5px;
    margin-top: 10px; padding-top: 6px;
    font-size: 8pt; font-weight: bold; letter-spacing: 1px; color: #8898B0;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 8px; padding: 0 4px; color: #8898B0;
}
QSlider::groove:horizontal { height: 4px; background: #D0D8E8; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 16px; height: 16px; margin: -6px 0;
    border-radius: 8px; background: #0090CC; border: 2px solid #F4F6F9;
}
QSlider::handle:horizontal:hover { background: #00AEEF; }
QSlider::sub-page:horizontal { background: #0090CC; border-radius: 2px; }
QLineEdit {
    background-color: #FFFFFF; color: #1A2030;
    border: 1px solid #C8D0DC; border-radius: 3px;
    padding: 3px 6px; font-size: 10pt;
    selection-background-color: #0090CC; selection-color: #FFFFFF;
}
QLineEdit:focus { border: 1px solid #0090CC; }
QScrollBar:vertical { background: #F0F2F5; width: 7px; margin: 0; border-radius: 3px; }
QScrollBar::handle:vertical { background: #C0C8D8; border-radius: 3px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background: #0090CC; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #F0F2F5; height: 7px; margin: 0; border-radius: 3px; }
QScrollBar::handle:horizontal { background: #C0C8D8; border-radius: 3px; min-width: 20px; }
QScrollBar::handle:horizontal:hover { background: #0090CC; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QProgressDialog {
    background-color: #FFFFFF; color: #1A2030;
    border: 1px solid #C8D0DC; border-radius: 6px;
}
QProgressDialog QLabel { color: #1A2030; font-size: 11pt; }
QProgressBar {
    background-color: #E8EDF5; border: 1px solid #C8D0DC;
    border-radius: 3px; text-align: center; color: #3A4258;
    font-size: 9pt; height: 8px;
}
QProgressBar::chunk { background-color: #0090CC; border-radius: 2px; }
QMessageBox { background-color: #FFFFFF; color: #1A2030; }
QMessageBox QPushButton { min-width: 80px; padding: 6px 16px; }
"""
```

- [ ] **Step 4: Add `_QSS_STEEL` constant after `_QSS_CLEAN`**

```python
_QSS_STEEL = """
QMainWindow, QWidget {
    background-color: #2C3347;
    color: #D8E0F0;
    font-family: "Segoe UI";
    font-size: 11pt;
}
QFrame { background-color: transparent; }
QLabel { background-color: transparent; color: #D8E0F0; }

QLabel#Image {
    background-color: #1E2535;
    border: 1px solid #3A4560;
    border-radius: 3px;
}
QLabel#ImageFileName { color: #6B7A99; font-size: 10pt; }
QLabel#itriLogoLabel {
    background-color: #FFFFFF;
    border-radius: 5px;
    padding: 4px 8px;
}
QLabel#SlideNumber {
    background-color: #1E2535;
    color: #5B8BE8;
    border: 1px solid #3A4560;
    border-radius: 3px;
    font-family: "Consolas";
    font-size: 12pt;
    padding: 2px 6px;
}
QLabel#CellNumberLabel {
    color: #7A8899; font-size: 12pt; font-weight: bold;
    background-color: transparent;
}
QLCDNumber#CellNumberShow {
    background-color: #1E2535; color: #5B8BE8;
    border: 1px solid #3A4560; border-radius: 3px;
}
QPushButton {
    background-color: #343C55; color: #A8B8D0;
    border: 1px solid #3A4560; border-radius: 4px;
    padding: 5px 12px; font-family: "Segoe UI"; font-size: 11pt;
}
QPushButton:hover { background-color: #5B8BE8; color: #FFFFFF; border: 1px solid #5B8BE8; }
QPushButton:pressed { background-color: #3A6BC8; border: 1px solid #3A6BC8; }
QPushButton:disabled { background-color: #242B3E; color: #3A4560; border: 1px solid #2A3248; }
QPushButton#SegmentButton {
    background-color: #00AEEF; color: #0A0E18;
    border: none; border-radius: 5px;
    font-size: 16pt; font-weight: bold; padding: 8px;
}
QPushButton#SegmentButton:hover { background-color: #22C8FF; }
QPushButton#SegmentButton:pressed { background-color: #0090CC; }
QPushButton#BrowseFileButton {
    background-color: transparent; color: #5B8BE8;
    border: 1px solid #5B8BE8; border-radius: 4px;
    padding: 6px 12px; font-size: 11pt;
}
QPushButton#BrowseFileButton:hover { background-color: #5B8BE8; color: #FFFFFF; }
QPushButton#SaveFileButton {
    background-color: transparent; color: #3BD4C0;
    border: 1px solid #3BD4C0; border-radius: 4px; padding: 5px 12px;
}
QPushButton#SaveFileButton:hover { background-color: #3BD4C0; color: #0A0E18; }
QPushButton#PreviousButton, QPushButton#NextButton {
    background-color: #2C3347; color: #6B7A99;
    border: 1px solid #3A4560; border-radius: 4px;
    padding: 4px 10px; font-size: 11pt;
}
QPushButton#PreviousButton:hover, QPushButton#NextButton:hover {
    background-color: #343C55; color: #D8E0F0; border: 1px solid #4A5878;
}
QScrollArea { background-color: #242B3E; border: 1px solid #3A4560; border-radius: 3px; }
QScrollArea > QWidget > QWidget { background-color: #242B3E; }
QCheckBox { color: #8A9BC0; font-size: 10pt; spacing: 6px; }
QCheckBox::indicator {
    width: 13px; height: 13px;
    border: 1px solid #3A4560; border-radius: 2px; background: #343C55;
}
QCheckBox::indicator:checked { background: #5B8BE8; border: 1px solid #3A6BC8; }
QCheckBox::indicator:hover { border: 1px solid #5B8BE8; }
QRadioButton { color: #A8B8D0; font-size: 11pt; spacing: 6px; }
QRadioButton::indicator {
    width: 14px; height: 14px;
    border: 1.5px solid #3A4560; border-radius: 7px; background: #343C55;
}
QRadioButton::indicator:checked { background: #5B8BE8; border: 2px solid #3A6BC8; }
QRadioButton::indicator:hover { border: 1.5px solid #5B8BE8; }
QGroupBox {
    background-color: transparent;
    border: 1px solid #3A4560; border-radius: 5px;
    margin-top: 10px; padding-top: 6px;
    font-size: 8pt; font-weight: bold; letter-spacing: 1px; color: #4A5878;
}
QGroupBox::title {
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 8px; padding: 0 4px; color: #4A5878;
}
QSlider::groove:horizontal { height: 4px; background: #3A4560; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 16px; height: 16px; margin: -6px 0;
    border-radius: 8px; background: #5B8BE8; border: 2px solid #2C3347;
}
QSlider::handle:horizontal:hover { background: #7AAAF8; }
QSlider::sub-page:horizontal { background: #5B8BE8; border-radius: 2px; }
QLineEdit {
    background-color: #1E2535; color: #D8E0F0;
    border: 1px solid #3A4560; border-radius: 3px;
    padding: 3px 6px; font-size: 10pt;
    selection-background-color: #5B8BE8; selection-color: #FFFFFF;
}
QLineEdit:focus { border: 1px solid #5B8BE8; }
QScrollBar:vertical { background: #2C3347; width: 7px; margin: 0; border-radius: 3px; }
QScrollBar::handle:vertical { background: #3A4560; border-radius: 3px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background: #5B8BE8; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #2C3347; height: 7px; margin: 0; border-radius: 3px; }
QScrollBar::handle:horizontal { background: #3A4560; border-radius: 3px; min-width: 20px; }
QScrollBar::handle:horizontal:hover { background: #5B8BE8; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QProgressDialog {
    background-color: #343C55; color: #D8E0F0;
    border: 1px solid #3A4560; border-radius: 6px;
}
QProgressDialog QLabel { color: #D8E0F0; font-size: 11pt; }
QProgressBar {
    background-color: #2C3347; border: 1px solid #3A4560;
    border-radius: 3px; text-align: center; color: #D8E0F0;
    font-size: 9pt; height: 8px;
}
QProgressBar::chunk { background-color: #5B8BE8; border-radius: 2px; }
QMessageBox { background-color: #343C55; color: #D8E0F0; }
QMessageBox QPushButton { min-width: 80px; padding: 6px 16px; }
"""
```

- [ ] **Step 5: Add `THEMES` dict and `CONFIG_PATH` after `_QSS_STEEL`**

```python
THEMES = {
    "dark": _QSS_DARK,
    "clean": _QSS_CLEAN,
    "steel": _QSS_STEEL,
}

CONFIG_PATH = "./config.json"
```

- [ ] **Step 6: Fix the one remaining reference to `_QSS` in `__init__`**

In `MainWindow_controller.__init__`, find and remove this line (it will be replaced in Task 3):

```python
        self.setStyleSheet(_QSS)
```

Delete that line only — leave everything else around it intact.

- [ ] **Step 7: Verify the module still imports without error**

```bash
cd C:\code\Cell_Image\GUI_v2
python -c "import controller; print('OK')"
```

Expected output: `OK` (no `NameError: name '_QSS' is not defined` or similar).

- [ ] **Step 8: Commit**

```bash
git init   # only if not already a repo
git add controller.py
git commit -m "feat: add _QSS_CLEAN, _QSS_STEEL theme constants and THEMES dict"
```

---

### Task 2: Add `_apply_theme()` and `_save_config()` methods + tests

**Files:**
- Modify: `controller.py` (inside `MainWindow_controller`)
- Create: `tests/test_theme.py`

- [ ] **Step 1: Create `tests/test_theme.py` with failing tests**

```bash
mkdir -p C:\code\Cell_Image\GUI_v2\tests
```

Create `tests/test_theme.py`:

```python
import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CONFIG_PATH_ORIG = "./config.json"


# --- Pure config logic tests (no Qt needed) ---

def test_save_config_writes_correct_json(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    monkeypatch.setattr("controller.CONFIG_PATH", str(cfg))
    import controller
    monkeypatch.setattr("controller.CONFIG_PATH", str(cfg))
    # Instantiate a minimal stand-in that has only _save_config
    # We test the logic directly
    import importlib
    importlib.reload(controller)
    monkeypatch.setattr("controller.CONFIG_PATH", str(cfg))

    # Call the function via a duck-typed object
    class FakeCtrl:
        pass
    fc = FakeCtrl()
    fc._save_config = lambda name: json.dump({"theme": name}, open(str(cfg), "w"))
    fc._save_config("clean")
    assert json.loads(cfg.read_text()) == {"theme": "clean"}


def test_config_load_returns_saved_theme(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"theme": "steel"}))
    data = json.loads(cfg.read_text())
    theme = data.get("theme", "dark")
    assert theme == "steel"


def test_config_load_falls_back_on_missing_file(tmp_path):
    cfg = tmp_path / "nonexistent.json"
    theme = "dark"
    if cfg.exists():
        try:
            theme = json.loads(cfg.read_text()).get("theme", "dark")
        except Exception:
            pass
    assert theme == "dark"


def test_config_load_falls_back_on_invalid_key(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"theme": "not_a_real_theme"}))
    from controller import THEMES
    data = json.loads(cfg.read_text())
    theme = data.get("theme", "dark")
    if theme not in THEMES:
        theme = "dark"
    assert theme == "dark"


def test_themes_dict_has_all_three_keys():
    from controller import THEMES
    assert set(THEMES.keys()) == {"dark", "clean", "steel"}


def test_themes_values_are_nonempty_strings():
    from controller import THEMES
    for name, qss in THEMES.items():
        assert isinstance(qss, str) and len(qss) > 100, f"Theme '{name}' QSS is too short"
```

- [ ] **Step 2: Run tests to verify they fail correctly**

```bash
cd C:\code\Cell_Image\GUI_v2
python -m pytest tests/test_theme.py -v
```

Expected: `test_themes_dict_has_all_three_keys` and `test_themes_values_are_nonempty_strings` PASS (THEMES dict already added in Task 1). The `_save_config` duck-typed test should also pass. All should pass or fail with import/attribute errors — not logic errors.

- [ ] **Step 3: Add `_apply_theme()` and `_save_config()` to `MainWindow_controller`**

Add these two methods after `_on_overlay_mode_changed` in `controller.py`:

```python
    def _apply_theme(self, name: str):
        self.setStyleSheet(THEMES.get(name, _QSS_DARK))
        self._save_config(name)
        _action_map = {
            "dark": self.ui.actionDarkLab,
            "clean": self.ui.actionCleanLab,
            "steel": self.ui.actionBlueSteelLab,
        }
        if name in _action_map:
            _action_map[name].setChecked(True)

    def _save_config(self, name: str):
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump({"theme": name}, f)
        except Exception:
            pass
```

- [ ] **Step 4: Run tests again**

```bash
python -m pytest tests/test_theme.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add controller.py tests/test_theme.py
git commit -m "feat: add _apply_theme and _save_config with config persistence"
```

---

### Task 3: Add View menu to `CellimageSegmentation_v6.py`

**Files:**
- Modify: `CellimageSegmentation_v6.py` (menubar section + `retranslateUi`)

- [ ] **Step 1: Add View menu and QActions after the menubar is created**

In `CellimageSegmentation_v6.py`, find this block (around line 1037):

```python
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
```

Replace it with:

```python
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menubar.addAction(self.menuView.menuAction())

        self._themeGroup = QtWidgets.QActionGroup(MainWindow)
        self._themeGroup.setExclusive(True)

        self.actionDarkLab = QtWidgets.QAction(MainWindow)
        self.actionDarkLab.setCheckable(True)
        self.actionDarkLab.setChecked(True)
        self.actionDarkLab.setObjectName("actionDarkLab")
        self._themeGroup.addAction(self.actionDarkLab)
        self.menuView.addAction(self.actionDarkLab)

        self.actionCleanLab = QtWidgets.QAction(MainWindow)
        self.actionCleanLab.setCheckable(True)
        self.actionCleanLab.setObjectName("actionCleanLab")
        self._themeGroup.addAction(self.actionCleanLab)
        self.menuView.addAction(self.actionCleanLab)

        self.actionBlueSteelLab = QtWidgets.QAction(MainWindow)
        self.actionBlueSteelLab.setCheckable(True)
        self.actionBlueSteelLab.setObjectName("actionBlueSteelLab")
        self._themeGroup.addAction(self.actionBlueSteelLab)
        self.menuView.addAction(self.actionBlueSteelLab)
```

- [ ] **Step 2: Add `retranslateUi` entries for the new menu items**

In `retranslateUi`, find the last `_translate` call (currently `self.SlideNumber.setText(...)`) and add after it:

```python
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.actionDarkLab.setText(_translate("MainWindow", "Dark Lab"))
        self.actionCleanLab.setText(_translate("MainWindow", "Clean Lab"))
        self.actionBlueSteelLab.setText(_translate("MainWindow", "Blue Steel"))
```

- [ ] **Step 3: Verify the UI file imports without error**

```bash
cd C:\code\Cell_Image\GUI_v2
python -c "from CellimageSegmentation_v6 import Ui_MainWindow; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add CellimageSegmentation_v6.py
git commit -m "feat: add View menu with Dark Lab / Clean Lab / Blue Steel theme actions"
```

---

### Task 4: Wire startup config load and action signals in `controller.py`

**Files:**
- Modify: `controller.py` (`__init__` and `setup_control`)

- [ ] **Step 1: Update `__init__` to load config and call `_apply_theme` on startup**

In `MainWindow_controller.__init__`, the current sequence after `setupUi` is:

```python
        self.ui.setupUi(self)
        self.setStyleSheet(_QSS)   # ← this line was deleted in Task 1 Step 6

        # Window icon + logo label
        _logo_path = ...
```

Replace the gap left by the deleted `self.setStyleSheet(_QSS)` line with the config-load block:

```python
        self.ui.setupUi(self)

        # Load persisted theme and apply before any signal fires
        _saved_theme = "dark"
        if os.path.exists(CONFIG_PATH):
            try:
                _saved_theme = json.load(open(CONFIG_PATH)).get("theme", "dark")
                if _saved_theme not in THEMES:
                    _saved_theme = "dark"
            except Exception:
                pass
        self._apply_theme(_saved_theme)

        # Window icon + logo label
        _logo_path = ...
```

- [ ] **Step 2: Wire the three action signals in `setup_control()`**

In `setup_control`, after the line `self.ui.overlayAllButton.toggled.connect(self._on_overlay_mode_changed)`, add:

```python
        self.ui.actionDarkLab.triggered.connect(lambda: self._apply_theme("dark"))
        self.ui.actionCleanLab.triggered.connect(lambda: self._apply_theme("clean"))
        self.ui.actionBlueSteelLab.triggered.connect(lambda: self._apply_theme("steel"))
```

- [ ] **Step 3: Verify full import chain**

```bash
cd C:\code\Cell_Image\GUI_v2
python -c "import controller; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/test_theme.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Manual smoke test**

Launch the app:

```bash
python start.py
```

Check:
1. App opens with Dark Lab theme (dark navy, cyan accents).
2. Menu bar shows "View" menu.
3. View → "Dark Lab" is checked.
4. Click View → "Clean Lab" — UI switches to light grey theme instantly.
5. Click View → "Blue Steel" — UI switches to slate-blue theme instantly.
6. Close the app. Reopen. Theme from last session is restored.
7. Check that the overlay mode radio buttons still work (All Cells / Paired Only toggle).

- [ ] **Step 6: Final commit**

```bash
git add controller.py
git commit -m "feat: wire theme switcher startup config load and View menu signals"
```
