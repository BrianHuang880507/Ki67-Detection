# GUI Redesign + Theme Switcher — Design Spec
**Date:** 2026-05-14  
**Project:** Cell Image Analysis GUI v2  
**Scope:** Visual redesign (dark theme, ITRI branding), paired/independent overlay mode toggle, three-theme switcher with persistence.

---

## 1. Context

The existing GUI (`CellimageSegmentation_v6.py` + `controller.py`) uses default Qt palette colours — no intentional visual design. Three things are being addressed together:

1. **Overlay rendering split** — show all detected cells independently (default) or only paired cyto+nuc matches (optional, existing science still works).
2. **Visual redesign** — ITRI-branded dark theme, consistent QSS stylesheet, institutional logo in UI and title bar.
3. **Theme switcher** — three pre-designed themes selectable at runtime, choice persists across app restarts.

---

## 2. Overlay Mode Toggle

### Behaviour
- Default mode: **All Cells** — renders every detected cytoplasm region and every detected nucleus region independently, no pairing required. Maximises visible cell count when one compartment fails to segment.
- Optional mode: **Paired Only** — original behaviour, only renders cells where both cyto and nuc were matched by Shapely spatial containment.
- Switching modes reloads the pre-rendered overlay for the current image instantly (no re-render on navigation).

### UI placement
A `QGroupBox` labelled **OVERLAY MODE** sits between the Analyze row (row 4) and the Cell Number row (row 6) in `gridLayout_2`. It contains two `QRadioButton`s in an `QHBoxLayout`: `overlayAllButton` ("All Cells", default checked) and `overlayPairedButton` ("Paired Only").

### Pre-rendering
`render_and_save_overlay(file_path, overlay_dir)` saves **two** files per image after analysis:
- `<stem>_overlay_indep.npy` — all-cells render
- `<stem>_overlay_paired.npy` — paired-only render

`get_overlay_path(file_path, overlay_dir, paired=False)` returns the appropriate path.  
`apply_overlay(overlay, cyto_mask, nuc_mask, paired=False)` dispatches to `_apply_independent()` or `_apply_paired()`.

`_apply_paired()` and `find_paired_labels()` are preserved unchanged for the paired path.

---

## 3. Visual Redesign

### Design direction
**Dark Lab** — scientific instrument aesthetic. Dark navy base, ITRI cyan accents, Consolas for numeric readouts. Inspired directly by the ITRI logo palette (cyan `#00AEEF`, charcoal `#3C3C3C`).

### ITRI branding
- **Window icon**: `setWindowIcon(QIcon("itri_EL_C-png/itri_EL_C.png"))` — appears in taskbar and title bar.
- **Logo badge**: `itriLogoLabel` (QLabel, `autoFillBackground=True`, `fixedHeight=56`) at row 0 of `gridLayout_2`, spanning all 4 columns. QSS gives it a white rounded background so the logo renders cleanly on the dark panel. Pixmap loaded via `get_resource_path()` in `__init__`, scaled to 48px height.

### QSS strategy
A single stylesheet string (`_QSS_DARK`) applied via `self.setStyleSheet(...)` on the `QMainWindow`. Covers all widget types: `QPushButton`, `QRadioButton`, `QCheckBox`, `QGroupBox`, `QSlider`, `QLineEdit`, `QLCDNumber`, `QScrollBar`, `QScrollArea`, `QProgressBar`, `QMessageBox`. Widget-level palettes set by the auto-generated UI file are overridden by the stylesheet.

**Named widget overrides** (using `#objectName` selectors):
- `QPushButton#SegmentButton` — ITRI cyan fill, bold, large — primary action
- `QPushButton#BrowseFileButton` — transparent with cyan border
- `QPushButton#SaveFileButton` — transparent with teal `#00D4B4` border
- `QLCDNumber#CellNumberShow` — dark inset, cyan digit colour
- `QLabel#SlideNumber` — Consolas font, cyan on dark inset
- `QLabel#itriLogoLabel` — white background pill

---

## 4. Three-Theme System

### Themes
| Name | Key | Base bg | Accent | Secondary accent |
|------|-----|---------|--------|-----------------|
| Dark Lab | `"dark"` | `#1C2030` | `#00AEEF` (ITRI cyan) | `#00D4B4` (teal) |
| Clean Lab | `"clean"` | `#F4F6F9` | `#0090CC` (ITRI blue) | `#00A890` (green-teal) |
| Blue Steel | `"steel"` | `#2C3347` | `#5B8BE8` (blue-violet) + `#00AEEF` | `#3BD4C0` |

Three QSS constants at the top of `controller.py`: `_QSS_DARK`, `_QSS_CLEAN`, `_QSS_STEEL`.  
A module-level dict: `THEMES = {"dark": _QSS_DARK, "clean": _QSS_CLEAN, "steel": _QSS_STEEL}`.

### View menu
The existing empty `menubar` in `CellimageSegmentation_v6.py` is populated with a single **View** menu.  
The View menu contains a `QActionGroup` (exclusive) with three checkable `QAction`s:
- `actionDarkLab` — "Dark Lab" (default checked)
- `actionCleanLab` — "Clean Lab"
- `actionBlueSteelLab` — "Blue Steel"

Wired in `setup_control()`: each action's `triggered` signal connects to `_apply_theme("dark"/"clean"/"steel")`.

### `_apply_theme(name: str)`
```
def _apply_theme(self, name):
    self.setStyleSheet(THEMES[name])
    self._save_config(name)
    # sync menu checkmarks
    self.ui.actionDarkLab.setChecked(name == "dark")
    self.ui.actionCleanLab.setChecked(name == "clean")
    self.ui.actionBlueSteelLab.setChecked(name == "steel")
```

### Persistence
`CONFIG_PATH = "./config.json"` — written relative to the app working directory.  

**On startup** (in `__init__`, before `setup_control`):
```
theme = "dark"
if os.path.exists(CONFIG_PATH):
    try:
        theme = json.load(open(CONFIG_PATH)).get("theme", "dark")
        if theme not in THEMES:
            theme = "dark"
    except Exception:
        pass
self._apply_theme(theme)
```

**On change** (`_save_config`):
```
def _save_config(self, theme):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump({"theme": theme}, f)
    except Exception:
        pass
```

Failure to read or write config is silently ignored — the app always has a valid theme.

---

## 5. Files Changed

| File | Change |
|------|--------|
| `overlay_utils.py` | `_apply_independent()`, `apply_overlay(paired=False)`, `get_overlay_path(paired=False)`, `render_and_save_overlay()` saves both npy variants |
| `CellimageSegmentation_v6.py` | Row shifts (+1 for logo), `itriLogoLabel` row 0, `overlayModeGroupBox` row 5, View menu + 3 QActions in menubar |
| `controller.py` | `_QSS_DARK/CLEAN/STEEL` constants, `THEMES` dict, `CONFIG_PATH`, `_apply_theme()`, `_save_config()`, `_on_overlay_mode_changed()`, startup config load, window icon + logo label load |

`ki67dtc/`, `main_test2.py`, `start.py`, `start.spec`, `VCsetup.iss` — unchanged.

---

## 6. Out of Scope

- Light/dark auto-detection from Windows system preference
- Per-widget theme customisation
- Font size or scale factor settings
- Packaging changes for the new `itri_EL_C-png/` asset (already in working directory; spec file update is a separate task)
