# Cell Image Analysis GUI v2 — Project Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [File Relationships & Architecture](#3-file-relationships--architecture)
4. [Module Reference](#4-module-reference)
5. [UI Layout](#5-ui-layout)
6. [Data Flow](#6-data-flow)
7. [Configuration & Defaults](#7-configuration--defaults)
8. [Environment & Dependencies](#8-environment--dependencies)
9. [Packaging & Distribution](#9-packaging--distribution)
10. [Known Issues & Tips](#10-known-issues--tips)

---

## 1. Project Overview

A desktop GUI application for biomedical microscopy image analysis. Given phase-contrast cell images, the app:

- Segments cytoplasm and nucleus using pre-trained Cellpose deep learning models
- Computes 13 morphological parameters per cell (area, perimeter, roundness, karyoplasmic ratio, etc.)
- Performs ring-based fluorescence intensity quantification
- Renders color-coded segmentation overlays on the original images
- Generates scatter plots and log-scale area histograms across all images
- Exports merged CSV results and publication-ready figures

**Domain**: Cell biology / pathology research (Ki67 staining, IDO/Ki67 workflow).

**Platform**: Windows x64 desktop.

---

## 2. Directory Structure

```
GUI_v2/
├── start.py                        Entry point — env setup + Qt app launch
├── controller.py                   Main GUI controller (MVC controller layer)
├── CellimageSegmentation_v6.py     Qt UI definition (auto-generated from Qt Designer)
├── main_test2.py                   Analysis pipeline coordinator
├── overlay_utils.py                Overlay rendering & mask visualization
├── requirements.txt                pip dependencies (pinned versions)
├── environment.yml                 Conda environment (Python 3.10, OpenBLAS)
├── start.spec                      PyInstaller spec file
├── VCsetup.iss                     Inno Setup installer script
├── VCsetup.txt                     Notes on VC++ redistributable setup
│
├── ki67dtc/                        Core analysis library package
│   ├── __init__.py
│   ├── img_prep.py                 Cellpose segmentation + outline conversion
│   ├── cell_anal.py                Morphology & fluorescence parameter calculation
│   ├── cell_anal_plot.py           Matplotlib plot generation
│   └── utils/
│       ├── __init__.py
│       └── io.py                   File I/O, CSV merging, data utilities
│
├── model/                          Pre-trained Cellpose model files (do not modify)
│   ├── model_BDL6_label_new        Cytoplasm segmentation model
│   └── model_BDL3_label_dapi       Nucleus segmentation model
│
├── data2/                          Sample dataset (input + output)
│   ├── input/
│   └── output/
│       ├── results/                Per-image and combined CSV files
│       ├── figure/                 Scatter plots and histograms (PNG)
│       ├── outline/                Cell outline coordinate text files
│       └── overlays/               Pre-rendered overlay arrays (.npy)
│
├── dist/start/                     PyInstaller output (distributable)
└── build/start/                    PyInstaller build cache
```

**Legacy files** (kept for reference, not active):
- `controller_old.py`, `controller_v2.py` — earlier controller versions
- `CellimageSegmentation_v2-5.py` — earlier UI versions
- `ki67dtc/img_prep_old.py`, `ki67dtc/cell_anal_plot_old.py`

---

## 3. File Relationships & Architecture

The application follows an MVC pattern:

```
start.py
  └─ creates MainWindow_controller (controller.py)
       │
       ├── View: Ui_MainWindow (CellimageSegmentation_v6.py)
       │         Qt widgets, layouts, signals
       │
       ├── Analysis thread: AnalysisWorker(QThread)
       │     └─ analyze_cell() ──── main_test2.py
       │           ├── segment_all()       ki67dtc/img_prep.py
       │           ├── mask2txt_all()      ki67dtc/img_prep.py
       │           ├── combined()          ki67dtc/img_prep.py
       │           ├── run_all()           ki67dtc/cell_anal.py
       │           │     ├── param_anal()
       │           │     ├── flour_anal()
       │           │     └── merged_excel()  ki67dtc/utils/io.py
       │           ├── plot_global_area_analysis()  ki67dtc/cell_anal_plot.py
       │           └── render_and_save_overlay()    overlay_utils.py
       │
       └── Display functions (run on main thread after analysis)
             ├── show_image()       → overlay_utils.apply_overlay()
             ├── slider()           → ki67dtc/cell_anal_plot.plot_global_area_analysis()
             └── save_results()     → copies files to user-chosen directory
```

**Import chain** (no circular imports):
```
start.py → controller.py → CellimageSegmentation_v6.py
                         → main_test2.py → ki67dtc/* → ki67dtc/utils/io.py
                         → overlay_utils.py
```

---

## 4. Module Reference

### `start.py` — Entry Point

Sets up the runtime environment before any Qt or ML import.

| Step | What it does |
|------|-------------|
| Frozen detection | Checks `sys.frozen` to detect PyInstaller mode; adds `_MEIPASS` to `PATH` for DLL loading |
| Conda DLL path | Adds `sys.prefix/Library/bin` to `PATH` so native conda libraries are found |
| OpenMP tolerance | Sets `KMP_DUPLICATE_LIB_OK=TRUE` before torch import to avoid libiomp5 crash |
| Qt app | Creates `QApplication`, instantiates `MainWindow_controller`, calls `app.exec_()` |

> **Tip**: `KMP_DUPLICATE_LIB_OK` must be set before `import torch` — do it at the top of `start.py` or in the spec's runtime hook, never inside controller.py.

---

### `controller.py` — GUI Controller

**Class `AnalysisWorker(QThread)`**

Runs `analyze_cell()` on a background thread. Emits:
- `progress(int)` — percentage complete (0–100)
- `status(str)` — human-readable step description
- `finished()` — analysis done
- `error(str)` — exception message

**Class `MainWindow_controller(QMainWindow)`**

| Method | Trigger | Description |
|--------|---------|-------------|
| `open_files()` | Browse button | QFileDialog → populate checkbox list |
| `run_analysis()` | Analyze button | Start AnalysisWorker; show progress dialog |
| `analysis_finished()` | Worker `finished` signal | Load first image, enable controls |
| `show_image(idx)` | Navigation / load | Load image + overlay from cache, display |
| `toggle_mask()` | Mask radio button | Show image with or without overlay |
| `slider()` | Slider move | Debounce 200ms → `_update_slider_plot()` |
| `_update_slider_plot()` | Debounce timer | Re-run `plot_global_area_analysis()` with new threshold |
| `save_results()` | Save button | Copy results/figures/images to user directory |
| `display_image_on_label(arr)` | Internal | Scale numpy RGB → QPixmap → QLabel |

**Image cache**: `self._image_cache[path] = (original_rgb, overlay_rgb)` — avoids re-loading on navigation.

**QSS Themes** — three named themes stored as module-level strings; applied via `_apply_theme(name)`:

| Theme key | Background | Accent | Description |
|-----------|-----------|--------|-------------|
| `"dark"` | `#1C2030` | `#00AEEF` | Dark Lab — dark navy UI |
| `"clean"` | `#F4F6F9` | `#0090CC` | Clean Lab — light grey UI |
| `"steel"` | `#2C3347` | `#5B8BE8` | Blue Steel — dark blue-grey UI |

Key per-theme QSS rules:
- `QLabel#ImageFileName` — secondary label, 14 pt; color brightened per theme for legibility at larger size
- `QPushButton#PreviousButton, QPushButton#NextButton` — `min-width: 88px` ensures equal width regardless of text length (root cause: `QSizePolicy.Minimum` inside QSplitter has no natural size floor)
- `widthScaleAppliedLabel` / `heightScaleAppliedLabel` — color set dynamically to theme accent in `_apply_theme()`

---

### `CellimageSegmentation_v6.py` — UI Definition

Auto-generated by Qt Designer. Do **not** hand-edit; regenerate with `pyuic5` if the `.ui` file changes.

Key widgets and their names (referenced in controller.py by attribute):

| Widget name | Type | Role |
|-------------|------|------|
| `Image` | QLabel | Main image display (expandable) |
| `BrowseFileButton` | QPushButton | Open file dialog |
| `SegmentButton` | QPushButton | Launch analysis (18 pt, bold) |
| `PreviousButton` / `NextButton` | QPushButton | Navigate images |
| `ImageFileName` | QLabel | Current file name display |
| `MaskOnButton` | QRadioButton | Toggle overlay |
| `CellNumberShow` | QLCDNumber | Cell count |
| `SaveFileButton` | QPushButton | Export results |
| `horizontalSlider` | QSlider | Area threshold (raw: 30–110, display: ÷10) |
| `SlideNumber` | QLabel | Threshold value text |
| `AreaScatteringPlot` | QLabel | Cytoplasm vs Nucleus scatter plot |
| `AreaScalablePlot` | QLabel | Log-scale area histogram |
| `widthScaleInput` / `heightScaleInput` | QLineEdit | µm/pixel conversion factors |
| `widthScaleAppliedLabel` / `heightScaleAppliedLabel` | QLabel | Show "Applied: —" / "Applied: X.XXXX" after scale is set |
| `scrollArea` + `fileCheckboxContainerLayout` | QScrollArea + QVBoxLayout | Dynamic file checkboxes |

`pixelScaleGroupBox` title: **"Pixel Scale (µm/pixel)"**.

Window size: **1600 × 900 px**.

---

### `main_test2.py` — Analysis Pipeline

**`analyze_cell(input_folder, output_folder, threshold, cyto_model, nuc_model, width_scale, height_scale, progress_callback)`**

Five sequential steps, each calling `progress_callback(pct, msg)`:

| Step | % | Function called | Output |
|------|---|-----------------|--------|
| 1 — Segment | 0–40 | `segment_all()` | `*_cyto_seg.npy`, `*_nuc_seg.npy` |
| 2 — Outline | 40–55 | `mask2txt_all()` | `*_cp_outlines.txt` |
| 3 — Merge | 55–65 | `combined()` | `*_merged_cp_outlines.txt` |
| 4 — Analyze | 65–85 | `run_all()` | `*_final.csv`, `ALL_para_combine.csv` |
| 5 — Plot + Overlay | 85–100 | `plot_global_area_analysis()` + `render_and_save_overlay()` | PNG plots, `*_overlay.npy` |

---

### `ki67dtc/img_prep.py` — Segmentation

| Function | Description |
|----------|-------------|
| `segment(img_path, model_path, is_nucleus)` | Run one Cellpose model; for nucleus resizes to 1280×1024 training size then resizes mask back |
| `segment_all(input_dir, output_dir, cyto_model, nuc_model)` | Process all `.jpg`/`.png`/`.tif` files (skips Ki67/DF fluorescence files) |
| `mask2txt(npy_path, out_dir)` | Convert `.npy` mask array → `.txt` outline coordinates (one polygon per line) |
| `combined(cyto_txt, nuc_txt, out_path)` | Spatially pair nucleus centers to cytoplasm polygons using Shapely `contains()`; write merged outlines |

**GPU**: `segment()` passes `gpu=True` to Cellpose — falls back to CPU if CUDA unavailable.

---

### `ki67dtc/cell_anal.py` — Parameter Calculation

| Function | Description |
|----------|-------------|
| `param_anal(outline_path)` | Parse merged outline → compute 13 metrics per cell (see below) |
| `flour_anal(img_path, mask_path)` | Progressive nucleus dilation → ring-by-ring IntDen/RawIntDen |
| `run_all(input_dir, output_dir)` | Iterate all merged outlines; call param + flour; call `merged_excel()` |

**13 morphological metrics**:

| Metric | Formula |
|--------|---------|
| Area | Polygon area (pixels²) |
| Perimeter | Polygon perimeter |
| Convex Perimeter | Convex hull perimeter |
| Circular Diameter | Equivalent circle diameter |
| Feret Length | Max caliper distance |
| Feret Width | Min caliper distance |
| Aspect Ratio | Feret Length / Feret Width |
| Roundness | 4·Area / (π · FeretLength²) |
| Circularity | 4π·Area / Perimeter² |
| Sphericity | Feret Width / Feret Length |
| Roughness | Convex Perimeter / Perimeter |
| Karyoplasmic Ratio | Nucleus Area / Cytoplasm Area |

Areas are converted from pixels² to µm² using `width_scale × height_scale`.

---

### `ki67dtc/cell_anal_plot.py` — Visualization

| Function | Description |
|----------|-------------|
| `plot_area_analysis(csv_path, out_dir, threshold)` | Per-image scatter + histogram |
| `plot_global_area_analysis(combined_csv, out_dir, threshold, width_scale, height_scale)` | All-images combined scatter + histogram; annotates % cells above threshold |

**Scatter plot**: Cytoplasm Area (µm²) vs Nucleus Area (µm²), log-log scale.  
**Histogram**: Cell area distribution on log₃ scale; vertical red line at threshold; shows `N above / N total`.

---

### `overlay_utils.py` — Overlay Rendering

| Function | Description |
|----------|-------------|
| `render_and_save_overlay(img_path, cyto_npy, nuc_npy, out_path)` | Pre-render and save as `.npy` (call once after analysis) |
| `apply_overlay(orig_rgb, overlay_npy, show_mask)` | Blend pre-rendered overlay onto image; returns RGB array |
| `_apply_paired(img, cyto_mask, nuc_mask)` | Nucleus: red contour + dark-blue fill; cytoplasm: rotating 10-color palette, 30% alpha |
| `_apply_cyto_only(img, cyto_mask)` | Used when nucleus mask is absent |
| `load_mask_file(npy_path, target_size)` | Load `.npy` mask; resize to match image dimensions |
| `find_paired_labels(cyto_mask, nuc_mask)` | Match nucleus label IDs to cytoplasm regions by spatial overlap |

**Overlay color scheme**:
- Nucleus: Red contour `(255,0,0)`, dark-blue fill `(0,0,240)`, alpha 0.3
- Cytoplasm: 10-color rotating palette (red, green, blue, yellow, magenta, cyan, orange, purple, light-blue, lime), alpha 0.3

---

### `ki67dtc/utils/io.py` — Data Utilities

| Function | Description |
|----------|-------------|
| `list_files(dir, ext)` | Natural-sorted file listing |
| `output_dir(base, subdir)` | Create timestamped output subdirectory |
| `load_outlines(txt_path)` | Parse `.txt` outline file → list of coordinate arrays |
| `merged_excel(nuc_csv, cyto_csv, out_path)` | Join per-cell nucleus & cytoplasm CSVs; suffix columns `_nuc`, `_cyto`; add `Nuc_Cyto_Ratio` |
| `flatten_fluor_table(df)` | Long → wide format for fluorescence ring data |
| `merge_with_flour(param_df, fluor_df)` | Combine morphology + fluorescence tables |
| `merge_all_final_csvs(results_dir)` | Concatenate all `*_final.csv` → `ALL_para_combine.csv` |
| `generate_image_mapping(input_dir, results_dir)` | Create metadata CSV linking input images to output CSVs |

---

## 5. UI Layout

```
QMainWindow (1600 × 900)
└── QWidget (centralwidget)
    └── QHBoxLayout
        ├── LEFT — Image panel
        │   ├── QLabel "Image"        (expandable, KeepAspectRatio display)
        │   └── QSplitter (nav row)
        │       ├── QPushButton "PreviousButton"
        │       ├── QLabel "ImageFileName"
        │       └── QPushButton "NextButton"
        │
        └── RIGHT — Controls panel (min width 250 px)
            ├── QPushButton "BrowseFileButton"
            ├── QScrollArea → dynamic QCheckBox list (one per file)
            ├── QPushButton "SegmentButton"  (18pt, prominent)
            ├── QRadioButton "MaskOnButton"
            ├── QLCDNumber "CellNumberShow"
            ├── QPushButton "SaveFileButton"
            ├── QGroupBox "pixelScaleGroupBox"
            │   ├── QLineEdit "widthScaleInput"
            │   ├── QLineEdit "heightScaleInput"
            │   ├── QLabel "widthScaleAppliedLabel"
            │   └── QLabel "heightScaleAppliedLabel"
            ├── QSlider "horizontalSlider"  (raw 30–110 → display ÷10 = 3.0–11.0)
            ├── QLabel "SlideNumber"
            ├── QLabel "AreaScatteringPlot"  (PNG rendered into QPixmap)
            └── QLabel "AreaScalablePlot"    (PNG rendered into QPixmap)
```

---

## 6. Data Flow

### Per-image pipeline

```
Input: image.jpg
  │
  ├─[Cellpose cyto model]─→ image_cyto_seg.npy  (2D label array)
  ├─[Cellpose nuc model] ─→ image_nuc_seg.npy
  │
  ├─[mask2txt] ──────────→ image_cyto_seg_cp_outlines.txt
  │                         image_nuc_seg_cp_outlines.txt
  │
  ├─[combined] ──────────→ image_merged_cp_outlines.txt
  │                         (nucleus polygon paired to its cytoplasm polygon)
  │
  ├─[run_all]
  │   ├─[param_anal] ────→ image_params.csv  (13 metrics × N cells)
  │   ├─[flour_anal] ────→ image_fluorescence.csv  (ring intensity × N cells)
  │   └─[merged_excel] ──→ image_final.csv  (nucleus+cytoplasm columns)
  │
  ├─[plot_area_analysis]─→ image_scatter.png, image_histogram.png
  │
  └─[render_overlay] ────→ image_overlay.npy  (pre-rendered RGB numpy array)
```

### Global summary

```
All image_final.csv files
  └─[merge_all_final_csvs]──→ ALL_para_combine.csv
        └─[plot_global_area_analysis]──→ all_cell_nucleus_area.png
                                         all_log_cell_area_distribution.png
```

### GUI display

```
image.jpg + image_overlay.npy
  └─[apply_overlay(show_mask=True/False)]──→ RGB numpy array
        └─[display_image_on_label]──────────→ QPixmap → QLabel "Image"
```

---

## 7. Configuration & Defaults

All configurable values live in `controller.py` and `main_test2.py`.  
None are stored in external config files — change them in source or via the GUI at runtime.

| Parameter | Default | Location | Notes |
|-----------|---------|----------|-------|
| Cytoplasm model path | `model/model_BDL6_label_new` | `main_test2.py` | Relative to working dir |
| Nucleus model path | `model/model_BDL3_label_dapi` | `main_test2.py` | Relative to working dir |
| Width scale (µm/px) | `1.5896` | `controller.py` | Configurable in GUI |
| Height scale (µm/px) | `1.5876` | `controller.py` | Configurable in GUI |
| Default threshold | `7.0` (log₃ scale) | `controller.py` | Slider default |
| Slider range | 3.0 – 11.0 | `CellimageSegmentation_v6.py` | Raw ×10 internally |
| Slider debounce | 200 ms | `controller.py` | QTimer single-shot |
| Overlay alpha | 0.3 | `overlay_utils.py` | Cytoplasm fill transparency |
| Nucleus resize | 1280 × 1024 | `ki67dtc/img_prep.py` | Training image size |

---

## 8. Environment & Dependencies

### Conda environment (recommended)

```bash
conda env create -f environment.yml
conda activate cell_image_gui
```

Key decisions in `environment.yml`:
- **Python 3.10** — required by Cellpose 3.x and PyQt5 compatibility
- **OpenBLAS** (not MKL) — avoids OpenMP conflicts when PyTorch and scipy both load BLAS
- numpy and scipy installed via conda before pip packages to resolve native library order

### pip dependencies (pinned)

```
PyQt5==5.15.11
numpy==1.26.4
scipy==1.13.1
scikit-image==0.24.0
opencv-python-headless==4.10.0.84   # headless variant — no extra Qt conflict
pandas==2.2.3
tqdm==4.67.1
natsort==8.4.0
shapely==2.0.6
imageio==2.35.1
cellpose==3.1.1.1
Pillow==10.4.0
matplotlib==3.9.2
torch==2.4.1+cu124                  # CUDA 12.4 wheels
torchvision==0.19.1+cu124
```

Install torch separately with the CUDA index URL before `pip install -r requirements.txt`:
```bash
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

> **Why `opencv-python-headless`?** The headless build omits OpenCV's own Qt libraries, preventing conflicts with PyQt5 at runtime. Never mix `opencv-python` and `opencv-python-headless` in the same environment.

---

## 9. Packaging & Distribution

### PyInstaller (`start.spec`)

Build command:
```bash
pyinstaller start.spec
```

Output: `dist/start/start.exe` + all bundled libraries.

**Critical spec settings**:

| Setting | Value | Why |
|---------|-------|-----|
| `console=True` | keep console window | Shows torch/CUDA init logs; useful for debugging; set `False` for clean release if desired |
| `collect_all(pkg)` | all major packages | Ensures binaries, data files, and hidden submodules are all included |
| Model data | `('model', 'model')` in `datas` | Copies `model/` dir into bundle root |
| `KMP_DUPLICATE_LIB_OK` | set in `start.py` before torch import | Prevents crash from duplicate libiomp5 |
| `sys.frozen` check | in `start.py` | Adds `sys._MEIPASS` to `PATH` so bundled DLLs are found |

**Packages requiring `collect_all`** (not just `hiddenimports`):
- `cellpose`, `torch`, `torchvision`, `cv2`, `scipy`, `skimage`, `imageio`, `PyQt5`

If a package fails to collect automatically, add explicit `hiddenimports` entries for its submodules.

### Inno Setup (`VCsetup.iss`)

Build with Inno Setup Compiler to produce `CellImageAnalysisSetup.exe`.

Key sections:
- Installs to `{commonpf64}\CellImageAnalysis` (requires admin)
- Bundles VC++ redistributable (`vc_redist.x64.exe`), runs silently, deletes after install
- Creates Start Menu + Desktop shortcuts pointing to `start.exe`
- Auto-launches app on install finish
- Compression: LZMA2

**Deployment checklist**:
1. `pyinstaller start.spec` → verify `dist/start/start.exe` runs on a clean machine
2. Compile `VCsetup.iss` → `CellImageAnalysisSetup.exe`
3. Test installer on a VM without conda, without Python
4. Verify models are in `dist/start/model/` before compiling installer

---

## 10. Known Issues & Tips

### OpenMP / libiomp5 crash

**Symptom**: App crashes on import with "OMP: Error #15: Initializing libiomp5md.dll..."  
**Fix**: Set `KMP_DUPLICATE_LIB_OK=TRUE` in the environment before `import torch`. In PyInstaller builds this must go in `start.py` before any import, or in a runtime hook.

### PyInstaller missing modules

**Symptom**: `ModuleNotFoundError` at runtime that doesn't appear in dev environment.  
**Fix**: Add `collect_all('package_name')` to the spec. If only a submodule is missing, add to `hiddenimports`. Run with `console=True` to see the error clearly.

### Cellpose GPU / CUDA

**Symptom**: Segmentation runs but is very slow.  
**Check**: `torch.cuda.is_available()` — if `False`, torch or CUDA driver mismatch. Verify `cu124` wheels match the installed CUDA runtime.

**Symptom**: Segmentation crashes on GPU.  
**Fix**: Cellpose falls back to CPU automatically, but if it crashes entirely, check VRAM (models need ~4 GB). Reduce batch size in `segment()`.

### Slider debounce

The slider calls `plot_global_area_analysis()` which reads and re-plots from `ALL_para_combine.csv`. On large datasets this can take ~1–2 s. The 200 ms debounce prevents excessive calls but will still fire once the user stops. Increase the debounce timer if lag is noticeable.

### Qt image display performance

Images are loaded from disk and scaled on every `show_image()` call. The `_image_cache` dict stores `(original_rgb, overlay_rgb)` tuples per file path. If memory is a concern with many large images, implement an LRU eviction policy on the cache.

### Adding a new analysis metric

1. Add calculation to `ki67dtc/cell_anal.py` → `param_anal()`
2. Add column to the CSV output in `merged_excel()` if it's a combined nucleus+cytoplasm metric
3. Update any plot functions in `cell_anal_plot.py` if the new metric needs visualization
4. No UI changes needed unless a new display widget is required

### Changing the UI

1. Open `CellimageSegmentation_v6.py`-equivalent `.ui` file in Qt Designer
2. Make changes, save
3. Regenerate Python: `pyuic5 -x CellimageSegmentation_v6.ui -o CellimageSegmentation_v6.py`
4. Update widget references in `controller.py` if widget names changed

### Upgrading Cellpose model

Replace files in `model/` — the filenames are passed as arguments in `main_test2.py` (variables `cyto_model`, `nuc_model`). No code changes needed if the Cellpose API is unchanged.

---

---

## Changelog

### 2026-05-14 — UI Polish Pass

**`CellimageSegmentation_v6.py`** — Chinese → English translations:

| Widget / element | Before | After |
|-----------------|--------|-------|
| Comment above pixel scale group | `# 影像物理尺寸轉換倍率 輸入區（新增）` | `# Pixel scale input area` |
| `pixelScaleGroupBox` title | `"影像物理尺寸轉換倍率 (µm/pixel)"` | `"Pixel Scale (µm/pixel)"` |
| `widthScaleAppliedLabel` default text | `"套用：—"` | `"Applied: —"` |
| `heightScaleAppliedLabel` default text | `"套用：—"` | `"Applied: —"` |

**`controller.py`** — QSS changes applied to all three themes (`_QSS_DARK`, `_QSS_CLEAN`, `_QSS_STEEL`):

| Selector | Property | Before | After | Reason |
|----------|----------|--------|-------|--------|
| `QLabel#ImageFileName` | `font-size` | `10pt` | `14pt` | Label was too small relative to the image panel |
| `QLabel#ImageFileName` | `color` (Dark) | `#5A6878` | `#8A9AB0` | Brightened — muted color too dim at larger size |
| `QLabel#ImageFileName` | `color` (Clean) | `#8898B0` | `#5A6A88` | Darkened — light bg needs more contrast at larger size |
| `QLabel#ImageFileName` | `color` (Steel) | `#6B7A99` | `#8A9ABB` | Brightened — same reasoning as Dark |
| `QPushButton#PreviousButton, QPushButton#NextButton` | `min-width` | *(absent)* | `88px` | "Previous" and "Next" rendered at different widths; root cause: `QSizePolicy.Minimum` inside `QSplitter` has no natural size floor — `min-width` in QSS is the correct fix without touching Python layout code |

**Font licensing audit** (informational, no code change):  
Segoe UI, Segoe UI Black, Times New Roman, Consolas are all Microsoft proprietary fonts. Safe for commercial Windows desktop distribution — users license them via the OS; no font files are redistributed in the installer.

---

*Last updated: 2026-05-14*
