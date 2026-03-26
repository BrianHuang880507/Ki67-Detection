# Cytoplasm Intensity Quantification From Contours (PyImageJ)

This repository now includes a contour-driven quantification script:

- `analyze_cytoplasm_from_contours.py`

The script **does not perform segmentation**. It uses your existing contour coordinates to build ROI masks and quantify fluorescence intensity in:

- nucleus
- whole cytoplasm (required)
- optional whole-cell (when available)

## What the script does

Pipeline (ImageJ-compatible + mask-based measurement):

1. Read image
2. Select signal channel
3. Convert to 16-bit (ImageJ macro)
4. Subtract background using ImageJ `Subtract Background...`
5. Read contour coordinates (CSV/JSON/TXT)
6. Group contours by `image_name` and `cell_id`
7. Build masks directly from polygons
8. Build nucleus ROI
9. Build whole-cell ROI or cytoplasm ROI depending on mode
10. If needed, compute `cytoplasm = cell - nucleus`
11. Estimate background from pixels outside all ROI masks
12. Measure area / mean / integrated density
13. Export ImageJ-style Results rows (one row per ROI)
14. Save QC overlays

## Supported contour modes

### Mode A: nucleus contour + whole-cell contour

- `--contour-mode nucleus_cell`
- Cytoplasm is computed as `whole_cell - nucleus`

### Mode B: nucleus contour + cytoplasm contour

- `--contour-mode nucleus_cytoplasm`
- If cytoplasm contour is true cytoplasm ROI: `--cytoplasm-contour-interpretation direct`
- If contour is actually outer cell boundary: `--cytoplasm-contour-interpretation outer_cell`

## Installation

Recommended for PyImageJ stability:

```bash
mamba create -n ki67 python=3.10 pyimagej openjdk=11 pip -c conda-forge -y
mamba activate ki67
pip install -r requirements.txt

# Windows only: persist OpenMP workaround for this environment (one-time setup)
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE
mamba deactivate
mamba activate ki67
```

Optional: set Fiji path if you want to use local Fiji.app:

```bash
export FIJI_APP_PATH="/path/to/Fiji.app"
```

## Input contour schema

The script expects these logical fields:

- `image_name`
- `cell_id`
- `contour_type`
- polygon coordinates

You can remap your schema using arguments such as:

- `--image-name-column`
- `--cell-id-column`
- `--contour-type-column`
- `--coords-column`
- `--x-column --y-column` (for long CSV format)
- `--json-records-key`

### TXT support (main pipeline outlines)

Supported directly:

- one merged file: `*_merged_cp_outlines.txt`
- or a folder containing many merged files

Merged txt format is interpreted as pairs:

- line 1: nucleus
- line 2: second contour (`cytoplasm` by default, configurable via `--txt-second-contour-type`)
- `-1,-1` is treated as missing contour

### CSV support

Two styles are supported:

1. One row = one polygon (e.g., `polygon` column as JSON list or `x,y;x,y;...`)
2. Long format (one row = one point) using `--x-column --y-column` and optional `--point-order-column`

### JSON support

- Top-level list of contour records, or
- Top-level dict with list key (set via `--json-records-key`)

## Usage examples

### Quick mode (main-pipeline style, only dataset folder)

```bash
python analyze_cytoplasm_from_contours.py my_dataset
```

Quick mode defaults:

- auto-pick image folder under `data/input/my_dataset` (priority: `PC -> DF -> KI67 -> IDO -> LT -> DAPI`)
- contours from `data/output/outline/my_dataset`
- output layout as `data/output/results/my_dataset/cell_measurements.csv`
- no QC overlay images (unless you add `--save-qc-overlays`)

### Example 1: nucleus + whole-cell contours

```bash
python analyze_cytoplasm_from_contours.py \
  --input-dir ./data/images \
  --contours-file ./data/contours.csv \
  --output-dir ./output/contour_analysis \
  --contour-format csv \
  --contour-mode nucleus_cell \
  --signal-channel 1 \
  --rolling-ball-radius 50 \
  --save-qc-overlays \
  --save-summary-per-image
```

### Example 1b: use main pipeline output layout

```bash
python analyze_cytoplasm_from_contours.py \
  --input-dir ./data/input/my_dataset/DF \
  --contours-file ./data/input/my_dataset/contours.csv \
  --use-main-output-layout
```

This writes into:

- `data/output/results/my_dataset/cell_measurements.csv`
- `data/output/results/my_dataset/image_summary.csv` (if enabled)
- `data/output/qc_overlays/my_dataset/*.png` (if enabled)

### Example 1c: use existing merged txt outlines from main pipeline

```bash
python analyze_cytoplasm_from_contours.py \
  --input-dir ./data/input/my_dataset/DF \
  --contours-file ./data/output/outline/my_dataset \
  --contour-format txt \
  --txt-glob "*_merged_cp_outlines.txt" \
  --txt-second-contour-type cell \
  --contour-mode nucleus_cell \
  --use-main-output-layout
```

### Example 2: nucleus + cytoplasm contours (direct cytoplasm)

```bash
python analyze_cytoplasm_from_contours.py \
  --input-dir ./data/images \
  --contours-file ./data/contours.json \
  --output-dir ./output/contour_analysis \
  --contour-format json \
  --contour-mode nucleus_cytoplasm \
  --cytoplasm-contour-interpretation direct \
  --signal-channel 0
```

### Example 3: nucleus + cytoplasm field that is actually outer cell boundary

```bash
python analyze_cytoplasm_from_contours.py \
  --input-dir ./data/images \
  --contours-file ./data/contours.csv \
  --output-dir ./output/contour_analysis \
  --contour-mode nucleus_cytoplasm \
  --cytoplasm-contour-interpretation outer_cell
```

## Output files

In `--output-dir` (or in `data/output/...` when `--use-main-output-layout` is enabled):

- `cell_measurements.csv` (one row per cell, aligned with main pipeline targets except Ki67)
- `image_summary.csv` (optional, if `--save-summary-per-image`)
- `qc_overlays/*.png` (optional, if `--save-qc-overlays`)

## Required CSV output columns

The output keeps main-pipeline measurement targets (without `ki67_positive`) and appends ImageJ-style intensity fields at the end:

- `Cell_ID`
- `Area_nuc`, `Perimeter_nuc`, `Convex Perimeter_nuc`, `Circular Diameter_nuc`, `Feret Length_nuc`, `Feret Width_nuc`, `Aspect Ratio_nuc`, `Roundness_nuc`, `Circularity_nuc`, `Sphericity_nuc`, `Roughness_nuc`
- `Area_cyto`, `Perimeter_cyto`, `Convex Perimeter_cyto`, `Circular Diameter_cyto`, `Feret Length_cyto`, `Feret Width_cyto`, `Aspect Ratio_cyto`, `Roundness_cyto`, `Circularity_cyto`, `Sphericity_cyto`, `Roughness_cyto`
- `Karyoplasmic Ratio_cyto`
- `IntDen` (background-corrected: `ID - Area * mean_background`)
- `RawIntDen` (raw integrated density)

## QC behavior

The script flags malformed or suspicious inputs, including:

- missing contour pairs
- zero-area polygons
- invalid polygons
- contour points outside image bounds
- nucleus not inside cell (when whole-cell mode is used)
- border-touching ROIs (if border check is enabled)

Use:

- `--exclude-flagged` to drop flagged rows from final CSV
- `--flag-border-touching` to only flag border-touching ROIs
- `--exclude-border-touching` to apply border-touch checks and exclude them

## Notes

- No threshold-based segmentation is used.
- No ring/concentric/perinuclear/peripheral analysis is performed.
- Cytoplasm quantification is whole-region only.
