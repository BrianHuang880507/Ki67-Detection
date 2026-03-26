# METHOD.md

## Project Goal

Build a PyImageJ-based fluorescence quantification pipeline that uses **pre-existing contour coordinates** for nuclei and cell/cytoplasm regions, rather than performing image segmentation.

The main objective is to quantify fluorescence intensity over the **entire cytoplasm** of each cell.

This workflow does **not** use:

- automatic segmentation
- ring-based analysis
- perinuclear or pericellular regional binning

Instead, it uses user-provided contour coordinates to construct ROIs directly.

---

## Analysis Principle

The analysis is ROI-based, not segmentation-based.

Available inputs:

- microscopy image
- nucleus contour coordinates
- cytoplasm contour coordinates and/or whole-cell contour coordinates

The script should convert these contours into masks and quantify fluorescence features directly from the masks.

---

## ROI Definitions

Two possible contour-input modes must be supported:

### Mode 1: nucleus contour + whole-cell contour

In this mode:

- whole_cell_mask = whole-cell contour
- nucleus_mask = nucleus contour
- cytoplasm_mask = whole_cell_mask - nucleus_mask

### Mode 2: nucleus contour + cytoplasm contour

If the provided cytoplasm contour already represents the cytoplasm-only region:

- cytoplasm_mask = provided cytoplasm contour directly
- nucleus_mask = nucleus contour

If the provided “cytoplasm contour” is actually the outer cell boundary:

- treat it as whole-cell contour
- compute cytoplasm_mask = whole_cell_mask - nucleus_mask

The implementation should allow this behavior to be controlled by a parameter.

---

## Input Assumptions

### Image input

Supported image formats:

- `.tif`
- `.tiff`
- `.png`

### Contour input

Contour coordinates are already available for each cell.

Supported contour storage should be easy to adapt, such as:

- CSV
- JSON

Each contour should contain:

- image_name
- cell_id
- contour_type (`nucleus`, `cell`, or `cytoplasm`)
- ordered polygon coordinates

Example logical structure:

- one nucleus polygon per cell
- one cell or cytoplasm polygon per cell

---

## Processing Steps

### 1. Load image

- Open the microscopy image with PyImageJ or Python-compatible I/O
- Select the target signal channel
- Convert to 16-bit if needed

### 2. Preprocess signal image

- Apply ImageJ-style background subtraction
- Keep this configurable
- Do not perform segmentation

### 3. Load contour coordinates

- Read contour data from CSV or JSON
- Group contours by image_name and cell_id
- Validate that required contour types exist

### 4. Build ROI masks from contours

For each cell:

- convert polygon coordinates into binary masks
- create nucleus mask
- create whole-cell or cytoplasm mask depending on input mode

If needed:

- clip masks to image boundaries
- fill polygon interiors

### 5. Build cytoplasm ROI

If whole-cell contour is provided:

- cytoplasm_mask = whole_cell_mask AND NOT nucleus_mask

If cytoplasm contour is already cytoplasm-only:

- cytoplasm_mask = provided cytoplasm mask directly

### 6. Measure background

Estimate background from pixels outside all cell-related masks in the image.

Calculate:

- mean_background_intensity

### 7. Measure intensity features

For each cell, compute:

#### Nucleus

- nucleus_area_px
- nucleus_mean
- nucleus_integrated_density

#### Cytoplasm

- cytoplasm_area_px
- cytoplasm_mean
- cytoplasm_integrated_density
- cytoplasm_stddev
- cytoplasm_min
- cytoplasm_max

#### Whole cell (if available or reconstructable)

- whole_cell_area_px
- whole_cell_mean
- whole_cell_integrated_density

### 8. Background correction

For each ROI:

- corrected_integrated_density = integrated_density - area_px \* mean_background

At minimum provide:

- nucleus_corrected_intden
- cytoplasm_corrected_intden

If whole-cell ROI exists:

- whole_cell_corrected_intden

### 9. Output

Export one row per cell to CSV.

Required columns:

- image_name
- cell_id
- mean_background
- nucleus_area_px
- nucleus_mean
- nucleus_integrated_density
- nucleus_corrected_intden
- cytoplasm_area_px
- cytoplasm_mean
- cytoplasm_integrated_density
- cytoplasm_corrected_intden

Optional columns:

- whole_cell_area_px
- whole_cell_mean
- whole_cell_integrated_density
- whole_cell_corrected_intden
- cytoplasm_stddev
- cytoplasm_min
- cytoplasm_max
- cytoplasm_to_nucleus_mean_ratio
- cytoplasm_to_nucleus_corrected_intden_ratio

---

## Quality Control

The script should flag or exclude:

- missing contour pairs
- self-intersecting polygons if invalid
- zero-area masks
- nucleus mask outside cell mask when whole-cell mode is used
- contours that fall outside image boundaries
- overlapping or duplicated cell IDs

The script should optionally save:

- overlay QC images
- mask preview images
- rejected-cell log

---

## Implementation Requirements

### Language / stack

- Python
- PyImageJ
- numpy
- pandas

Additional polygon/mask helpers are acceptable if needed:

- scikit-image
- shapely
- opencv-python

### Preferred implementation style

- use provided contours directly
- do not implement threshold-based segmentation unless explicitly requested later
- use PyImageJ mainly for image loading / ImageJ-compatible preprocessing
- use NumPy-based mask measurements for robustness

---

## Parameters to Expose

At minimum expose:

- input_image_dir
- contour_file
- output_dir
- signal_channel_index
- rolling_ball_radius
- contour_mode (`cell_and_nucleus` or `cytoplasm_and_nucleus`)
- cytoplasm_contour_is_direct_roi (true/false)
- exclude_border_objects
- save_qc_overlay
- file_extensions

---

## Important Constraints

- Do NOT do segmentation from image intensity
- Do NOT implement concentric rings
- Do NOT calculate perinuclear or peripheral shells
- Quantify the **entire cytoplasm**
- Use the provided contour coordinates as the primary source of ROI definition

---

## Deliverables Expected From Codex

Codex should generate:

1. `analyze_cytoplasm_from_contours.py`
2. `requirements.txt`
3. `README.md`

The script should batch-process images and export per-cell CSV results plus QC overlays.

## Workflow Preservation Requirement

This project must **preserve the current main analysis workflow** as much as possible.

The intention is **not** to redesign or replace the existing pipeline architecture.
Instead, only the ROI-definition step should be adapted so that existing contour coordinates are used directly.

Therefore:

- keep the current main processing order unchanged whenever possible
- keep the existing image loading, preprocessing, measurement, and export logic unchanged whenever possible
- do not refactor the whole pipeline unless required for compatibility
- replace only the ROI acquisition method:
    - from image-based segmentation
    - to contour-based ROI construction

In other words, this is a **minimal-intrusion modification** to the current workflow, not a new pipeline design.
