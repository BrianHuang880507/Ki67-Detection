# Ki67 Cell Image Analysis Pipeline

This project runs the full cell image analysis pipeline, including segmentation.

The main entry point is `main.py`. For batch processing, use
`scripts/run_all_data_input.py`, which runs `main.py` once for each first-level
dataset folder under `data/input`.

## Pipeline

`main.py` runs these steps:

1. Segmentation for cytoplasm and nucleus.
2. Convert Cellpose segmentation masks (`*.npy`) to outline text files.
3. Merge nucleus and cytoplasm outlines.
4. Measure geometry, fluorescence intensity, and optional Ki67 positivity.

## Input Layout

Expected dataset layout:

```text
data/
  input/
    example_dataset/
      PC/
      DAPI/
      IDO/
      KI67/
  output/
```

Folders:

- `PC/`: phase contrast images. Cytoplasm segmentation always uses this folder.
- `DAPI/`: optional nucleus images. Used when `--nuc_source dapi`.
- `IDO/`, `DF/`, `LT/`: optional fluorescence channels.
- `KI67/`: optional Ki67 images.

## Single Dataset

Run one dataset:

```bash
python main.py --data_folder example_dataset --nuc_source dapi --fluor_analy --ki67
```

You can also pass an absolute or relative dataset path:

```bash
python main.py --data_folder data/input/example_dataset
```

## Batch Processing

Run every dataset folder under `data/input`:

```bash
python scripts/run_all_data_input.py --fluor_analy --ki67
```

Run only selected datasets:

```bash
python scripts/run_all_data_input.py --only dataset_a dataset_b --fluor_analy --ki67
```

Preview commands without executing:

```bash
python scripts/run_all_data_input.py --dry-run --fluor_analy --ki67
```

## Common Arguments

- `--data_folder`: dataset folder name or path. Required for `main.py`.
- `--nuc_source`: nucleus segmentation source, either `dapi` or `pc`.
- `--fluor_analy`: enable fluorescence analysis.
- `--ki67`: enable Ki67 positivity analysis.
- `--ki67_backend`: Ki67 binarization backend, either `pyimagej` or `opencv`.
- `--clean_temp`: remove temporary intermediate files.

## Output Layout

Pipeline outputs are written under `data/output`:

```text
data/output/
  segment/
  outline/
  results/
```

Typical outputs:

- `data/output/segment/<dataset>/`: Cellpose segmentation masks.
- `data/output/outline/<dataset>/`: cytoplasm, nucleus, and merged outline files.
- `data/output/results/<dataset>/`: measurement CSV files and final analysis results.
