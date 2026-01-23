#!/usr/bin/env bash
set -euo pipefail

# Config
# Pick python: respect $PYTHON if set; otherwise try system python, else fallback to cellpose env.
if [ -z "${PYTHON:-}" ]; then
    if command -v python >/dev/null 2>&1; then
        PYTHON="$(command -v python)"
    elif [ -x "/usr/bin/python" ]; then
        PYTHON="/usr/bin/python"
    elif [ -x "/mnt/d/Anaconda3/envs/cellpose/python.exe" ]; then
        PYTHON="/mnt/d/Anaconda3/envs/cellpose/python.exe"
    elif [ -x "/mnt/d/anaconda3/envs/cellpose/python.exe" ]; then
        PYTHON="/mnt/d/anaconda3/envs/cellpose/python.exe"
    elif [ -x "D:/Anaconda3/envs/cellpose/python.exe" ]; then
        PYTHON="D:/Anaconda3/envs/cellpose/python.exe"
    elif [ -x "D:/anaconda3/envs/cellpose/python.exe" ]; then
        PYTHON="D:/anaconda3/envs/cellpose/python.exe"
    else
        echo "python not found; set PYTHON to your interpreter path." >&2
        exit 1
    fi
fi
ROOT_DIR_UNIX="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$PYTHON" == *.exe ]]; then
    if [[ "$PYTHON" == /mnt/* ]]; then
        if command -v cygpath >/dev/null 2>&1; then
            ROOT_DIR_FOR_PY="$(cygpath -w "$ROOT_DIR_UNIX")"
        else
            ROOT_DIR_FOR_PY="D:/Project/ki67dtc"
        fi
    else
        ROOT_DIR_FOR_PY="D:/Project/ki67dtc"
    fi
else
    ROOT_DIR_FOR_PY="$ROOT_DIR_UNIX"
fi
PREDICT_PY="$ROOT_DIR_FOR_PY/predict.py"
WORKDIR="$ROOT_DIR_UNIX"
MODEL_DIR="outputs_models/20251031-190834"
MODEL_KEY="xgb_tab"
IMAGE_ROOT="data/output/cyto_crops"
CHANNEL="cyto"
BACKBONE=""  # Leave empty to use manifest; set to resnet18/resnet50 to override
BATCH_SIZE=64
THRESHOLD=0.5
OUT_DIR="predictions"
mkdir -p "$OUT_DIR"

# List cleaned CSV paths to run
CSV_LIST=(
    "data/output/results/0819/0819_cleaned.csv"
    "data/output/results/2025-06-19-B4-P6-P10-P14-Ki67-P6-1/2025-06-19-B4-P6-P10-P14-Ki67-P6-1_cleaned.csv"
    "data/output/results/2025-06-19-B4-P6-P10-P14-Ki67-P6-2/2025-06-19-B4-P6-P10-P14-Ki67-P6-2_cleaned.csv"
    "data/output/results/2025-06-19-B4-P6-P10-P14-Ki67-P10-1/2025-06-19-B4-P6-P10-P14-Ki67-P10-1_cleaned.csv"
    "data/output/results/2025-06-19-B4-P6-P10-P14-Ki67-P10-2/2025-06-19-B4-P6-P10-P14-Ki67-P10-2_cleaned.csv"
    "data/output/results/2025-06-19-B4-P6-P10-P14-Ki67-P14-1/2025-06-19-B4-P6-P10-P14-Ki67-P14-1_cleaned.csv"
    "data/output/results/2025-06-19-B4-P6-P10-P14-Ki67-P14-2/2025-06-19-B4-P6-P10-P14-Ki67-P14-2_cleaned.csv"
    "data/output/results/2025-07-10-B8-P6-P10-P14-Ki67-lot-2-P6/2025-07-10-B8-P6-P10-P14-Ki67-lot-2-P6_cleaned.csv"
    "data/output/results/2025-07-10-B8-P6-P10-P14-Ki67-lot-2-P10/2025-07-10-B8-P6-P10-P14-Ki67-lot-2-P10_cleaned.csv"
    "data/output/results/2025-07-10-B8-P6-P10-P14-Ki67-lot-2-P14/2025-07-10-B8-P6-P10-P14-Ki67-lot-2-P14_cleaned.csv"
    "data/output/results/P7-P10-P10/P7-P10-P10_cleaned.csv"
    "data/output/results/P11-P13-P13/P11-P13-P13_cleaned.csv"
    "data/output/results/P7-P10-P7/P7-P10-P7_cleaned.csv"
    "data/output/train/P7-P10-P8/P7-P10-P8_cleaned.csv"
    "data/output/train/P7-P10-P9/P7-P10-P9_cleaned.csv"
    "data/output/train/P11-P13-P11/P11-P13-P11_cleaned.csv"
    "data/output/train/P11-P13-P12/P11-P13-P12_cleaned.csv"
)

pushd "$WORKDIR" >/dev/null

for CSV in "${CSV_LIST[@]}"; do
    TAG="$(basename "${CSV%.csv}")"
    OUT_PATH="${OUT_DIR}/predictions_${MODEL_KEY}_${TAG}_$(date +%Y%m%d-%H%M%S).csv"

    "$PYTHON" "$PREDICT_PY" \
        --model-dir "$MODEL_DIR" \
        --model-key "$MODEL_KEY" \
        --csv "$CSV" \
        --image-root "$IMAGE_ROOT" \
        --channel "$CHANNEL" \
        --batch-size "$BATCH_SIZE" \
        --threshold "$THRESHOLD" \
        ${BACKBONE:+--backbone "$BACKBONE"} \
        --output "$OUT_PATH"

    echo "done: $OUT_PATH"
done

popd >/dev/null
