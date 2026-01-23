#!/usr/bin/env bash
set -euo pipefail

# Config
# Resolve python if not provided
if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif [ -x "/mnt/d/Anaconda3/envs/cellpose/python.exe" ]; then
    PYTHON_BIN="/mnt/d/Anaconda3/envs/cellpose/python.exe"
  elif [ -x "/mnt/d/anaconda3/envs/cellpose/python.exe" ]; then
    PYTHON_BIN="/mnt/d/anaconda3/envs/cellpose/python.exe"
  elif [ -x "D:/Anaconda3/envs/cellpose/python.exe" ]; then
    PYTHON_BIN="D:/Anaconda3/envs/cellpose/python.exe"
  elif [ -x "D:/anaconda3/envs/cellpose/python.exe" ]; then
    PYTHON_BIN="D:/anaconda3/envs/cellpose/python.exe"
  else
    echo "python not found; set PYTHON_BIN to your interpreter path." >&2
    exit 1
  fi
fi
NEG_THRESHOLD="${NEG_THRESHOLD:-0.15}"
PRECISION_TARGET="${PRECISION_TARGET:-0.9}"
PRED_DIR="${PRED_DIR:-predictions}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

shopt -s nullglob
files=(${PRED_DIR}/predictions_xgb_tab_*.csv)
if [ ${#files[@]} -eq 0 ]; then
  echo "No prediction files found under ${PRED_DIR}"
  exit 0
fi

for f in "${files[@]}"; do
  out="${f%.csv}_post.csv"
  "$PYTHON_BIN" postprocess_predictions.py "$f" \
    --output "$out" \
    --neg-threshold "$NEG_THRESHOLD" \
    --precision-target "$PRECISION_TARGET"
  echo "done: $out"
done
