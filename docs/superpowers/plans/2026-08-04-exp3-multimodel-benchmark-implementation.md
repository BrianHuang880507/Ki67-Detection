# Exp3 Phase-only Morphology 多模型 Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立與既有 main/core 完全隔離的 Exp3 pipeline，從九個 B-ID × Passage 資料夾的 PC／IDO 影像建立 phase-only morphology predictors，執行多模型 grouped nested validation，並產生可稽核的實驗記錄。

**Architecture:** 新功能只放在 `immunity/exp3/`，直接讀取 raw PC／IDO paths，將 segmentation masks、cell-level features 與 secondary features 寫入 `immunity/outputs/exp3/feature_cache/`。Primary benchmark 只啟用 33 個 basic median predictors；66／93-feature sets 採明確設定與 lazy computation，模型、split、ranking、reporting 都由 Exp3 專用 CLI 串接，不改動 `main.py`、GUI、cleaned CSV／XLSX 或既有 predictor schema。

**Tech Stack:** Python 3.10、NumPy、pandas、SciPy、scikit-learn、Cellpose、OpenCV、scikit-image、mahotas、matplotlib、PyYAML、pytest

## Global Constraints

- 正式入口固定為 `python -m immunity.exp3.run_benchmark --config immunity/configs/exp3.yaml`。
- Exp3 code、cache 與結果只能位於 `immunity/exp3/`、`immunity/configs/exp3.yaml`、`immunity/outputs/exp3/` 及 Exp3 專屬 tests。
- 不修改 `main.py` CLI、GUI、main cleaned CSV／XLSX、`immunity/run_experiment.py`、`immunity/build_dataset.py` 或 `ki67dtc` predictor schema。
- Primary run 預設只計算 33 個 basic medians；`basic_median_iqr` 與 `paper_style_median` 必須在 Exp3 config 明確啟用。
- Phase-only predictors 不得包含 IDO intensity、IFN/TNF dose、condition、檔名、路徑或任何 test-derived value。
- IDO 只作為 image-level background-corrected target，報告名稱固定為 `IDO proxy`，不得稱為整體免疫抑制能力。
- B4、B7、B8 只標記為 `B-ID`；沒有 donor／lot metadata 前不得改稱 donor 或 lot。
- 正式 condition mapping 未經實驗人員確認前，`immunity/configs/exp3.yaml` 保持空 mapping，CLI 必須在寫出 pairing QC 後停止模型分析。
- 所有 preprocessing、target scaling、hyperparameter tuning 與 supervised selection 必須只 fit outer-training data。
- 舊 DAPI 只能作為 PC nucleus segmentation 的 development-only reference；不得進入 predictors、正式推論或未來資料需求。
- 執行預估：unit tests 2–5 分鐘；72-image smoke segmentation 10–25 分鐘；719-image segmentation 約 1–3 小時；完整 nested benchmark 約 4–12 小時。

---

## File Structure

| Path | Responsibility |
|---|---|
| `immunity/exp3/__init__.py` | Exp3 package version 與公開常數；不得觸發計算。 |
| `immunity/exp3/manifest.py` | 九資料夾檔名解析、PC／IDO pairing、condition mapping 與 QC。 |
| `immunity/exp3/phase_features.py` | PC-only cell/nucleus segmentation、mask cache、33 basic features、IDO target。 |
| `immunity/exp3/feature_sets.py` | Primary／secondary registry、lazy extraction、FOV aggregation、ΔMorphology。 |
| `immunity/exp3/benchmark.py` | grouped splits、model registry、nested tuning、metrics、ranking 與 sensitivity analysis。 |
| `immunity/exp3/reporting.py` | 固定輸出 schema、圖表與 `EXPERIMENT_RECORD.md`。 |
| `immunity/exp3/run_benchmark.py` | Exp3 config、output boundary、pipeline orchestration 與 CLI。 |
| `immunity/configs/exp3.yaml` | 九組輸入資料、空白但強制的 condition mapping、模型與輸出設定。 |
| `immunity/exp3/README.md` | 執行順序、mapping blocker、輸出與證據限制。 |
| `tests/test_immunity_exp3_manifest.py` | parser、pairing、missing/duplicate 與 mapping tests。 |
| `tests/test_immunity_exp3_phase_features.py` | PC-only segmentation cache、mask pairing、basic feature 與 target tests。 |
| `tests/test_immunity_exp3_feature_sets.py` | 33／66／93 registry、lazy computation、whitelist 與 ΔMorphology tests。 |
| `tests/test_immunity_exp3_benchmark.py` | split leakage、nested models、metrics、ranking、eligibility 與 sensitivity tests。 |
| `tests/test_immunity_exp3_reporting.py` | output schema、no-winner report 與 figures smoke tests。 |
| `tests/test_immunity_exp3_cli.py` | output isolation、empty mapping blocker、synthetic end-to-end 與 main contract tests。 |

---

### Task 1: 建立 Exp3 package、設定檔與硬隔離邊界

**Files:**
- Create: `immunity/exp3/__init__.py`
- Create: `immunity/exp3/run_benchmark.py`
- Create: `immunity/configs/exp3.yaml`
- Test: `tests/test_immunity_exp3_cli.py`

**Interfaces:**
- Produces: `load_config(path: str | Path) -> dict[str, Any]`
- Produces: `resolve_exp3_output_dir(path: str | Path) -> Path`
- Produces: `build_parser() -> argparse.ArgumentParser`
- Boundary: resolved output must equal or be below `<project>/immunity/outputs/exp3`; all other targets raise `ValueError`.

- [ ] **Step 1: Write failing isolation and config tests**

```python
from pathlib import Path

import pytest

import main
from immunity.build_dataset import morphology_feature_columns
from immunity.exp3.run_benchmark import load_config, resolve_exp3_output_dir


def test_exp3_output_must_stay_under_dedicated_root(tmp_path: Path) -> None:
    allowed = resolve_exp3_output_dir("immunity/outputs/exp3/smoke")
    assert allowed.as_posix().endswith("immunity/outputs/exp3/smoke")
    with pytest.raises(ValueError, match="immunity/outputs/exp3"):
        resolve_exp3_output_dir("immunity/outputs/b4_p6")
    with pytest.raises(ValueError, match="immunity/outputs/exp3"):
        resolve_exp3_output_dir(tmp_path)


def test_importing_exp3_does_not_expand_main_contract() -> None:
    destinations = {action.dest for action in main.build_parser()._actions}
    assert destinations == {
        "help", "data_folder", "device", "nuc_source", "fluor_analy",
        "ki67", "ki67_backend", "feature_backend", "clean_temp", "xlsx_version",
    }
    predictors = morphology_feature_columns()
    assert len(predictors) == 33
    assert not any("Zernike" in name or "__median" in name for name in predictors)


def test_exp3_config_loads_without_running_pipeline() -> None:
    config = load_config("immunity/configs/exp3.yaml")
    assert len(config["datasets"]) == 9
    assert config["expected_totals"] == {"pc": 720, "ido": 719, "paired": 719}
    assert config["condition_mapping"] == {}
    assert config["feature_sets"]["enabled"] == ["basic_median"]
```

- [ ] **Step 2: Run tests to verify missing package failure**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_cli.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'immunity.exp3'`.

- [ ] **Step 3: Add side-effect-free package and output guard**

```python
# immunity/exp3/__init__.py
"""Exp3 phase-only morphology 多模型實驗套件。"""

EXP3_SCHEMA_VERSION = "1.0"
PRIMARY_FEATURE_SET = "basic_median"

__all__ = ["EXP3_SCHEMA_VERSION", "PRIMARY_FEATURE_SET"]
```

```python
# immunity/exp3/run_benchmark.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP3_OUTPUT_ROOT = (PROJECT_ROOT / "immunity" / "outputs" / "exp3").resolve()


def resolve_exp3_output_dir(path: str | Path) -> Path:
    """解析並限制 Exp3 輸出位置。

    Args:
        path: 設定檔指定的輸出路徑。

    Returns:
        位於 Exp3 專屬根目錄內的絕對路徑。

    Raises:
        ValueError: 路徑不在 `immunity/outputs/exp3` 之下時拋出。
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve(strict=False)
    if resolved != EXP3_OUTPUT_ROOT and EXP3_OUTPUT_ROOT not in resolved.parents:
        raise ValueError("Exp3 output 必須位於 immunity/outputs/exp3。")
    return resolved


def load_config(path: str | Path) -> dict[str, Any]:
    """讀取 Exp3 YAML，但不執行影像或模型分析。"""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Exp3 config 最外層必須是 mapping。")
    required = {
        "datasets", "expected_totals", "condition_mapping", "segmentation",
        "development_validation", "feature_sets", "benchmark", "output",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Exp3 config 缺少區塊：{missing}")
    config["_config_path"] = str(config_path.resolve())
    config["_output_dir"] = str(resolve_exp3_output_dir(config["output"]["dir"]))
    return config


def build_parser() -> argparse.ArgumentParser:
    """建立 Exp3 專用命令列介面。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-fovs-per-condition", type=int, default=None)
    return parser
```

- [ ] **Step 4: Add the real nine-folder config with an intentional mapping blocker**

```yaml
experiment_name: exp3_phase_only_multimodel_benchmark

datasets:
  - {input_dir: "data/input/20260706_B4-P5", b_id: B4, passage: 5}
  - {input_dir: "data/input/20260706 _B7-P5", b_id: B7, passage: 5}
  - {input_dir: "data/input/20260706 _B8-P5", b_id: B8, passage: 5}
  - {input_dir: "data/input/20260715_B4-P6", b_id: B4, passage: 6}
  - {input_dir: "data/input/20260715_B7-P6", b_id: B7, passage: 6}
  - {input_dir: "data/input/20260715_B8-P6", b_id: B8, passage: 6}
  - {input_dir: "data/input/20260717_B4-P7", b_id: B4, passage: 7}
  - {input_dir: "data/input/20260717_B7-P7", b_id: B7, passage: 7}
  - {input_dir: "data/input/20260717_B8-P7", b_id: B8, passage: 7}

expected_totals: {pc: 720, ido: 719, paired: 719}

# 實驗人員尚未確認編號 1–8 的 IFN/TNF 對應；空 mapping 會在 pairing QC 後停止。
condition_mapping: {}

segmentation:
  device: gpu
  force: false
  cellprob_threshold: 0.0
  min_area_ratio: 0.15
  min_area_floor: 30
  max_nucleus_outside_fraction: 0.05
  min_cells_per_image: 3

# 只用舊 B4 p6 DAPI 評估 PC nucleus masks；正式 Exp3 預設不啟用。
development_validation:
  enabled: false
  input_dir: "data/input/B4  p6"
  sample_size: 10

feature_sets:
  enabled: [basic_median]
  paper_style_max_features: 93

benchmark:
  seed: 20260804
  inner_splits: 5
  max_hyperparameter_candidates: 24
  n_jobs: 1
  permutation_repeats: 20
  simplicity_order:
    [paper_linear_3f, ridge, elasticnet, rbf_svr, hist_gradient_boosting, random_forest, extra_trees]

output:
  dir: immunity/outputs/exp3
```

- [ ] **Step 5: Run the isolation tests**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_cli.py -v`

Expected: 3 passed.

- [ ] **Step 6: Commit the isolated skeleton**

```powershell
git add immunity/exp3/__init__.py immunity/exp3/run_benchmark.py immunity/configs/exp3.yaml tests/test_immunity_exp3_cli.py
git commit -m "feat(immunity): 建立隔離的 Exp3 執行邊界"
```

---

### Task 2: 建立跨 P5–P7 的 manifest parser 與 pairing QC

**Files:**
- Create: `immunity/exp3/manifest.py`
- Test: `tests/test_immunity_exp3_manifest.py`

**Interfaces:**
- Produces: `parse_image_name(name: str, expected_channel: str) -> tuple[int, int]`
- Produces: `scan_datasets(specs: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]`
- Produces: `validate_expected_totals(qc: pd.DataFrame, expected: Mapping[str, int]) -> None`
- Produces: `apply_condition_mapping(raw_manifest: pd.DataFrame, mapping: Mapping[Any, Any]) -> pd.DataFrame`
- Raw manifest contains only complete PC／IDO pairs; pairing QC contains `paired`, `missing_pc`, `missing_ido`, `duplicate_pc`, `duplicate_ido`, and `parse_error` rows.

- [ ] **Step 1: Write parser, missing-pair, extra-dot and mapping tests**

```python
from pathlib import Path

import pandas as pd
import pytest

from immunity.exp3.manifest import (
    apply_condition_mapping,
    parse_image_name,
    scan_datasets,
    validate_expected_totals,
)


def test_parser_accepts_all_three_naming_styles() -> None:
    assert parse_image_name("4-phase-100X-10.jpg", "pc") == (4, 10)
    assert parse_image_name("B4-P6-10x-4-10.jpg", "pc") == (4, 10)
    assert parse_image_name("B8-P7-10X-8.-10.jpg", "pc") == (8, 10)
    assert parse_image_name("B8-P7-10X-8-IDO.-10.jpg", "ido") == (8, 10)


def test_scan_keeps_pair_and_records_missing_ido(tmp_path: Path) -> None:
    root = tmp_path / "B7-P5"
    (root / "PC").mkdir(parents=True)
    (root / "IDO").mkdir()
    (root / "PC" / "1-phase-100X-1.jpg").touch()
    (root / "IDO" / "1-IDO-100X-1.jpg").touch()
    (root / "PC" / "4-phase-100X-10.jpg").touch()
    manifest, qc = scan_datasets([{"input_dir": root, "b_id": "B7", "passage": 5}])
    assert list(manifest[["condition_index", "fov"]].itertuples(index=False, name=None)) == [(1, 1)]
    missing = qc[qc["status"].eq("missing_ido")]
    assert list(missing[["condition_index", "fov"]].itertuples(index=False, name=None)) == [(4, 10)]


def test_condition_mapping_is_mandatory_and_exact() -> None:
    raw = pd.DataFrame([{"b_id": "B4", "passage": 5, "condition_index": 1, "fov": 1,
                         "group_id": "B4_P5", "pc_path": "pc.jpg", "ido_path": "ido.jpg"}])
    with pytest.raises(ValueError, match="condition mapping"):
        apply_condition_mapping(raw, {})
    mapping = {
        index: {"ifn_dose": float(index - 1), "tnf_dose": 0.0,
                "condition": f"condition_{index}"}
        for index in range(1, 9)
    }
    mapped = apply_condition_mapping(raw, mapping)
    assert mapped.loc[0, "condition"] == "condition_1"


def test_expected_totals_are_enforced() -> None:
    qc = pd.DataFrame([
        {"status": "paired", "pc_count": 1, "ido_count": 1},
        {"status": "missing_ido", "pc_count": 1, "ido_count": 0},
    ])
    validate_expected_totals(qc, {"pc": 2, "ido": 1, "paired": 1})
    with pytest.raises(ValueError, match="expected totals"):
        validate_expected_totals(qc, {"pc": 2, "ido": 2, "paired": 2})
```

- [ ] **Step 2: Run tests to verify missing module failure**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_manifest.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'immunity.exp3.manifest'`.

- [ ] **Step 3: Implement explicit filename parsers**

```python
P5_PATTERN = re.compile(
    r"^(?P<condition>[1-8])-(?P<channel>phase|IDO)-100X-(?P<fov>\d{1,2})$",
    re.IGNORECASE,
)
P67_PATTERN = re.compile(
    r"^B\d+-P[67]-10X-(?P<condition>[1-8])\.?-"
    r"(?:(?P<ido>IDO)\.?-)?(?P<fov>\d{1,2})$",
    re.IGNORECASE,
)


def parse_image_name(name: str, expected_channel: str) -> tuple[int, int]:
    """解析 condition index 與 FOV，並驗證所在 channel folder。"""
    stem = Path(name).stem
    match = P5_PATTERN.fullmatch(stem) or P67_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"無法解析 Exp3 影像檔名：{name}")
    parsed_channel = "ido" if match.groupdict().get("ido") or "-IDO-" in stem.upper() else "pc"
    if "phase" in stem.lower():
        parsed_channel = "pc"
    if parsed_channel != expected_channel.lower():
        raise ValueError(f"資料夾與檔名 channel 不一致：{name}")
    condition = int(match.group("condition"))
    fov = int(match.group("fov"))
    if not 1 <= fov <= 10:
        raise ValueError(f"FOV 必須介於 1–10：{name}")
    return condition, fov
```

- [ ] **Step 4: Implement deterministic pairing and QC rows**

`scan_datasets()` must resolve every input path, scan only supported image suffixes, group files by `(b_id, passage, condition_index, fov)`, and use the following exact output columns:

```python
MANIFEST_COLUMNS = [
    "b_id", "passage", "condition_index", "fov", "group_id", "pc_path", "ido_path",
]
PAIRING_QC_COLUMNS = [
    "b_id", "passage", "condition_index", "fov", "status",
    "pc_count", "ido_count", "pc_path", "ido_path", "detail",
]
```

For a key, emit `paired` only when `pc_count == ido_count == 1`; do not put incomplete or duplicate keys in the raw manifest. Add parse failures as `parse_error` with the original file path in `detail`. Sort both frames by `b_id`, `passage`, `condition_index`, `fov`, `status` so repeated runs are byte-stable.

`validate_expected_totals()` sums `pc_count`, `ido_count`, and paired rows from QC and compares them to config. Any mismatch raises before segmentation. The production expectation is 720 PC, 719 IDO, and 719 complete pairs.

- [ ] **Step 5: Implement strict condition mapping**

```python
def apply_condition_mapping(
    raw_manifest: pd.DataFrame,
    mapping: Mapping[Any, Any],
) -> pd.DataFrame:
    """套用經實驗人員確認的 1–8 condition mapping。"""
    normalized = {int(key): value for key, value in mapping.items()}
    if set(normalized) != set(range(1, 9)):
        raise ValueError("condition mapping 必須完整提供編號 1–8。")
    required = {"condition", "ifn_dose", "tnf_dose"}
    for index, values in normalized.items():
        if not isinstance(values, Mapping) or not required.issubset(values):
            raise ValueError(f"condition mapping {index} 缺少 {sorted(required)}。")
    dose_pairs = {
        (float(values["ifn_dose"]), float(values["tnf_dose"]))
        for values in normalized.values()
    }
    labels = {str(values["condition"]) for values in normalized.values()}
    if len(dose_pairs) != 8 or len(labels) != 8:
        raise ValueError("condition mapping 必須提供八組不重複條件。")
    rows = []
    for row in raw_manifest.to_dict(orient="records"):
        condition = normalized[int(row["condition_index"])]
        rows.append({
            **row,
            "condition": str(condition["condition"]),
            "ifn_dose": float(condition["ifn_dose"]),
            "tnf_dose": float(condition["tnf_dose"]),
            "image_key": (
                f"{row['b_id']}_P{int(row['passage'])}_"
                f"C{int(row['condition_index']):02d}_F{int(row['fov']):02d}"
            ),
        })
    return pd.DataFrame(rows).sort_values(
        ["b_id", "passage", "condition_index", "fov"]
    ).reset_index(drop=True)
```

- [ ] **Step 6: Run manifest tests and read-only real-data count check**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_manifest.py -v`

Run: `conda run --no-capture-output -n ki67dtc python -c "from immunity.exp3.run_benchmark import load_config; from immunity.exp3.manifest import scan_datasets; c=load_config('immunity/configs/exp3.yaml'); m,q=scan_datasets(c['datasets']); print(len(m), q.status.value_counts().to_dict())"`

Expected: unit tests pass; real scan prints `719` paired and exactly one `missing_ido` for `B7_P5/C04/F10`.

- [ ] **Step 7: Commit manifest and QC**

```powershell
git add immunity/exp3/manifest.py tests/test_immunity_exp3_manifest.py
git commit -m "feat(immunity): 新增 Exp3 影像配對與 QC"
```

---

### Task 3: 建立 PC-only segmentation 與 mask cache

**Files:**
- Create: `immunity/exp3/phase_features.py`
- Test: `tests/test_immunity_exp3_phase_features.py`

**Interfaces:**
- Produces: `PhaseSegmenter.segment(path: Path) -> tuple[np.ndarray, np.ndarray]`
- Produces: `cache_phase_masks(manifest, cache_dir, config, segmenter=None) -> pd.DataFrame`
- Produces: `load_cached_masks(mask_path: str | Path) -> tuple[np.ndarray, np.ndarray]`
- Produces: `compare_nucleus_masks(pc_mask: np.ndarray, dapi_mask: np.ndarray) -> pd.DataFrame`
- Produces: `run_development_nucleus_validation(config, cache_dir) -> pd.DataFrame`
- Dependency adapter: reuse `CYTO_MODEL_PATH`, `PC_NUC_MODEL_PATH`, model input sizes, and `_filter_small_unpaired_labels` without changing `ki67dtc.img_prep` public behavior.

- [ ] **Step 1: Write a synthetic cache test that never requests DAPI**

```python
from pathlib import Path

import numpy as np
import pandas as pd

from immunity.exp3.phase_features import (
    cache_phase_masks,
    compare_nucleus_masks,
    load_cached_masks,
)


class FakeSegmenter:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def segment(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        self.paths.append(path)
        cell = np.zeros((12, 12), dtype=np.int32)
        nucleus = np.zeros((12, 12), dtype=np.int32)
        cell[2:10, 2:10] = 1
        nucleus[4:7, 4:7] = 1
        return cell, nucleus


def test_cache_phase_masks_uses_only_pc_path(tmp_path: Path) -> None:
    pc_path = tmp_path / "pc.jpg"
    pc_path.touch()
    manifest = pd.DataFrame([{
        "image_key": "B4_P5_C01_F01", "group_id": "B4_P5",
        "pc_path": str(pc_path), "ido_path": str(tmp_path / "ido.jpg"),
    }])
    fake = FakeSegmenter()
    qc = cache_phase_masks(manifest, tmp_path / "feature_cache", {}, fake)
    assert fake.paths == [pc_path]
    assert qc.loc[0, "paired_cells"] == 1
    cell, nucleus = load_cached_masks(qc.loc[0, "mask_path"])
    assert cell.shape == nucleus.shape == (12, 12)


def test_pc_nucleus_reference_comparison_reports_perfect_overlap() -> None:
    pc = np.zeros((12, 12), dtype=np.int32)
    dapi = np.zeros((12, 12), dtype=np.int32)
    pc[3:8, 4:9] = 1
    dapi[3:8, 4:9] = 7
    comparison = compare_nucleus_masks(pc, dapi)
    assert comparison.loc[0, "dice"] == 1.0
    assert comparison.loc[0, "iou"] == 1.0
    assert comparison.loc[0, "area_ratio_pc_to_dapi"] == 1.0
    assert comparison.loc[0, "matched_cell_coverage"] == 1.0
```

- [ ] **Step 2: Run test to verify missing implementation failure**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_phase_features.py::test_cache_phase_masks_uses_only_pc_path -v`

Expected: FAIL with missing `phase_features` import.

- [ ] **Step 3: Implement a cache-local Cellpose adapter**

`PhaseSegmenter` must instantiate the two models once, read only the PC path, resize to the existing model input sizes, restore labels with nearest-neighbor interpolation, then apply the existing paired-aware size filter:

```python
class PhaseSegmenter:
    """以同一張 PC 影像產生 whole-cell 與 nucleus masks。"""

    def __init__(self, config: Mapping[str, Any]) -> None:
        from cellpose import models
        from ki67dtc.img_prep import CYTO_MODEL_PATH, PC_NUC_MODEL_PATH

        use_gpu = str(config.get("device", "gpu")).lower() != "cpu"
        self.cell_model = models.CellposeModel(gpu=use_gpu, pretrained_model=CYTO_MODEL_PATH)
        self.nucleus_model = models.CellposeModel(gpu=use_gpu, pretrained_model=PC_NUC_MODEL_PATH)
        self.config = dict(config)

    def segment(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """分割 PC，回傳配對感知過濾後的 whole-cell 與 nucleus labels。"""
        from cellpose import io
        from ki67dtc.img_prep import (
            CYTO_MODEL_INPUT_SIZE,
            NUC_MODEL_INPUT_SIZE,
            _filter_small_unpaired_labels,
        )

        image = io.imread(path)
        cell = _eval_label_mask(
            self.cell_model, image, CYTO_MODEL_INPUT_SIZE,
            float(self.config.get("cellprob_threshold", 0.0)),
        )
        nucleus = _eval_label_mask(
            self.nucleus_model, image, NUC_MODEL_INPUT_SIZE,
            float(self.config.get("cellprob_threshold", 0.0)),
        )
        filtered = _filter_small_unpaired_labels(
            cell,
            nucleus,
            min_area_ratio=float(self.config.get("min_area_ratio", 0.15)),
            min_area_floor=int(self.config.get("min_area_floor", 30)),
        )
        return filtered.cytoplasm_mask, filtered.nucleus_mask
```

`_eval_label_mask()` must use `cv2.INTER_LINEAR` for the image, call
`model.eval(resized_image, diameter=None, channels=[0, 0], cellprob_threshold=threshold, flow_threshold=0.4, invert=False)`, and use `cv2.INTER_NEAREST` when restoring labels. It returns `np.int32` with exactly the original image height and width.

- [ ] **Step 4: Implement deterministic `.npz` cache and segmentation QC**

`cache_phase_masks()` stores each pair at `feature_cache/masks/<group_id>/<image_key>.npz` with keys `cell_mask` and `nucleus_mask`. Use `ki67dtc.paired_overlay.find_paired_labels()` for counts. With `force: false`, reuse a readable cache; with `force: true`, replace it. Output columns:

```python
SEGMENTATION_QC_COLUMNS = [
    "image_key", "group_id", "mask_path", "cache_status", "status", "error",
    "whole_cell_labels", "nucleus_labels", "paired_cells",
]
```

For different mask shapes, non-2D masks, zero paired cells, or inference exceptions, emit `status="failed"` and the exact exception in `error`, then continue to the next FOV. The orchestrator writes the complete QC table and stops before feature extraction when any required FOV failed. Never read or construct a DAPI path in this Primary path.

- [ ] **Step 5: Implement opt-in development-only DAPI comparison**

`compare_nucleus_masks()` uses Hungarian assignment on pairwise IoU to match nonzero PC and DAPI nucleus labels. For each accepted match with IoU greater than zero, output Dice, IoU, PC/DAPI area ratio, PC and DAPI Feret lengths, plus image-level matched-cell coverage. `run_development_nucleus_validation()` executes only when `development_validation.enabled` is true, pairs old B4 p6 PC/DAPI files with `immunity.build_dataset.build_manifest()`, limits the sorted pairs to `sample_size`, segments PC with `PC_NUC_MODEL_PATH` and DAPI with `DAPI_NUC_MODEL_PATH` using `channels=[3, 3]`, and stores masks under `feature_cache/development_validation/`. Label every row `role="development_only_dapi_reference"`. These metrics never enter feature extraction, tuning, ranking or eligibility.

- [ ] **Step 6: Run phase cache and development-validation tests**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_phase_features.py -v`

Expected: cache test, shape validation test, force/reuse test, zero-pair test, and DAPI-reference overlap tests pass.

- [ ] **Step 7: Commit PC-only segmentation cache**

```powershell
git add immunity/exp3/phase_features.py tests/test_immunity_exp3_phase_features.py
git commit -m "feat(immunity): 新增 Exp3 PC 核質分割快取"
```

---

### Task 4: 提取 33 個 Primary morphology features 與 IDO proxy target

**Files:**
- Modify: `immunity/exp3/phase_features.py`
- Create: `immunity/exp3/feature_sets.py`
- Modify: `tests/test_immunity_exp3_phase_features.py`
- Create: `tests/test_immunity_exp3_feature_sets.py`

**Interfaces:**
- Produces: `extract_basic_cell_features(manifest, segmentation_qc, config) -> tuple[pd.DataFrame, pd.DataFrame]`
- Produces: `extract_features_from_arrays(image_key, phase, ido, cell_mask, nucleus_mask, max_nucleus_outside_fraction, enabled_feature_sets=("basic_median",)) -> tuple[list[dict[str, Any]], dict[str, Any]]`
- Produces: `aggregate_fov_features(cells, manifest, feature_set_names) -> tuple[pd.DataFrame, dict[str, list[str]]]`
- Produces: `validate_phase_predictors(columns: Sequence[str]) -> None`
- Primary base columns are 16 whole-cell geometry + 16 nucleus geometry + one nucleus/cytoplasm area ratio.

- [ ] **Step 1: Write exact 33-feature, target, and leakage tests**

```python
import numpy as np
import pandas as pd
import pytest

from immunity.exp3.feature_sets import (
    PRIMARY_CELL_FEATURES,
    aggregate_fov_features,
    validate_phase_predictors,
)
from immunity.exp3.phase_features import extract_features_from_arrays


def test_primary_registry_contains_exactly_33_features() -> None:
    assert len(PRIMARY_CELL_FEATURES) == 33
    assert "cell__perimeter" in PRIMARY_CELL_FEATURES
    assert "nucleus__feret_length" in PRIMARY_CELL_FEATURES
    assert "nucleus_cytoplasm_area_ratio" in PRIMARY_CELL_FEATURES


def test_feature_extraction_uses_true_cytoplasm_for_ido_target() -> None:
    phase = np.arange(100, dtype=float).reshape(10, 10)
    ido = np.full((10, 10), 5.0)
    cell = np.zeros((10, 10), dtype=np.int32)
    nucleus = np.zeros((10, 10), dtype=np.int32)
    cell[2:8, 2:8] = 1
    nucleus[4:6, 4:6] = 1
    ido[cell == 1] = 20.0
    ido[nucleus == 1] = 100.0
    rows, qc = extract_features_from_arrays("img", phase, ido, cell, nucleus, 0.05)
    assert len(rows) == 1
    assert rows[0]["IDO_score"] == pytest.approx(15.0)
    assert rows[0]["nucleus_cytoplasm_area_ratio"] == pytest.approx(4 / 32)
    assert qc["kept_full_cells"] == 1


def test_phase_predictor_whitelist_rejects_target_and_metadata() -> None:
    validate_phase_predictors(["cell__area__median"])
    for forbidden in ["IDO_score", "IFN_dose", "condition", "pc_path"]:
        with pytest.raises(ValueError, match="phase-only"):
            validate_phase_predictors([forbidden])
```

- [ ] **Step 2: Run tests to verify missing feature registry failure**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_phase_features.py tests/test_immunity_exp3_feature_sets.py -v`

Expected: FAIL importing `PRIMARY_CELL_FEATURES` and `extract_features_from_arrays`.

- [ ] **Step 3: Define the exact Primary registry**

```python
BASIC_GEOMETRY = (
    "area", "compactness", "eccentricity", "extent", "sphericity",
    "major_axis_length", "feret_length", "minor_axis_length", "feret_width",
    "maximum_radius", "mean_radius", "median_radius", "aspect_ratio",
    "perimeter_area_ratio", "perimeter", "solidity",
)
PRIMARY_CELL_FEATURES = (
    *(f"cell__{name}" for name in BASIC_GEOMETRY),
    *(f"nucleus__{name}" for name in BASIC_GEOMETRY),
    "nucleus_cytoplasm_area_ratio",
)
PRIMARY_FOV_FEATURES = tuple(f"{name}__median" for name in PRIMARY_CELL_FEATURES)
```

`validate_phase_predictors()` lowercases each name and rejects tokens `ido`, `ifn`, `tnf`, `dose`, `condition`, `_path`, `filename`, and `delta__`. It also rejects columns not present in one of the feature-set registries.

- [ ] **Step 4: Implement array-level morphology and target extraction**

Use these existing pure measurements through an Exp3 adapter:

```python
from ki67dtc.cell_anal import _geometry_from_measurements, _measure_roi_with_python, _safe_divide
from ki67dtc.paired_overlay import find_paired_labels


def _geometry_values(signal: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float]:
    measured = _geometry_from_measurements(_measure_roi_with_python(signal, mask))
    return {f"{prefix}__{name}": float(measured[name]) for name in BASIC_GEOMETRY}
```

For each `(cell_label, nucleus_label)`, require nucleus outside fraction `<= max_nucleus_outside_fraction` and at least one true cytoplasm pixel. Verify phase and IDO dimensions match. Compute background as the median IDO value outside the union of all whole-cell labels. Compute cell-level `IDO_score = mean(true-cytoplasm IDO) - background_median`, then use the median cell-level score as the FOV target. Return cell rows plus counts for excluded outside nuclei and empty cytoplasm.

- [ ] **Step 5: Implement image-level aggregation with medians only by default**

`aggregate_fov_features()` must preserve `image_key`, `b_id`, `passage`, `group_id`, `condition_index`, `condition`, `ifn_dose`, `tnf_dose`, `fov`, `cell_count`, and median `IDO_score`. For `basic_median`, aggregate only `PRIMARY_CELL_FEATURES` with median; do not calculate IQR. Reject any FOV below `min_cells_per_image`, duplicated `image_key`, non-finite target, or target with fewer than two unique values.

Cache the returned tables as `feature_cache/cell_level_basic.csv` and `feature_cache/image_level_basic.csv`. They are Exp3-internal artifacts and must never be copied into main cleaned CSV／XLSX or legacy result directories.

- [ ] **Step 6: Run Primary feature tests**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_phase_features.py tests/test_immunity_exp3_feature_sets.py -v`

Expected: all feature count, target mask, FOV aggregation, minimum-cell, and whitelist tests pass.

- [ ] **Step 7: Commit Primary features**

```powershell
git add immunity/exp3/phase_features.py immunity/exp3/feature_sets.py tests/test_immunity_exp3_phase_features.py tests/test_immunity_exp3_feature_sets.py
git commit -m "feat(immunity): 提取 Exp3 phase-only 基礎特徵"
```

---

### Task 5: 加入 lazy secondary feature sets 與描述性 ΔMorphology

**Files:**
- Modify: `immunity/exp3/phase_features.py`
- Modify: `immunity/exp3/feature_sets.py`
- Modify: `tests/test_immunity_exp3_feature_sets.py`

**Interfaces:**
- Produces: `FEATURE_SET_REGISTRY: dict[str, FeatureSetSpec]`
- Produces: `paper_style_extras(phase, cell_mask, nucleus_mask) -> dict[str, float]`
- Produces: `calculate_delta_signatures(images, feature_columns) -> pd.DataFrame`
- Secondary sets: `basic_median_iqr` has 66 FOV predictors; `paper_style_median` has no more than 93 FOV predictors.

- [ ] **Step 1: Write lazy-computation and feature-count tests**

```python
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from immunity.exp3.feature_sets import (
    FEATURE_SET_REGISTRY,
    PAPER_STYLE_CELL_FEATURES,
    calculate_delta_signatures,
    validate_phase_predictors,
)
from immunity.exp3.phase_features import extract_features_from_arrays


def test_secondary_feature_counts_are_fixed() -> None:
    assert FEATURE_SET_REGISTRY["basic_median"].predictor_count == 33
    assert FEATURE_SET_REGISTRY["basic_median_iqr"].predictor_count == 66
    assert len(PAPER_STYLE_CELL_FEATURES) == 93


def test_primary_path_does_not_compute_paper_features(monkeypatch) -> None:
    forbidden = Mock(side_effect=AssertionError("paper extras must stay lazy"))
    monkeypatch.setattr("immunity.exp3.phase_features.paper_style_extras", forbidden)
    phase = np.arange(144, dtype=float).reshape(12, 12)
    ido = np.full((12, 12), 5.0)
    cell = np.zeros((12, 12), dtype=np.int32)
    nucleus = np.zeros((12, 12), dtype=np.int32)
    cell[2:10, 2:10] = 1
    nucleus[4:7, 4:7] = 1
    extract_features_from_arrays(
        "img", phase, ido, cell, nucleus, 0.05,
        enabled_feature_sets=["basic_median"],
    )
    forbidden.assert_not_called()


def test_delta_is_descriptive_and_never_a_model_feature() -> None:
    conditions = [
        (1, 0.0, 0.0), (2, 25.0, 0.0), (3, 50.0, 0.0), (4, 100.0, 0.0),
        (5, 0.0, 25.0), (6, 0.0, 50.0), (7, 25.0, 25.0), (8, 25.0, 50.0),
    ]
    rows = [
        {
            "group_id": group_id,
            "condition_index": index,
            "ifn_dose": ifn,
            "tnf_dose": tnf,
            "cell__area__median": float(index + group_offset),
        }
        for group_offset, group_id in enumerate(["B4_P5", "B7_P5"])
        for index, ifn, tnf in conditions
    ]
    delta = calculate_delta_signatures(pd.DataFrame(rows), ["cell__area__median"])
    assert set(delta["comparison_type"]) == {"group_level_unpaired"}
    assert delta.groupby("group_id").size().eq(7).all()
    with pytest.raises(ValueError, match="phase-only"):
        validate_phase_predictors(["delta__cell__area"])
```

- [ ] **Step 2: Run tests to verify feature-set registry failure**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_feature_sets.py -v`

Expected: FAIL because secondary registries do not exist.

- [ ] **Step 3: Define the explicit 93-feature label-free approximation**

The 93 predictors are not presented as an exact fluorescence replication of the paper. Store `replication_scope="partial_label_free_approximation"` in `feature_sets.json`. Use this deterministic composition:

```python
PAPER_STYLE_EXTRA_FEATURES = (
    *(f"cell__zernike_{index:02d}" for index in range(25)),
    *(f"nucleus__zernike_{index:02d}" for index in range(25)),
    "nucleus_cytoplasm_mean_ratio",
    "nucleus_cytoplasm_intden_ratio",
    "nucleus_cytoplasm_raw_intden_ratio",
    "nucleus_cell_intden_ratio",
    "nucleus_cytoplasm_entropy_difference",
    "nucleus_cytoplasm_cv_difference",
    "nucleus_centroid_offset",
    "cell__phase_intensity_cv",
    "nucleus__phase_intensity_cv",
    "cytoplasm__phase_intensity_cv",
)
PAPER_STYLE_CELL_FEATURES = (*PRIMARY_CELL_FEATURES, *PAPER_STYLE_EXTRA_FEATURES)
```

This is exactly `33 + 50 + 6 + 1 + 3 = 93`. Use `_measure_roi_with_python()` for regional intensity values and `_zernike_feature_values_python()` for 25 cell and 25 nucleus moments. Compute the six ratios/differences with `_safe_divide`; compute centroid offset as nucleus-to-cell centroid distance divided by equivalent cell radius.

- [ ] **Step 4: Implement lazy aggregation rules**

`basic_median_iqr` reuses the same 33 cell-level basic values and outputs median plus IQR. `paper_style_median` calls `paper_style_extras()` during cell extraction and outputs medians for exactly the enabled, finite registry columns. If any enabled set has the wrong configured count or a duplicate name, fail before model fitting. Write the exact names, count, aggregation and replication scope to `feature_sets.json`.

When enabled, cache secondary tables separately as `feature_cache/image_level_basic_median_iqr.csv` and `feature_cache/image_level_paper_style_median.csv`; do not widen `image_level_basic.csv`.

- [ ] **Step 5: Implement seven predeclared group-level contrasts**

Use these exact contrasts, with raw delta defined as treated group median minus control group median:

```python
CONTRASTS = (
    ("IFN25_vs_0_at_TNF0", (0.0, 0.0), (25.0, 0.0)),
    ("IFN50_vs_0_at_TNF0", (0.0, 0.0), (50.0, 0.0)),
    ("IFN100_vs_0_at_TNF0", (0.0, 0.0), (100.0, 0.0)),
    ("TNF25_vs_0_at_IFN0", (0.0, 0.0), (0.0, 25.0)),
    ("TNF50_vs_0_at_IFN0", (0.0, 0.0), (0.0, 50.0)),
    ("TNF25_vs_0_at_IFN25", (25.0, 0.0), (25.0, 25.0)),
    ("TNF50_vs_0_at_IFN25", (25.0, 0.0), (25.0, 50.0)),
)
```

For every `group_id`, emit long rows with `contrast_id`, `feature`, `control_median`, `treated_median`, `delta_raw`, `global_iqr`, `delta_scaled_by_global_iqr`, and `comparison_type="group_level_unpaired"`. Missing required conditions are fatal. This table may feed heatmap, PCA, passage trend and feature ordering only; benchmark code must reject `delta__` names.

- [ ] **Step 6: Run secondary and ΔMorphology tests**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_feature_sets.py -v`

Expected: 33／66／93 counts, lazy-computation, seven-contrast, nine-signature, and leakage tests pass.

- [ ] **Step 7: Commit secondary feature isolation**

```powershell
git add immunity/exp3/phase_features.py immunity/exp3/feature_sets.py tests/test_immunity_exp3_feature_sets.py
git commit -m "feat(immunity): 新增 Exp3 延遲特徵與形態差異"
```

---

### Task 6: 建立 grouped outer／inner splits 與安全 metrics

**Files:**
- Create: `immunity/exp3/benchmark.py`
- Create: `tests/test_immunity_exp3_benchmark.py`

**Interfaces:**
- Produces: `OuterSplit(validation: str, fold: str, train_index: np.ndarray, test_index: np.ndarray)`
- Produces: `make_outer_splits(images: pd.DataFrame) -> list[OuterSplit]`
- Produces: `outer_split_manifest(images, splits) -> pd.DataFrame`
- Produces: `make_inner_splits(training: pd.DataFrame, requested: int) -> list[tuple[np.ndarray, np.ndarray]]`
- Produces: `regression_metrics(observed, predicted) -> dict[str, float]`

- [ ] **Step 1: Write split disjointness and constant-vector tests**

```python
import numpy as np
import pandas as pd

from immunity.exp3.benchmark import make_inner_splits, make_outer_splits, regression_metrics
from immunity.exp3.feature_sets import PRIMARY_FOV_FEATURES


def make_grouped_images() -> pd.DataFrame:
    rows = []
    for b_index, b_id in enumerate(["B4", "B7", "B8"]):
        for passage in [5, 6, 7]:
            for condition_index in range(1, 9):
                signal = b_index + passage / 10 + condition_index / 20
                row = {
                    "image_key": f"{b_id}_P{passage}_C{condition_index:02d}_F01",
                    "b_id": b_id,
                    "passage": passage,
                    "group_id": f"{b_id}_P{passage}",
                    "condition_index": condition_index,
                    "condition": f"condition_{condition_index}",
                    "ifn_dose": float(condition_index - 1),
                    "tnf_dose": 0.0,
                    "IDO_score": signal,
                    "cell__area__median": signal,
                    "cell__perimeter__median": signal + 0.1,
                    "cell__feret_length__median": signal + 0.2,
                    "nucleus_cytoplasm_area_ratio__median": signal / 10,
                }
                for feature_index, feature in enumerate(PRIMARY_FOV_FEATURES):
                    row.setdefault(feature, signal + feature_index / 1000)
                rows.append(row)
    return pd.DataFrame(rows)


def test_outer_split_counts_and_group_disjointness() -> None:
    grouped_images = make_grouped_images()
    splits = make_outer_splits(grouped_images)
    counts = {}
    for split in splits:
        counts[split.validation] = counts.get(split.validation, 0) + 1
        train = grouped_images.iloc[split.train_index]
        test = grouped_images.iloc[split.test_index]
        if split.validation == "leave_one_b_out":
            assert set(train.b_id).isdisjoint(test.b_id)
        elif split.validation == "leave_one_passage_out":
            assert set(train.passage).isdisjoint(test.passage)
        elif split.validation == "leave_one_group_out":
            assert set(train.group_id).isdisjoint(test.group_id)
        else:
            assert set(train.condition_index).isdisjoint(test.condition_index)
    assert counts == {
        "leave_one_b_out": 3,
        "leave_one_passage_out": 3,
        "leave_one_group_out": 9,
        "leave_one_condition_out": 8,
    }


def test_inner_splits_never_mix_group_id() -> None:
    grouped_images = make_grouped_images()
    for train_index, test_index in make_inner_splits(grouped_images, requested=5):
        assert set(grouped_images.iloc[train_index].group_id).isdisjoint(
            grouped_images.iloc[test_index].group_id
        )


def test_constant_vectors_keep_errors_but_return_nan_correlations() -> None:
    metrics = regression_metrics([1, 1, 1], [1, 1, 1])
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert np.isnan(metrics["r2"])
    assert np.isnan(metrics["spearman"])
```

- [ ] **Step 2: Run tests to verify missing benchmark failure**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_benchmark.py -v`

Expected: FAIL importing `immunity.exp3.benchmark`.

- [ ] **Step 3: Implement four deterministic outer split families**

```python
@dataclass(frozen=True)
class OuterSplit:
    """保存一個可稽核的 outer train/test split。"""

    validation: str
    fold: str
    train_index: np.ndarray
    test_index: np.ndarray
```

Build folds by sorted unique values of `b_id`, `passage`, `group_id`, and `condition_index`. Verify train and test are non-empty, indices are disjoint, every test index appears exactly once within each validation family, and each family covers all image rows. `outer_split_manifest()` emits one row per image per fold with `validation`, `fold`, `image_key`, `role`, and all grouping metadata.

- [ ] **Step 4: Implement GroupKFold inner splits**

Use `GroupKFold(n_splits=min(requested, training.group_id.nunique()))`. Require at least two unique training groups. Verify train/test `group_id` sets are disjoint for every inner fold and return positional indices relative to the outer-training frame.

- [ ] **Step 5: Implement metrics with explicit undefined behavior**

Compute MAE and RMSE normally. Return `NaN` for R² when the observed vector is constant or shorter than two rows. Return `NaN` for Spearman when either vector is constant or shorter than two rows. Reject unequal lengths, empty arrays, and non-finite predictions.

- [ ] **Step 6: Run split and metric tests**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_benchmark.py -v`

Expected: split counts, complete coverage, group isolation, and metric edge cases pass.

- [ ] **Step 7: Commit split engine**

```powershell
git add immunity/exp3/benchmark.py tests/test_immunity_exp3_benchmark.py
git commit -m "feat(immunity): 新增 Exp3 分組驗證切分"
```

---

### Task 7: 實作多模型 registry 與 nested benchmark

**Files:**
- Modify: `immunity/exp3/benchmark.py`
- Modify: `tests/test_immunity_exp3_benchmark.py`

**Interfaces:**
- Produces: `ModelSpec(name, role, feature_set, scaled, estimator_factory, parameters)`
- Produces: `build_model_registry(config: Mapping[str, Any]) -> dict[str, ModelSpec]`
- Produces: `run_nested_benchmark(images, feature_sets, splits, config, model_names=None) -> BenchmarkResult`
- `BenchmarkResult` contains predictions, fold metrics, hyperparameters, feature importance, and failures.

- [ ] **Step 1: Write registry and shared-split synthetic benchmark tests**

```python
from immunity.exp3.benchmark import build_model_registry, make_outer_splits, run_nested_benchmark


def make_tiny_config() -> dict[str, object]:
    return {
        "seed": 42,
        "inner_splits": 2,
        "max_hyperparameter_candidates": 1,
        "n_jobs": 1,
        "permutation_repeats": 2,
        "tree_estimators": 10,
        "simplicity_order": [
            "paper_linear_3f", "ridge", "elasticnet", "rbf_svr",
            "hist_gradient_boosting", "random_forest", "extra_trees",
        ],
    }


def test_registry_contains_seven_candidates_and_three_diagnostics() -> None:
    registry = build_model_registry(make_tiny_config())
    assert {name for name, spec in registry.items() if spec.role == "candidate"} == {
        "paper_linear_3f", "ridge", "elasticnet", "rbf_svr",
        "random_forest", "extra_trees", "hist_gradient_boosting",
    }
    assert {name for name, spec in registry.items() if spec.role == "diagnostic"} == {
        "dummy_median", "dose_ridge", "dose_plus_morphology_ridge",
    }


def test_all_models_use_identical_outer_split_ids() -> None:
    grouped_images = make_grouped_images()
    tiny_config = make_tiny_config()
    result = run_nested_benchmark(
        grouped_images,
        {"basic_median": list(PRIMARY_FOV_FEATURES)},
        make_outer_splits(grouped_images),
        tiny_config,
    )
    expected = result.fold_metrics.groupby("model")["split_id"].apply(set)
    assert expected.map(len).nunique() == 1
    assert all(value == expected.iloc[0] for value in expected)
    assert result.failures.empty
```

- [ ] **Step 2: Run tests to verify missing registry failure**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_benchmark.py -v`

Expected: FAIL importing `build_model_registry`.

- [ ] **Step 3: Implement the exact model registry**

Use `SimpleImputer(strategy="median", keep_empty_features=True)` for every non-dummy model. Add `StandardScaler()` only for LinearRegression, Ridge, ElasticNet, and RBF-SVR. Wrap non-dummy pipelines in `TransformedTargetRegressor(transformer=StandardScaler())` so target scaling is fit inside the outer training fold.

Production hyperparameters:

```python
PARAMETERS = {
    "ridge": {"regressor__model__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]},
    "elasticnet": {
        "regressor__model__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
        "regressor__model__l1_ratio": [0.1, 0.5, 0.9, 1.0],
    },
    "rbf_svr": {
        "regressor__model__C": [0.1, 1, 10, 100],
        "regressor__model__gamma": ["scale", 0.01, 0.1, 1.0],
        "regressor__model__epsilon": [0.05, 0.1, 0.2],
    },
    "random_forest": {
        "regressor__model__max_depth": [None, 4, 8, 16],
        "regressor__model__min_samples_leaf": [1, 5, 10],
        "regressor__model__max_features": [1.0, "sqrt", 0.5],
    },
    "extra_trees": {
        "regressor__model__max_depth": [None, 4, 8, 16],
        "regressor__model__min_samples_leaf": [1, 5, 10],
        "regressor__model__max_features": [1.0, "sqrt", 0.5],
    },
    "hist_gradient_boosting": {
        "regressor__model__learning_rate": [0.03, 0.1],
        "regressor__model__max_leaf_nodes": [7, 15, 31],
        "regressor__model__min_samples_leaf": [10, 20, 40],
        "regressor__model__l2_regularization": [0, 1, 10],
    },
}
```

Set both tree ensembles to `int(config.get("tree_estimators", 400))` trees with the fold seed; production config omits the override and therefore uses 400. Set ElasticNet `max_iter=50000`. Paper LinearRegression uses exactly `cell__perimeter__median`, `nucleus_cytoplasm_area_ratio__median`, and `cell__feret_length__median` with no tuning. Diagnostic dose columns are `ifn_dose`, `tnf_dose`, and training-created `ifn_x_tnf`; dose-plus-morphology adds all 33 Primary features.

- [ ] **Step 4: Implement deterministic inner tuning**

Use `RandomizedSearchCV` with `scoring="neg_mean_absolute_error"`, `n_iter=min(max_hyperparameter_candidates, total_unique_combinations)`, `random_state=seed`, inner GroupKFold indices, `error_score="raise"`, and `refit=True`. Paper LinearRegression and DummyRegressor skip tuning. Do not catch an error and retry with another split.

- [ ] **Step 5: Implement outer-fold result collection**

For every `OuterSplit × model`, fit on outer train and predict outer test. Store:

```python
OOF_COLUMNS = [
    "validation", "fold", "split_id", "model", "role", "feature_set",
    "image_key", "b_id", "passage", "group_id", "condition_index",
    "condition", "observed_ido_score", "predicted_ido_score",
]
FOLD_METRIC_COLUMNS = [
    "validation", "fold", "split_id", "model", "role", "feature_set",
    "n_train", "n_test", "mae", "rmse", "r2", "spearman",
    "observed_sd", "prediction_sd", "status",
]
HYPERPARAMETER_COLUMNS = [
    "validation", "fold", "split_id", "model", "seed", "best_params_json", "inner_best_mae",
]
FAILURE_COLUMNS = [
    "validation", "fold", "split_id", "model", "exception_type", "message",
]
```

All models must receive the same `split_id = f"{validation}:{fold}"`. A fold failure is recorded once and the benchmark proceeds to other models/folds; the failed model is later ineligible.

- [ ] **Step 6: Add fold-level feature importance**

For standardized linear models, record coefficients from the fitted pipeline as `importance_type="standardized_coefficient"`. For RBF-SVR and tree models, run `sklearn.inspection.permutation_importance()` on that outer test fold with negative MAE, configured repeats, and fold seed; record `importance_type="outer_test_permutation_diagnostic"`. This importance never enters tuning or ranking.

- [ ] **Step 7: Run synthetic all-model benchmark tests**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_benchmark.py -v`

Expected: ten registry models, shared splits, finite predictions, deterministic best parameters, and fold-level importance tests pass under tiny parameter grids.

- [ ] **Step 8: Commit the benchmark engine**

```powershell
git add immunity/exp3/benchmark.py tests/test_immunity_exp3_benchmark.py
git commit -m "feat(immunity): 新增 Exp3 多模型巢狀驗證"
```

---

### Task 8: 實作 ranking、eligibility、no-winner 與 condition-adjusted sensitivity

**Files:**
- Modify: `immunity/exp3/benchmark.py`
- Modify: `tests/test_immunity_exp3_benchmark.py`

**Interfaces:**
- Produces: `rank_phase_models(fold_metrics, predictions, failures, config) -> pd.DataFrame`
- Produces: `select_winner(ranking: pd.DataFrame) -> str | None`
- Produces: `condition_adjust_targets(train, test) -> tuple[pd.Series, pd.Series]`
- Produces: `run_condition_adjusted_sensitivity(images, feature_sets, splits, config, model_names) -> pd.DataFrame`
- Produces: `fit_final_phase_model(images, winner, feature_sets, config, output_dir) -> Path | None`

- [ ] **Step 1: Write eligibility and no-winner tests**

```python
import joblib

from immunity.exp3.benchmark import (
    condition_adjust_targets,
    fit_final_phase_model,
    rank_phase_models,
    select_winner,
)
from immunity.exp3.feature_sets import PRIMARY_FOV_FEATURES


VALIDATIONS = (
    "leave_one_b_out",
    "leave_one_passage_out",
    "leave_one_group_out",
    "leave_one_condition_out",
)
DEFAULT_CONFIG = {
    "simplicity_order": [
        "paper_linear_3f", "ridge", "elasticnet", "rbf_svr",
        "hist_gradient_boosting", "random_forest", "extra_trees",
    ]
}


def make_ineligible_metric_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_rows = []
    for validation in VALIDATIONS:
        for model, mae, r2 in [("ridge", 2.0, -0.5), ("dummy_median", 1.0, -0.1)]:
            metric_rows.append({
                "validation": validation, "fold": "1", "split_id": f"{validation}:1",
                "model": model, "role": "candidate" if model == "ridge" else "diagnostic",
                "mae": mae, "rmse": mae, "r2": r2, "spearman": 0.0,
                "observed_sd": 1.0, "prediction_sd": 0.0, "status": "ok",
            })
            for index in range(3):
                prediction_rows.append({
                    "validation": validation, "fold": "1", "model": model,
                    "observed_ido_score": float(index), "predicted_ido_score": 1.0,
                })
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def make_rank_fixture_with_nine_group_folds() -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_counts = dict(zip(VALIDATIONS, [3, 3, 9, 8], strict=True))
    metric_rows = []
    prediction_rows = []
    for validation, fold_count in fold_counts.items():
        for fold in range(1, fold_count + 1):
            for model, role, mae in [
                ("ridge", "candidate", 0.5),
                ("paper_linear_3f", "candidate", 0.7),
                ("dummy_median", "diagnostic", 1.0),
            ]:
                metric_rows.append({
                    "validation": validation, "fold": str(fold),
                    "split_id": f"{validation}:{fold}", "model": model, "role": role,
                    "mae": mae, "rmse": mae, "r2": 0.2, "spearman": 0.5,
                    "observed_sd": 1.0, "prediction_sd": 0.5, "status": "ok",
                })
                prediction_rows.extend([
                    {"validation": validation, "fold": str(fold), "model": model,
                     "observed_ido_score": 0.0, "predicted_ido_score": 0.1},
                    {"validation": validation, "fold": str(fold), "model": model,
                     "observed_ido_score": 1.0, "predicted_ido_score": 0.9},
                ])
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def make_condition_shift_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame({"condition_index": [1, 1, 2, 2], "IDO_score": [1.0, 3.0, 10.0, 14.0]})
    test = pd.DataFrame({"condition_index": [1, 2], "IDO_score": [5.0, 20.0]})
    return train, test


def test_model_must_pass_gate_instead_of_forcing_winner() -> None:
    metrics, predictions = make_ineligible_metric_fixture()
    ranking = rank_phase_models(metrics, predictions, pd.DataFrame(), {"simplicity_order": []})
    assert not ranking["eligible"].any()
    assert select_winner(ranking) is None
    assert ranking["winner"].eq(False).all()


def test_equal_validation_weights_not_fold_weights() -> None:
    metrics, predictions = make_rank_fixture_with_nine_group_folds()
    ranking = rank_phase_models(metrics, predictions, pd.DataFrame(), DEFAULT_CONFIG)
    ridge = ranking.set_index("model").loc["ridge"]
    assert ridge["leave_one_b_out_weight"] == 0.25
    assert ridge["leave_one_group_out_weight"] == 0.25


def test_condition_adjustment_uses_training_means_only() -> None:
    train, test = make_condition_shift_fixture()
    train_residual, test_residual = condition_adjust_targets(train, test)
    assert train_residual.groupby(train.condition_index).mean().abs().max() < 1e-12
    expected = test.IDO_score - test.condition_index.map(train.groupby("condition_index").IDO_score.mean())
    np.testing.assert_allclose(test_residual, expected)


def test_final_model_is_saved_only_for_eligible_phase_winner(tmp_path) -> None:
    grouped_images = make_grouped_images()
    tiny_config = make_tiny_config()
    feature_sets = {"basic_median": list(PRIMARY_FOV_FEATURES)}
    assert fit_final_phase_model(grouped_images, None, feature_sets, tiny_config, tmp_path) is None
    model_path = fit_final_phase_model(
        grouped_images, "ridge", feature_sets, tiny_config, tmp_path
    )
    bundle = joblib.load(model_path)
    assert bundle["model"] == "ridge"
    assert bundle["feature_columns"] == list(PRIMARY_FOV_FEATURES)
    assert not any("dose" in name.lower() or "ido" in name.lower() for name in bundle["feature_columns"])
```

- [ ] **Step 2: Run tests to verify missing ranking failure**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_benchmark.py -v`

Expected: FAIL importing ranking functions.

- [ ] **Step 3: Implement equal-weight average ranking**

For each validation, aggregate each candidate model with median fold MAE, median fold R² and median fold Spearman, then rank MAE ascending. Overall rank is the arithmetic mean of the four validation ranks; do not weight the nine-fold validation more heavily. Diagnostic models never receive a phase-model rank.

- [ ] **Step 4: Implement the five eligibility checks**

A candidate is eligible only when all are true:

1. Its median MAE is lower than Dummy in at least three of four validations.
2. Its median fold R² is greater than zero in at least two validations.
3. It has no failed or missing outer folds.
4. Its feature whitelist contains no IDO, dose, condition, path, filename or ΔMorphology column.
5. For every validation, OOF prediction SD is at least 5% of observed target SD.

Store each result as boolean columns plus `ineligibility_reasons_json`. If no candidate passes, `select_winner()` returns `None` and the report status is `no_eligible_phase_only_model`.

- [ ] **Step 5: Implement tie-break rules**

Candidates whose average ranks differ by `<= 0.25` are tied. Resolve in order: lower worst validation rank, lower leave-one-B-out MAE, higher overall OOF Spearman, then the first model in configured `simplicity_order`. Store every tie-break field in `model_ranking.csv`.

- [ ] **Step 6: Implement condition-adjusted sensitivity**

Run candidate phase-only models on leave-one-B-out, leave-one-passage-out and leave-one-group-out only. Within each outer fold, calculate condition means from outer training targets, subtract them from both training and test targets, and run the same nested pipeline against residual target. Do not residualize leave-one-condition-out because the held-out condition has no training mean. Save metrics with `analysis="training_condition_mean_residual"`; never use them in winner ranking.

- [ ] **Step 7: Fit and save only an eligible phase-only winner**

If `select_winner()` returns a model name, rerun grouped inner tuning on all FOV rows using `group_id`, the same feature whitelist, MAE scoring and deterministic seed. Save `models/{winner}.joblib` containing the fitted target-transformed pipeline, exact feature names, target label `image-level background-corrected IDO proxy`, config hash, manifest hash, model name and training scope. Write `final_model.json` with best parameters and paths. If there is no eligible winner, create neither model file nor final-model metadata. Diagnostic dose models can never be passed to this function.

- [ ] **Step 8: Run ranking, sensitivity and final-model tests**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_benchmark.py -v`

Expected: eligibility, no-winner, equal weights, tie-break, training-only residualization, LOCO exclusion and phase-only model bundle tests pass.

- [ ] **Step 9: Commit selection and sensitivity analysis**

```powershell
git add immunity/exp3/benchmark.py tests/test_immunity_exp3_benchmark.py
git commit -m "feat(immunity): 新增 Exp3 模型篩選與敏感度分析"
```

---

### Task 9: 產生固定結果檔、圖表與白話實驗記錄

**Files:**
- Create: `immunity/exp3/reporting.py`
- Create: `tests/test_immunity_exp3_reporting.py`

**Interfaces:**
- Produces: `write_result_tables(output_dir: Path, tables: Mapping[str, pd.DataFrame]) -> None`
- Produces: `write_figures(output_dir: Path, artifacts: Mapping[str, Any]) -> list[Path]`
- Produces: `write_experiment_record(output_dir: Path, context: Mapping[str, Any]) -> Path`

- [ ] **Step 1: Write fixed-output and no-winner wording tests**

```python
from pathlib import Path

import pandas as pd

from immunity.exp3.reporting import REQUIRED_TABLES, write_experiment_record, write_result_tables


def make_report_tables() -> dict[str, pd.DataFrame]:
    return {name: pd.DataFrame([{"status": "synthetic"}]) for name in REQUIRED_TABLES}


def make_no_winner_context() -> dict[str, object]:
    return {
        "status": "no_eligible_phase_only_model",
        "winner": None,
        "raw_pc": 720,
        "raw_ido": 719,
        "complete_pairs": 719,
        "ranking": pd.DataFrame([{
            "model": "ridge", "eligible": False,
            "ineligibility_reasons_json": '["MAE 未優於 Dummy"]',
        }]),
        "limitations": ["IDO fluorescence is an IDO proxy."],
        "metadata": {"git_commit": "synthetic", "seed": 42},
    }


def test_result_writer_creates_exact_required_tables(tmp_path: Path) -> None:
    write_result_tables(tmp_path, make_report_tables())
    assert {path.name for path in tmp_path.glob("*.csv")} == set(REQUIRED_TABLES)


def test_record_states_no_winner_without_overclaiming(tmp_path: Path) -> None:
    path = write_experiment_record(tmp_path, make_no_winner_context())
    text = path.read_text(encoding="utf-8")
    assert "沒有符合門檻的 phase-only 模型" in text
    assert "IDO proxy" in text
    assert "整體免疫抑制能力" not in text
    assert "證據" in text and "推論" in text and "限制" in text
```

- [ ] **Step 2: Run tests to verify missing reporting failure**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_reporting.py -v`

Expected: FAIL importing `immunity.exp3.reporting`.

- [ ] **Step 3: Implement the required table contract**

```python
REQUIRED_TABLES = (
    "data_manifest.csv",
    "pairing_qc.csv",
    "segmentation_qc.csv",
    "outer_splits.csv",
    "oof_predictions.csv",
    "fold_metrics.csv",
    "hyperparameters.csv",
    "model_ranking.csv",
    "feature_importance.csv",
    "model_failures.csv",
    "condition_adjusted_metrics.csv",
    "pc_nucleus_dapi_validation.csv",
    "morphology_delta_signatures.csv",
)
```

Use atomic writes: write each CSV to a sibling `.tmp`, then replace the destination. JSON outputs are `feature_sets.json` and `run_metadata.json`. Reject unexpected output directories through `resolve_exp3_output_dir()` before any write.

- [ ] **Step 4: Implement required figures with `matplotlib.use("Agg")`**

Create these files under `figures/`:

1. `model_validation_rank_heatmap.png`
2. `fold_mae_distributions.png`
3. `observed_vs_predicted.png`
4. `diagnostic_model_comparison.png`
5. `feature_importance_stability.png`
6. `residuals_by_b_passage_condition.png`
7. `morphology_delta_heatmap.png`
8. `morphology_delta_pca.png`

When there is no eligible winner, the observed-vs-predicted plot may show the best-ranked ineligible candidate only if its title includes `Exploratory — ineligible`; never label it winner. Plot functions must close every figure.

- [ ] **Step 5: Implement the experiment record sections**

`EXPERIMENT_RECORD.md` must contain, in order:

1. `結論狀態` — winner name or no-eligible-model.
2. `資料與 QC` — raw PC/IDO counts, 719 complete pairs, exclusions, segmentation pass rate, and optional development-only PC-vs-DAPI nucleus metrics.
3. `Primary benchmark` — four validation tables with MAE, RMSE, R², Spearman.
4. `Eligibility gate` — pass/fail reason for every candidate.
5. `Dose confounding diagnostics` — Dummy, dose-only, dose-plus-morphology comparison.
6. `Condition-adjusted sensitivity` — explicitly excluded from ranking.
7. `ΔMorphology` — descriptive only; nine biological-group signatures are not 719 independent samples.
8. `證據`、`推論`、`限制` — separate subsections.
9. `Reproducibility` — commit, dirty status, config hash, manifest hash, Python/package versions, seeds and runtime.
10. `下一批資料需求` — verified condition mapping, donor/lot metadata, and independent functional assay.

- [ ] **Step 6: Run reporting and figure smoke tests**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_reporting.py -v`

Expected: required table names, atomic writes, no-winner language, all eight non-empty PNGs, and figure-close tests pass.

- [ ] **Step 7: Commit reporting**

```powershell
git add immunity/exp3/reporting.py tests/test_immunity_exp3_reporting.py
git commit -m "feat(immunity): 產生 Exp3 圖表與實驗記錄"
```

---

### Task 10: 串接 CLI、Round 2 ablation 與 synthetic end-to-end test

**Files:**
- Modify: `immunity/exp3/run_benchmark.py`
- Create: `immunity/exp3/README.md`
- Modify: `tests/test_immunity_exp3_cli.py`

**Interfaces:**
- Produces: `run_benchmark(config: Mapping[str, Any], smoke_fovs_per_condition: int | None = None) -> Path`
- Produces: `main() -> int`
- Round 1: all seven phase candidates on `basic_median` plus three diagnostics.
- Round 2: only Round 1 first- and second-ranked candidate, and only explicitly enabled secondary sets; results remain exploratory.
- If `paper_linear_3f` is among the top two, keep its fixed three paper predictors as a reference row and mark secondary sets `not_applicable`; never silently turn it into a 66／93-feature linear model.

- [ ] **Step 1: Write empty-mapping blocker and synthetic end-to-end tests**

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import immunity.exp3.run_benchmark as run_module
from immunity.exp3.run_benchmark import run_benchmark


SYNTHETIC_MAPPING = {
    1: {"condition": "IFN0_TNF0", "ifn_dose": 0.0, "tnf_dose": 0.0},
    2: {"condition": "IFN25_TNF0", "ifn_dose": 25.0, "tnf_dose": 0.0},
    3: {"condition": "IFN50_TNF0", "ifn_dose": 50.0, "tnf_dose": 0.0},
    4: {"condition": "IFN100_TNF0", "ifn_dose": 100.0, "tnf_dose": 0.0},
    5: {"condition": "IFN0_TNF25", "ifn_dose": 0.0, "tnf_dose": 25.0},
    6: {"condition": "IFN0_TNF50", "ifn_dose": 0.0, "tnf_dose": 50.0},
    7: {"condition": "IFN25_TNF25", "ifn_dose": 25.0, "tnf_dose": 25.0},
    8: {"condition": "IFN25_TNF50", "ifn_dose": 25.0, "tnf_dose": 50.0},
}


def tiny_benchmark_config() -> dict[str, object]:
    return {
        "seed": 42,
        "inner_splits": 2,
        "max_hyperparameter_candidates": 1,
        "n_jobs": 1,
        "permutation_repeats": 2,
        "tree_estimators": 10,
        "simplicity_order": [
            "paper_linear_3f", "ridge", "elasticnet", "rbf_svr",
            "hist_gradient_boosting", "random_forest", "extra_trees",
        ],
    }


class SyntheticSegmenter:
    def segment(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        condition = int(path.stem.split("-", maxsplit=1)[0])
        edge = 9 + condition // 2
        cell = np.zeros((16, 16), dtype=np.int32)
        nucleus = np.zeros((16, 16), dtype=np.int32)
        cell[2:edge, 2:edge] = 1
        nucleus[4:4 + max(2, condition // 2), 4:4 + max(2, condition // 2)] = 1
        return cell, nucleus


def write_synthetic_pair(root: Path, condition: int, fov: int, ido_value: int) -> None:
    (root / "PC").mkdir(parents=True, exist_ok=True)
    (root / "IDO").mkdir(parents=True, exist_ok=True)
    phase = np.tile(np.arange(16, dtype=np.uint8), (16, 1))
    ido = np.full((16, 16), 5, dtype=np.uint8)
    edge = 9 + condition // 2
    ido[2:edge, 2:edge] = ido_value
    Image.fromarray(phase).save(root / "PC" / f"{condition}-phase-100X-{fov}.png")
    Image.fromarray(ido).save(root / "IDO" / f"{condition}-IDO-100X-{fov}.png")


def base_test_config(output_dir: Path) -> dict[str, object]:
    return {
        "condition_mapping": SYNTHETIC_MAPPING,
        "segmentation": {"device": "cpu", "force": True, "min_cells_per_image": 1},
        "development_validation": {"enabled": False},
        "feature_sets": {"enabled": ["basic_median"], "paper_style_max_features": 93},
        "benchmark": tiny_benchmark_config(),
        "output": {"dir": str(output_dir)},
        "_output_dir": str(output_dir),
        "_config_path": "synthetic",
    }


def make_scan_only_config(tmp_path: Path, condition_mapping: dict) -> dict[str, object]:
    root = tmp_path / "scan" / "B4-P5"
    write_synthetic_pair(root, 1, 1, 20)
    config = base_test_config(tmp_path / "output")
    config.update({
        "datasets": [{"input_dir": str(root), "b_id": "B4", "passage": 5}],
        "expected_totals": {"pc": 1, "ido": 1, "paired": 1},
        "condition_mapping": condition_mapping,
    })
    return config


def make_complete_synthetic_config(tmp_path: Path) -> tuple[dict[str, object], SyntheticSegmenter]:
    specs = []
    for b_id in ["B4", "B7", "B8"]:
        for passage in [5, 6, 7]:
            root = tmp_path / "data" / f"{b_id}-P{passage}"
            specs.append({"input_dir": str(root), "b_id": b_id, "passage": passage})
            for condition in range(1, 9):
                write_synthetic_pair(root, condition, 1, 15 + condition + passage)
    config = base_test_config(tmp_path / "output")
    config.update({
        "datasets": specs,
        "expected_totals": {"pc": 72, "ido": 72, "paired": 72},
    })
    return config, SyntheticSegmenter()


def test_empty_real_mapping_writes_pairing_qc_then_stops(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_module, "EXP3_OUTPUT_ROOT", (tmp_path / "output").resolve())
    config = make_scan_only_config(tmp_path, condition_mapping={})
    with pytest.raises(ValueError, match="condition mapping"):
        run_benchmark(config)
    assert (Path(config["_output_dir"]) / "pairing_qc.csv").is_file()
    assert not (Path(config["_output_dir"]) / "fold_metrics.csv").exists()


def test_synthetic_pipeline_writes_recomputable_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_module, "EXP3_OUTPUT_ROOT", (tmp_path / "output").resolve())
    config, fake_segmenter = make_complete_synthetic_config(tmp_path)
    monkeypatch.setattr("immunity.exp3.run_benchmark.PhaseSegmenter", lambda _: fake_segmenter)
    record = run_benchmark(config)
    assert record.name == "EXPERIMENT_RECORD.md"
    output = record.parent
    metrics = pd.read_csv(output / "fold_metrics.csv")
    ranking = pd.read_csv(output / "model_ranking.csv")
    assert set(metrics.validation) == {
        "leave_one_b_out", "leave_one_passage_out",
        "leave_one_group_out", "leave_one_condition_out",
    }
    assert ranking["overall_rank"].notna().all()
```

- [ ] **Step 2: Run CLI tests to verify missing orchestration failure**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_cli.py -v`

Expected: FAIL because `run_benchmark()` is not implemented.

- [ ] **Step 3: Implement stage-ordered orchestration**

`run_benchmark()` executes in this exact order:

1. Resolve the Exp3 output boundary and create `feature_cache/` plus `figures/`; when a smoke limit is present, use the isolated child directory `immunity/outputs/exp3/smoke/` so smoke tables never masquerade as full-run results.
2. Scan all datasets, immediately write `pairing_qc.csv`, and enforce configured raw/paired totals.
3. Apply optional smoke limit deterministically by the first configured number of FOVs within every `group_id × condition_index`; a value of 1 therefore retains all 8 conditions in all 9 groups for 72 images.
4. Apply condition mapping; if missing, stop here with no model outputs.
5. Write `data_manifest.csv`, segment/reuse cache, optionally run the old-DAPI development validation, then extract enabled features and target.
6. Write `feature_sets.json`, `segmentation_qc.csv`, and ΔMorphology table.
7. Build and write outer splits; run Round 1 and condition-adjusted sensitivity.
8. Rank Round 1, fit/save an eligible phase-only winner, and save no model when the gate returns no winner.
9. If secondary sets are enabled, run only the top two Round 1 models and label rows `round="exploratory_round_2"`.
10. Write all fixed tables including failures and development validation, figures, metadata and `EXPERIMENT_RECORD.md`.

Never delete or overwrite paths outside the resolved Exp3 output root. Reusing an existing mask or feature cache must be recorded in QC and metadata.

- [ ] **Step 4: Add reproducibility metadata**

Record UTC start/end timestamps, elapsed seconds, `git rev-parse HEAD`, `git status --porcelain`, SHA-256 of config and manifest, Python/platform versions, NumPy/pandas/scikit-learn/Cellpose versions, all seeds, enabled feature sets, smoke limit, input counts, cache hits, and failure count. Hash files by bytes after deterministic CSV writing.

- [ ] **Step 5: Implement CLI logging and exit behavior**

`main()` loads config, sets the project working directory, tees stdout/stderr to `immunity/outputs/exp3/run.log`, invokes `run_benchmark()`, and returns 0 only after the record exists. Configuration, mapping, pairing, segmentation or model errors return non-zero through `SystemExit` and retain the already-written QC/log evidence.

- [ ] **Step 6: Document the mapping gate and execution commands**

`immunity/exp3/README.md` must lead with:

```markdown
# Exp3 Phase-only Morphology Benchmark

> 目前下一步：先向實驗人員取得 condition index 1–8 對應的 IFN/TNF 劑量，填入
> `immunity/configs/exp3.yaml`。在 mapping 完整前，程式只產生 pairing QC，不會訓練模型。

## 正式執行

```powershell
conda run --no-capture-output -n ki67dtc python -m immunity.exp3.run_benchmark --config immunity/configs/exp3.yaml
```

## 72-image smoke run

```powershell
conda run --no-capture-output -n ki67dtc python -m immunity.exp3.run_benchmark --config immunity/configs/exp3.yaml --smoke-fovs-per-condition 1
```
```

Also document that Primary is 33 medians, 66／93 are opt-in, IDO is target only, and ΔMorphology is descriptive only.

- [ ] **Step 7: Run synthetic integration and dedicated Exp3 suite**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_manifest.py tests/test_immunity_exp3_phase_features.py tests/test_immunity_exp3_feature_sets.py tests/test_immunity_exp3_benchmark.py tests/test_immunity_exp3_reporting.py tests/test_immunity_exp3_cli.py -v`

Expected: all Exp3 unit and synthetic integration tests pass; no real Cellpose inference runs.

- [ ] **Step 8: Commit CLI and integration**

```powershell
git add immunity/exp3/run_benchmark.py immunity/exp3/README.md tests/test_immunity_exp3_cli.py
git commit -m "feat(immunity): 串接 Exp3 多模型實驗流程"
```

---

### Task 11: 執行完整 regression、隔離稽核與實驗前 smoke gate

**Files:**
- Verify only: `main.py`
- Verify only: `immunity/build_dataset.py`
- Verify only: `immunity/run_experiment.py`
- Verify only: `ki67dtc/`
- Verify only: `tests/`

**Interfaces:**
- Produces no code unless verification finds a defect.
- Gate: no full 719-image experiment until condition mapping is populated and the 72-image smoke run passes.

- [ ] **Step 1: Verify no forbidden core files changed in the implementation branch**

Run: `$implementationBase = git merge-base HEAD main; git diff --name-only "$implementationBase..HEAD"`

Expected: changed application files are limited to `immunity/exp3/`, `immunity/configs/exp3.yaml`, Exp3 tests, and this plan-approved documentation.

- [ ] **Step 2: Run the complete test suite**

Run: `conda run --no-capture-output -n ki67dtc python -m pytest -q`

Expected: zero failures; pre-existing main, GUI, workbook and immunity tests remain green.

- [ ] **Step 3: Run import and output isolation checks**

Run: `conda run --no-capture-output -n ki67dtc python -c "import main; import immunity.exp3; from immunity.build_dataset import morphology_feature_columns; assert len(morphology_feature_columns()) == 33; assert 'paper_style_median' not in str(main.build_parser()._actions); print('isolation-ok')"`

Expected: `isolation-ok`.

- [ ] **Step 4: Run the real read-only pairing gate**

Run: `conda run --no-capture-output -n ki67dtc python -m immunity.exp3.run_benchmark --config immunity/configs/exp3.yaml`

Expected before mapping is supplied: non-zero exit mentioning the missing `condition mapping`; `immunity/outputs/exp3/pairing_qc.csv` exists with 719 paired rows and one B7-P5 missing-IDO row; no `fold_metrics.csv` exists.

- [ ] **Step 5: After verified mapping is supplied, run the 72-image smoke experiment**

Run: `conda run --no-capture-output -n ki67dtc python -m immunity.exp3.run_benchmark --config immunity/configs/exp3.yaml --smoke-fovs-per-condition 1`

Expected: non-empty `EXPERIMENT_RECORD.md`, all required CSV/JSON outputs, eight figures, no failed folds, and metadata recording `smoke_fovs_per_condition: 1` plus 72 analyzed images.

All smoke artifacts must be under `immunity/outputs/exp3/smoke/`; the full-run root must not contain smoke metrics or ranking files.

- [ ] **Step 6: Inspect smoke evidence before authorizing the full experiment**

Open and verify:

- `pairing_qc.csv`: missing/duplicate/parse counts match expectations.
- `segmentation_qc.csv`: each smoke image has at least three paired cells.
- `model_ranking.csv`: eligibility and no-winner logic is recomputable.
- `EXPERIMENT_RECORD.md`: uses `IDO proxy`, separates evidence/inference/limits, and does not overclaim biological validation.

- [ ] **Step 7: Commit only verification-driven fixes, if any**

If no defect is found, do not create an empty commit. If a defect is found, add its failing regression test first, make the minimal fix, rerun Steps 2–6, inspect the diff, then use the narrowest AngularJS type and scope in Traditional Chinese.

---

## Full Experiment Authorization Gate

The full 719-pair run is authorized only after all four conditions are true:

1. Condition index 1–8 mapping has been verified by the experiment owner and committed in `immunity/configs/exp3.yaml`.
2. The dedicated Exp3 suite and complete repository suite have zero failures.
3. The 72-image smoke run has complete QC, metrics, figures and experiment record.
4. No forbidden main/core files appear in the implementation diff.

Then run:

```powershell
conda run --no-capture-output -n ki67dtc python -m immunity.exp3.run_benchmark --config immunity/configs/exp3.yaml
```

Do not reuse smoke ranking as the final result. The full run must write its own manifest hash, split manifest, OOF predictions, ranking, eligibility result and `EXPERIMENT_RECORD.md`.
