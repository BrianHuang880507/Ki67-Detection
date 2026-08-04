"""驗證 Exp3 CLI 骨架與既有主流程的隔離邊界。"""

from pathlib import Path

import pytest

import main
from immunity.build_dataset import morphology_feature_columns
from immunity.exp3.run_benchmark import load_config, resolve_exp3_output_dir


def test_exp3_output_must_stay_under_dedicated_root(tmp_path: Path) -> None:
    """Exp3 的輸出目錄只能位於專用根目錄中。"""
    allowed = resolve_exp3_output_dir("immunity/outputs/exp3/smoke")

    assert allowed.as_posix().endswith("immunity/outputs/exp3/smoke")
    with pytest.raises(ValueError, match="immunity/outputs/exp3"):
        resolve_exp3_output_dir("immunity/outputs/b4_p6")
    with pytest.raises(ValueError, match="immunity/outputs/exp3"):
        resolve_exp3_output_dir(tmp_path)


def test_importing_exp3_does_not_expand_main_contract() -> None:
    """Exp3 匯入不可改變主流程 CLI 或 predictor schema。"""
    destinations = {action.dest for action in main.build_parser()._actions}

    assert destinations == {
        "help",
        "data_folder",
        "device",
        "nuc_source",
        "fluor_analy",
        "ki67",
        "ki67_backend",
        "feature_backend",
        "clean_temp",
        "xlsx_version",
    }
    predictors = morphology_feature_columns()
    assert len(predictors) == 33
    assert not any("Zernike" in name or "__median" in name for name in predictors)


def test_exp3_config_loads_without_running_pipeline() -> None:
    """Exp3 設定檔可在不執行 pipeline 的前提下讀取。"""
    config = load_config("immunity/configs/exp3.yaml")

    assert len(config["datasets"]) == 9
    assert config["expected_totals"] == {"pc": 720, "ido": 719, "paired": 719}
    assert config["condition_mapping"] == {}
    assert config["feature_sets"]["enabled"] == ["basic_median"]
