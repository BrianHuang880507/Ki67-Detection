"""驗證 Exp3 Round 2 CLI 的設定與輸出邊界。"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

import immunity.exp3.run_round2_paper93 as run_module
from immunity.exp3.run_round2_paper93 import (
    load_round2_config,
    resolve_round2_output_dir,
)


def _create_directory_link(link: Path, target: Path) -> None:
    """建立目錄 symlink，必要時在 Windows 改用 junction。"""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_round2_config_locks_exact_models_features_seed_and_frozen_hashes() -> None:
    """設定不得偏離 Round 2 的模型、特徵、種子與凍結證據。"""
    config = load_round2_config("immunity/configs/exp3_round2_paper93.yaml")

    assert config["models"] == ["extra_trees", "random_forest"]
    assert config["features"] == {
        "feature_set": "paper_style_median",
        "predictor_count": 93,
        "extra_count": 60,
        "min_finite_cells_per_feature": 3,
        "numeric_atol": 1e-12,
    }
    assert config["benchmark"]["seed"] == 20260804
    assert config["round1"]["expected"]["analyzed_images"] == 693
    assert config["round1"]["expected"]["valid_cells"] == 23976
    assert config["round1"]["roster_sha256"] == (
        "A8333F12E1591D9E4C4174F5C6FE13DD31550124B19AEA3522F4D0F812E0E426"
    )
    assert len(config["round1"]["artifact_sha256"]) == 14


@pytest.mark.parametrize(
    "mutation",
    [
        "manifest_hash",
        "config_hash",
        "roster_sha256",
        "artifact_name",
        "artifact_hash",
    ],
)
def test_round2_config_rejects_mutated_frozen_round1_identity(
    tmp_path: Path,
    mutation: str,
) -> None:
    """即使格式正確，Round 1 的凍結 identity 不可遭竄改。"""
    source = Path("immunity/configs/exp3_round2_paper93.yaml")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    if mutation == "manifest_hash":
        config["round1"]["expected"]["manifest_hash"] = "0" * 64
    elif mutation == "config_hash":
        config["round1"]["expected"]["config_hash"] = "0" * 64
    elif mutation == "roster_sha256":
        config["round1"]["roster_sha256"] = "0" * 64
    elif mutation == "artifact_name":
        artifacts = config["round1"]["artifact_sha256"]
        artifacts["unexpected.csv"] = artifacts.pop("pairing_qc.csv")
    else:
        config["round1"]["artifact_sha256"]["pairing_qc.csv"] = "0" * 64
    path = tmp_path / "mutated-round2.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Round 2"):
        load_round2_config(path)


def test_round2_output_resolver_accepts_only_formal_root_and_smoke_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式根目錄與其 smoke 子目錄是唯一允許的輸出位置。"""
    root = (tmp_path / "immunity" / "outputs" / "exp3" / "round2_paper93").resolve()
    monkeypatch.setattr(run_module, "ROUND2_OUTPUT_ROOT", root)

    assert resolve_round2_output_dir(root, smoke=False) == root
    assert resolve_round2_output_dir(root, smoke=True) == root / "smoke"


@pytest.mark.parametrize(
    "relative",
    ["immunity/outputs/exp3", "immunity/outputs/exp3/other", "legacy/results"],
)
def test_round2_output_resolver_rejects_parent_and_legacy_paths(
    relative: str,
) -> None:
    """父目錄、其他實驗與舊輸出路徑均不可作為 Round 2 輸出。"""
    with pytest.raises(ValueError, match="round2_paper93"):
        resolve_round2_output_dir(relative, smoke=False)


def test_round2_output_resolver_rejects_link_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正式根目錄本身若為 link/junction，必須拒絕路徑逃逸。"""
    root = tmp_path / "immunity" / "outputs" / "exp3" / "round2_paper93"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.parent.mkdir(parents=True)
    _create_directory_link(root, outside)
    monkeypatch.setattr(run_module, "ROUND2_OUTPUT_ROOT", root)

    with pytest.raises(ValueError, match="symlink|junction"):
        resolve_round2_output_dir(root, smoke=False)


def test_round2_output_resolver_rejects_existing_smoke_child_link_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """既有 smoke 子目錄若為 link/junction，必須拒絕路徑逃逸。"""
    root = (tmp_path / "immunity" / "outputs" / "exp3" / "round2_paper93").resolve()
    outside = tmp_path / "outside"
    root.mkdir(parents=True)
    outside.mkdir()
    _create_directory_link(root / "smoke", outside)
    monkeypatch.setattr(run_module, "ROUND2_OUTPUT_ROOT", root)

    with pytest.raises(ValueError, match="symlink|junction"):
        resolve_round2_output_dir(root, smoke=True)
