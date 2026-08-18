# Exp3 Round 2 Paper-style 93 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 建立完全隔離的 Exp3 Round 2 runner，以相同 693 張 FOV 與 frozen outer splits，只新訓練 Extra Trees／Random Forest 的 93-feature 版本，並與唯讀的 Round 1 33-feature evidence 比較。

**Architecture:** 新增 Round 2 專用 evidence、feature、benchmark、reporting 與 CLI modules；現有 Round 1 runner、main.py、GUI 與 legacy schema 維持不變。Round 1 artifacts 與 mask cache 一律唯讀且以 SHA-256、semantic identity、cell roster 與 split membership fail-closed；成功結果只發布到 immunity/outputs/exp3/round2_paper93/。

**Tech Stack:** Python 3.10、pandas、NumPy、scikit-learn、OpenCV、PyYAML、pytest、既有 immunity.exp3 phase segmentation／benchmark utilities。

## Global Constraints

- 工作目錄固定為 D:/Project/Ki67-Detection/.worktrees/exp3-multimodel-benchmark，branch 為 codex/exp3-multimodel-benchmark。
- 現有未提交檔 immunity/configs/exp3.yaml 與 tests/test_immunity_exp3_cli.py 屬於既存工作；任何 Round 2 commit 都不得 stage 或修改這兩檔。
- 每次 commit 都使用精確 git add 路徑；禁止 git add .、git add -A 或任何會夾帶既存 dirty files 的操作。
- Predictor 僅能是 93 個 registered phase-derived morphology columns；IDO、IFN、TNF、dose、condition、path、filename、donor、b_id、batch 或 delta 不可進入 predictor matrix。
- IDO_score 只作 frozen FOV target；不建立免疫力、功能性 assay 或 donor label。
- 新 model fits 固定只有 extra_trees 與 random_forest，feature set 固定為 paper_style_median；不得重跑 33-feature × 7-model Round 1。
- 正式資料固定 693 FOV、23,976 frozen valid cells、26 exclusions、seed 20260804 與 23 個 outer folds（3 B、3 passage、9 group、8 condition）。
- Mask cache 缺少、PC hash 改變、embedded provenance 不符或內容損毀時停止；不得呼叫 segmentation、重寫 cache、自動新增 exclusion、降低 min cells 或強制配對。
- 每個新增 extra 在每張 FOV 至少要有 3 個 finite cell observations；33 個 basic predictors 與 IDO_score 使用 rtol=0、atol=1e-12、equal_nan=False 鎖定。
- 正式輸出只能位於 immunity/outputs/exp3/round2_paper93/；smoke 只能位於其 smoke/ child；不得搬動或覆寫 Round 1 artifacts。
- Smoke 只驗證流程，不形成 33 vs 93 科學結論；正式 run 才套用 693-row、完整 frozen membership 與唯一 recommendation gates。
- 所有 public function／class 與複雜 I/O 使用繁體中文 Google-style docstring。
- Commit message 使用 AngularJS convention，type／scope 保持英文，subject 使用繁體中文。
- 所有 Python 與 pytest 命令都透過 conda run --no-capture-output -n ki67dtc 執行。

## Frozen Round 1 Evidence

Round 2 config 必須 pin 下列 artifacts；EXPERIMENT_RECORD.md 不作 modeling evidence，因其曾在 run 後人工補充 exclusion QC。

| Artifact | SHA-256 |
|---|---|
| pairing_qc.csv | 0FA953ECEB35678AAB0209214EB1EE267E844A0B81E649B0C0ACCA487246E0EB |
| data_manifest.csv | B4508AAF830F4C456759D1F02458606AA28ED385856575C6FB99A5AC2E163771 |
| segmentation_qc.csv | 2A1E8969FC8D0A81D3DD737EF99DF57F58CE3E1FAF0206E833463F2DC9D28C1E |
| outer_splits.csv | C7E9E26A9AD0C6F73D8F14C5FA4C73747F6E58865AAC963BF8E16E527FA05A3E |
| feature_cache/image_level_basic.csv | DAD479258D08846D05A43F551B0001A52D96FA5E2E8EA6A98645723931AA0D14 |
| feature_cache/cell_level_basic.csv | 7D999A728D848CCB61A3910CB5230C9EFDE6D7E1FA14F22DE792990EE4AD7850 |
| fold_metrics.csv | 8CE0DF3866822A54C98CB1CD95DBEC3C3A8FFA130AF54890692B240250091884 |
| oof_predictions.csv | 637242DC0F45C4DC436643214C21AC451FC560D7AE36B25A6DB04AAFE5D6A2C8 |
| model_ranking.csv | D279AE973D5AB7508E97460BB1B6F94DD95345CE27D4CCA0AD2F587516CB72E4 |
| hyperparameters.csv | 5DA9C936B573126D21417752D0FD140306562ED387B299364A5CD356268BFE00 |
| feature_importance.csv | 17CF4BAFDB1D7B87654D7EFD58EF79584F1324C40ABAC3CB7BDFC617A510B9F6 |
| model_failures.csv | B1CE311D21BE365F1E5A010D9EB16AAEA997E5D843E5B630AEF77C91B52D98EA |
| feature_sets.json | 49E65AFFEBBAB6285170E45048EEE3991BFC70E1A53454D7E6BBC94A092E28CD |
| run_metadata.json | D0CDFABEA5F9243EEADF7EB2278019F9497A75C3E82BA129B4077E1F62566B9F |

Canonical frozen roster 以 (image_key, cell_label, nucleus_label) 轉成 str／int／int 後 stable sort，再輸出 UTF-8、LF、無 index CSV；SHA-256 固定為 A8333F12E1591D9E4C4174F5C6FE13DD31550124B19AEA3522F4D0F812E0E426。

## File Structure

### Create

- immunity/configs/exp3_round2_paper93.yaml：唯一 Round 2 config，pin Round 1 hashes、模型、特徵、seed、QC 與輸出。
- immunity/exp3/round2_evidence.py：唯讀載入 Round 1 artifacts、semantic validation、roster/hash、mask provenance 與 frozen split restoration。
- immunity/exp3/round2_features.py：從 verified masks 與 frozen roster 重算 33 cell features、計算 60 extras、valid-count QC 與 93-feature FOV bundle。
- immunity/exp3/round2_benchmark.py：只執行兩個 93-feature models，合併 frozen baseline／Dummy evidence，計算四 configuration eligibility、ranking 與 recommendation。
- immunity/exp3/round2_reporting.py：Round 2 專用 staged generation、exact artifact writer、bundle validator 與繁體中文實驗記錄。
- immunity/exp3/run_round2_paper93.py：config／CLI、formal／smoke orchestration、run.log 與 failure quarantine。
- tests/test_immunity_exp3_round2_evidence.py：artifact、roster、mask 與 frozen split tests。
- tests/test_immunity_exp3_round2_features.py：93-feature extraction、finite counts、leakage 與 data-lock tests。
- tests/test_immunity_exp3_round2_benchmark.py：雙模型 adapter、baseline reuse、eligibility 與 tie-rule tests。
- tests/test_immunity_exp3_round2_reporting.py：atomic publication、bundle completeness 與 record tests。
- tests/test_immunity_exp3_round2_cli.py：config/output boundary、smoke/full orchestration 與 failure tests。

### Modify

- immunity/exp3/README.md:20：新增 Round 2 command、輸出位置與判讀限制；不改 Round 1 command。

### Explicitly Do Not Modify

- main.py
- immunity/exp3/run_benchmark.py
- immunity/exp3/reporting.py
- immunity/configs/exp3.yaml
- tests/test_immunity_exp3_cli.py
- legacy CSV／XLSX schema 或 GUI modules

---

### Task 1: Round 2 Config 與輸出邊界

**Files:**
- Create: immunity/configs/exp3_round2_paper93.yaml
- Create: immunity/exp3/run_round2_paper93.py
- Create: tests/test_immunity_exp3_round2_cli.py

**Interfaces:**
- Produces: load_round2_config(path: str | Path) -> dict[str, Any]
- Produces: resolve_round2_output_dir(path: str | Path, *, smoke: bool) -> Path
- Produces: build_parser() -> argparse.ArgumentParser
- Consumes later: 其餘 tasks 從 config 的 round1、features、benchmark、smoke 與 output sections 取得固定 contract。

- [ ] **Step 1: Write the failing config and output-boundary tests**

在 tests/test_immunity_exp3_round2_cli.py 建立下列核心測試；link escape helper 沿用既有 Exp3 測試的 Windows junction fallback。

~~~python
from pathlib import Path

import pytest

import immunity.exp3.run_round2_paper93 as run_module
from immunity.exp3.run_round2_paper93 import (
    load_round2_config,
    resolve_round2_output_dir,
)


def test_round2_config_locks_exact_models_features_seed_and_frozen_hashes() -> None:
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


def test_round2_output_resolver_accepts_only_formal_root_and_smoke_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    with pytest.raises(ValueError, match="round2_paper93"):
        resolve_round2_output_dir(relative, smoke=False)
~~~

另加入 test_round2_output_resolver_rejects_link_escape 與
test_round2_output_resolver_rejects_existing_smoke_child_link_escape：只要
ROUND2_OUTPUT_ROOT 任一 component 或既有 smoke child 是 junction／symlink，就要求
ValueError，且 outside directory 不得新增檔案。

- [ ] **Step 2: Run the tests and verify RED**

Run:

~~~powershell
conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_round2_cli.py -q
~~~

Expected: collection FAIL，因 immunity.exp3.run_round2_paper93 尚不存在。

- [ ] **Step 3: Add the exact YAML contract**

建立 immunity/configs/exp3_round2_paper93.yaml：

~~~yaml
experiment_name: exp3_round2_phase_only_paper93

round1:
  dir: immunity/outputs/exp3
  expected:
    raw_pc: 720
    raw_ido: 719
    paired: 719
    exclusions: 26
    analyzed_images: 693
    valid_cells: 23976
    outer_folds: 23
    seed: 20260804
    manifest_hash: b4508aaf830f4c456759d1f02458606aa28ed385856575c6fb99a5ac2e163771
    config_hash: b7590ed3438cdb5062b2588e525e603dd0b7610d397322c761facdaea5b54bc0
  roster_sha256: a8333f12e1591d9e4c4174f5c6fe13dd31550124b19aea3522f4d0f812e0e426
  artifact_sha256:
    pairing_qc.csv: 0fa953eceb35678aab0209214eb1ee267e844a0b81e649b0c0acca487246e0eb
    data_manifest.csv: b4508aaf830f4c456759d1f02458606aa28ed385856575c6fb99a5ac2e163771
    segmentation_qc.csv: 2a1e8969fc8d0a81d3dd737ef99df57f58ce3e1faf0206e833463f2dc9d28c1e
    outer_splits.csv: c7e9e26a9ad0c6f73d8f14c5fa4c73747f6e58865aac963bf8e16e527fa05a3e
    feature_cache/image_level_basic.csv: dad479258d08846d05a43f551b0001a52d96fa5e2e8ea6a98645723931aa0d14
    feature_cache/cell_level_basic.csv: 7d999a728d848ccb61a3910cb5230c9efde6d7e1fa14f22de792990ee4ad7850
    fold_metrics.csv: 8ce0df3866822a54c98cb1cd95dbec3c3a8ffa130af54890692b240250091884
    oof_predictions.csv: 637242dc0f45c4dc436643214c21ac451fc560d7ae36b25a6db04aafe5d6a2c8
    model_ranking.csv: d279ae973d5ab7508e97460bb1b6f94dd95345ce27d4cca0ad2f587516cb72e4
    hyperparameters.csv: 5da9c936b573126d21417752d0fd140306562ed387b299364a5cd356268bfe00
    feature_importance.csv: 17cf4bafdb1d7b87654d7efd58ef79584f1324c40abac3cb7bdfc617a510b9f6
    model_failures.csv: b1ce311d21be365f1e5a010d9eb16aaea997e5d843e5b630aef77c91b52d98ea
    feature_sets.json: 49e65affebbab6285170e45048eee3991bfc70e1a53454d7e6bbc94a092e28cd
    run_metadata.json: d0cdfabea5f9243eeadf7eb2278019f9497a75c3e82ba129b4077e1f62566b9f

models: [extra_trees, random_forest]

features:
  feature_set: paper_style_median
  predictor_count: 93
  extra_count: 60
  min_finite_cells_per_feature: 3
  numeric_atol: 1.0e-12

benchmark:
  seed: 20260804
  inner_splits: 5
  max_hyperparameter_candidates: 24
  n_jobs: 1
  permutation_repeats: 20
  tree_estimators: 400
  simplicity_order:
    [paper_linear_3f, ridge, elasticnet, rbf_svr, hist_gradient_boosting, random_forest, extra_trees]

smoke:
  fovs_per_condition: 1
  max_hyperparameter_candidates: 1
  permutation_repeats: 2
  tree_estimators: 10

output:
  dir: immunity/outputs/exp3/round2_paper93
~~~

- [ ] **Step 4: Implement only config loading, parser, and exact output resolution**

在 immunity/exp3/run_round2_paper93.py 建立 PROJECT_ROOT、ROUND2_OUTPUT_ROOT 與三個 public APIs。load_round2_config 必須驗證 required sections、exact models、exact feature contract、seed、formal output root 與所有 SHA-256 是 64 位 hex；此 task 不加入 pipeline orchestration。

~~~python
ROUND2_MODELS = ("extra_trees", "random_forest")
ROUND2_FEATURE_SET = "paper_style_median"


def resolve_round2_output_dir(path: str | Path, *, smoke: bool) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve(strict=False)
    expected = Path(ROUND2_OUTPUT_ROOT).resolve(strict=False)
    if resolved != expected:
        raise ValueError("Round 2 output 必須是 immunity/outputs/exp3/round2_paper93")
    target = candidate / "smoke" if smoke else candidate
    if any(_is_reparse_point(part) for part in _existing_path_components(target)):
        raise ValueError("Round 2 output path 不可經過 symlink 或 junction")
    expected_target = expected / "smoke" if smoke else expected
    if target.resolve(strict=False) != expected_target:
        raise ValueError("Round 2 output target 不可離開隔離目錄")
    return expected_target
~~~

_existing_path_components 必須一路檢查到最後實際 target，包括既有的 smoke child；
_is_reparse_point 必須同時拒絕 symlink 與 Windows junction／reparse point。在
Python 3.10 以 os.lstat(component).st_file_attributes 搭配
getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0) 判定，不可只依賴 Path.is_symlink()。

build_parser 固定提供 --config 與 --smoke-fovs-per-condition；後者必須是正整數。

- [ ] **Step 5: Run focused tests**

Run:

~~~powershell
conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_round2_cli.py -q
~~~

Expected: PASS。

- [ ] **Step 6: Commit only Task 1 files**

~~~powershell
git add immunity/configs/exp3_round2_paper93.yaml immunity/exp3/run_round2_paper93.py tests/test_immunity_exp3_round2_cli.py
git diff --cached --name-only
git commit -m "feat(immunity): 新增 Exp3 Round 2 獨立設定與輸出邊界"
~~~

Cached names 必須只有上述三檔。

---

### Task 2: Frozen Round 1 Evidence、Mask 與 Outer Splits

**Files:**
- Create: immunity/exp3/round2_evidence.py
- Create: tests/test_immunity_exp3_round2_evidence.py

**Interfaces:**
- Produces: Round1Evidence dataclass
- Produces: load_round1_evidence(config: Mapping[str, Any]) -> Round1Evidence
- Produces: require_formal_round1_evidence(evidence: Round1Evidence) -> None
- Produces: validate_frozen_masks(evidence: Round1Evidence, image_keys: Sequence[str] | None = None) -> pd.DataFrame
- Produces: restore_frozen_outer_splits(images: pd.DataFrame, split_manifest: pd.DataFrame) -> tuple[OuterSplit, ...]
- Produces: make_smoke_outer_splits(images: pd.DataFrame, config: Mapping[str, Any]) -> tuple[OuterSplit, ...]
- Produces: expected_split_ids(splits: Sequence[OuterSplit]) -> dict[str, list[str]]
- Consumes: Task 1 round1 config 與既有 benchmark.OuterSplit／outer_split_manifest。

- [ ] **Step 1: Write synthetic artifact, hash, semantic identity, mask, and split tests**

在 tests/test_immunity_exp3_round2_evidence.py 建立 _write_round1_fixture(tmp_path)；fixture 必須寫出 Task 2 loader 所需的 14 個 artifacts，並回傳以實際 bytes 計算的 artifact_sha256。最小 fixture 使用 2 個 FOV、每張 3 cells、兩個 synthetic splits，只測結構化 loader；另以 _formal_evidence_fixture() 建立符合固定 693／23,976／23-fold contract 的 in-memory evidence，專門測 require_formal_round1_evidence。Production constants 不得由一般 config 任意放寬。

加入以下 tests：

~~~python
def test_round1_evidence_loads_pinned_artifacts_and_semantic_identity(
    tmp_path: Path,
) -> None:
    config = _write_round1_fixture(tmp_path)

    evidence = load_round1_evidence(config)

    assert evidence.basic_images["image_key"].is_unique
    assert len(evidence.basic_cells) == 6
    assert set(evidence.selected_metrics["model"]) == {
        "extra_trees",
        "random_forest",
        "dummy_median",
    }
    assert set(evidence.selected_metrics["source_round"]) == {"round1"}


def test_round1_evidence_rejects_bytes_drift_before_semantic_use(
    tmp_path: Path,
) -> None:
    config = _write_round1_fixture(tmp_path)
    target = Path(config["round1"]["dir"]) / "outer_splits.csv"
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="outer_splits.csv.*SHA-256"):
        load_round1_evidence(config)


@pytest.mark.parametrize("drift", ["image_key", "target", "basic_value", "roster"])
def test_round1_evidence_rejects_semantic_drift_after_hash_refresh(
    tmp_path: Path,
    drift: str,
) -> None:
    config = _write_round1_fixture(tmp_path)
    _mutate_fixture_and_refresh_hash(config, drift)

    with pytest.raises(ValueError, match="image_key|IDO_score|basic|roster"):
        load_round1_evidence(config)
~~~

另加入：

- test_round1_evidence_requires_exact_et_rf_dummy_roles_feature_sets_and_primary_round
- test_formal_round1_gate_accepts_only_693_23976_exact_hashes_roster_and_23_folds
- parametrized test_formal_round1_gate_rejects_count_roster_hash_fold_or_oof_drift
- test_frozen_mask_validation_accepts_exact_pc_and_embedded_provenance
- parametrized test_frozen_mask_validation_fails_closed_for_missing_changed_or_corrupt_cache
- test_frozen_mask_failure_does_not_change_cache_bytes_or_create_segmenter
- test_restore_frozen_splits_rejects_missing_duplicate_unknown_or_changed_membership
- test_restore_frozen_splits_round_trips_outer_split_manifest_exactly
- test_smoke_splits_have_four_validation_families_without_frozen_membership_gate

Mask fixture 的 NPZ 必須包含 cell_mask、nucleus_mask、canonical provenance_json 與相符 provenance_hash；PC bytes、embedded pc_sha256、segmentation_qc.pc_sha256 三者必須一致。

- [ ] **Step 2: Run tests and verify RED**

Run:

~~~powershell
conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_round2_evidence.py -q
~~~

Expected: FAIL with ModuleNotFoundError for immunity.exp3.round2_evidence。

- [ ] **Step 3: Implement the frozen evidence dataclass and loader**

Round1Evidence 固定保存：

~~~python
@dataclass(frozen=True)
class Round1Evidence:
    root: Path
    manifest: pd.DataFrame
    segmentation_qc: pd.DataFrame
    basic_cells: pd.DataFrame
    basic_images: pd.DataFrame
    split_manifest: pd.DataFrame
    selected_metrics: pd.DataFrame
    selected_predictions: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    selected_importance: pd.DataFrame
    selected_failures: pd.DataFrame
    model_ranking: pd.DataFrame
    metadata: Mapping[str, Any]
    artifact_hashes: Mapping[str, str]
~~~

load_round1_evidence 必須依下列順序 fail-closed：

1. 解析 round1.dir，確認它是 Exp3 parent root 而非 Round 2 output。
2. 確認 14 個 required files 都是 regular files，且 resolved path 沒有離開 round1 root。
3. 逐檔串流計算 SHA-256，先與傳入 config pins 比對，再讀 CSV／JSON。
4. 驗證 image/cell tables 非空、image key 與 roster 無重複、每 FOV 至少 3 cells、33 basic columns exact、target finite；這一層不硬編 formal row counts，讓 tiny fixture 可測 loader。
5. 重算 canonical roster hash並與傳入 config pin 比對；驗證 image-level basic/IDO medians，以 rtol=0、atol=1e-12、equal_nan=False 比對。
6. 驗證 run_metadata 與已載入 artifacts 在 analyzed_images、seed、manifest_hash、config_hash 上內部一致。
7. 只保留 primary_round_1 的 extra_trees／random_forest／dummy_median rows，並新增 source_round=round1 與 configuration_id；不得 fit estimator。
8. 驗證 ET/RF 是 candidate/basic_median、Dummy 是 diagnostic/none，metrics／OOF identity 無重複且 observed targets 對齊；此層不硬編 23 folds／2,772 OOF。

require_formal_round1_evidence 再套用不可由 config 放寬的 production contract：14 個固定
source SHA-256、693 unique images、23,976 cells、canonical roster SHA-256、seed
20260804、23 個 outer folds（3／3／9／8），以及 ET／RF／Dummy 各 23 個 successful
folds與 2,772 OOF rows。正式與 smoke runner 都必須先對完整 Round 1 evidence 呼叫此 gate，
之後才能擷取 formal 全集或 smoke subset。

Canonical roster hash 使用：

~~~python
def canonical_roster_sha256(cells: pd.DataFrame) -> str:
    roster = cells.loc[:, ["image_key", "cell_label", "nucleus_label"]].copy()
    roster["image_key"] = roster["image_key"].astype(str)
    roster["cell_label"] = roster["cell_label"].astype(int)
    roster["nucleus_label"] = roster["nucleus_label"].astype(int)
    roster = roster.sort_values(
        ["image_key", "cell_label", "nucleus_label"],
        kind="stable",
    )
    payload = roster.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
~~~

- [ ] **Step 4: Implement read-only mask verification**

validate_frozen_masks 必須由 round1 root、group_id 與 image_key 推導
feature_cache/masks/{group_id}/{image_key}.npz；segmentation_qc.mask_path 只作一致性證據，不作未驗證輸入路徑。

每列輸出：

~~~text
image_key,pc_path,mask_path,expected_pc_sha256,actual_pc_sha256,
expected_provenance_hash,actual_provenance_hash,status,reason
~~~

使用 numpy.load(..., allow_pickle=False) 驗證 arrays、canonical provenance JSON 與 embedded hash；再驗證 embedded pc_sha256。函式只回傳 QC，不直接 raise，讓 runner 能先保存 evidence。任何 failure 的 reason 必須含 image key 與 pc／mask 完整路徑。

- [ ] **Step 5: Implement exact frozen split restoration**

restore_frozen_outer_splits 必須：

- 依 source artifact 首次出現順序建立 (validation, fold)。
- 只接受完整 693-image formal snapshot。
- 每 split 僅允許 train／test roles、完整覆蓋 693 個 unique keys，且 train／test 不重疊。
- 四 family fold count 必須是 3／3／9／8。
- 每個 family 的每個輸入 image key 必須恰好一次 test。
- split rows 的 B、passage、group、condition metadata 必須與 images 一致。
- 建立 OuterSplit positional arrays 後，使用 outer_split_manifest(images, splits) canonical compare source artifact 的 exact membership。

make_smoke_outer_splits 只用 smoke subset 與既有 make_outer_splits 建立 deterministic diagnostic
splits，驗證四個 validation families、split_id 唯一、train／test 非空與無 leakage；不得執行
693-row 或 frozen membership canonical comparison。Smoke metadata 必須明寫 diagnostic_splits、
non_formal 與 no_scientific_conclusion。

- [ ] **Step 6: Run focused tests**

~~~powershell
conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_round2_evidence.py -q
~~~

Expected: PASS。

- [ ] **Step 7: Commit only evidence files**

~~~powershell
git add immunity/exp3/round2_evidence.py tests/test_immunity_exp3_round2_evidence.py
git diff --cached --name-only
git commit -m "feat(immunity): 鎖定 Round 1 資料與切分證據"
~~~

---

### Task 3: Frozen-roster Paper-style 93 Feature Extraction

**Files:**
- Create: immunity/exp3/round2_features.py
- Create: tests/test_immunity_exp3_round2_features.py

**Interfaces:**
- Produces: Paper93Bundle dataclass
- Produces: extract_locked_paper93(evidence: Round1Evidence, mask_qc: pd.DataFrame, image_keys: Sequence[str] | None = None, *, min_finite_cells: int = 3, atol: float = 1e-12) -> Paper93Bundle
- Produces: require_paper93_preflight(bundle: Paper93Bundle, *, formal: bool) -> None
- Consumes: Task 2 evidence／mask QC，以及既有 paper_style_extras、BASIC_GEOMETRY、PAPER_STYLE_EXTRA_FEATURES、PAPER_STYLE_FOV_FEATURES、validate_phase_predictors。

- [ ] **Step 1: Write phase-only, roster, finite-count, constant, and leakage tests**

建立 synthetic phase image、verified masks 與 3-cell frozen roster。IDO path 指向不存在檔案，證明 extractor 不讀 IDO。

~~~python
def test_paper93_bundle_preserves_roster_basic_values_and_target(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
) -> None:
    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)

    assert len(bundle.predictor_columns) == 93
    assert tuple(bundle.predictor_columns) == PAPER_STYLE_FOV_FEATURES
    assert set(bundle.paper_cells[ROSTER_COLUMNS].itertuples(index=False, name=None)) == (
        set(
            frozen_evidence.basic_cells[ROSTER_COLUMNS].itertuples(
                index=False,
                name=None,
            )
        )
    )
    np.testing.assert_allclose(
        bundle.images[PRIMARY_FOV_FEATURES],
        frozen_evidence.basic_images[PRIMARY_FOV_FEATURES],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        bundle.images["IDO_score"],
        frozen_evidence.basic_images["IDO_score"],
        rtol=0,
        atol=1e-12,
    )


def test_paper93_valid_counts_require_three_finite_cells_per_extra(
    frozen_evidence: Round1Evidence,
    verified_mask_qc: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {name: 1.0 for name in PAPER_STYLE_EXTRA_FEATURES}
    calls = 0

    def fake_extras(*args: object) -> dict[str, float]:
        nonlocal calls
        calls += 1
        row = dict(values)
        if calls >= 2:
            row["cell__zernike_00"] = np.nan
        return row

    monkeypatch.setattr(round2_features, "paper_style_extras", fake_extras)
    bundle = extract_locked_paper93(frozen_evidence, verified_mask_qc)

    failed = bundle.valid_counts.query(
        "feature == 'cell__zernike_00' and status == 'failed'"
    )
    assert failed["finite_cell_count"].tolist() == [1]
    with pytest.raises(ValueError, match="cell__zernike_00.*finite_cell_count=1"):
        require_paper93_preflight(bundle, formal=True)
~~~

另加入：

- test_paper93_recomputes_33_cell_features_and_rejects_mask_array_drift
- test_paper93_rejects_missing_extra_or_duplicate_roster_pair
- test_paper93_extraction_failure_contains_image_key_pc_path_and_mask_path
- test_paper93_constant_feature_is_reported_but_not_failed
- parametrized test_paper93_rejects_forbidden_predictor_at_exact_count_93，替換合法欄為 IDO、IFN、TNF、dose、condition、path、filename、donor、b_id、delta。
- test_paper93_rejects_duplicate_or_unregistered_predictor_at_count_93
- test_paper93_does_not_read_missing_ido_path_or_call_segmentation

- [ ] **Step 2: Run tests and verify RED**

~~~powershell
conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_round2_features.py -q
~~~

Expected: FAIL with ModuleNotFoundError for immunity.exp3.round2_features。

- [ ] **Step 3: Implement the immutable bundle contract**

~~~python
@dataclass(frozen=True)
class Paper93Bundle:
    images: pd.DataFrame
    paper_cells: pd.DataFrame
    valid_counts: pd.DataFrame
    extraction_qc: pd.DataFrame
    feature_qc: pd.DataFrame
    predictor_columns: tuple[str, ...]
~~~

extract_locked_paper93 必須：

1. 只處理 mask_qc.status=passed 且屬指定 image_keys 的 rows。
2. 讀 phase 與 verified masks，不接受 IDO array，也不建立 Segmenter。
3. 以 frozen (image_key, cell_label, nucleus_label) roster 為唯一 rows；duplicate／missing label、nucleus outside 或 empty cytoplasm 都記成 extraction failure，不靜默 drop。
4. 對每顆 roster cell 重算 33 basic phase features，與 frozen cell_level_basic.csv 逐欄比對。
5. 呼叫 paper_style_extras 計算固定 60 extras。
6. 對每個 FOV × extra 計算 roster_cell_count、finite_cell_count、required_minimum、status；只用 finite values 算 median。
7. 以 frozen basic image table 提供 metadata、33 predictors 與 IDO_score，再依 canonical order附加 60 medians。
8. 建立 93-row feature_qc，欄位至少包含 feature、feature_group、finite_fov_count、unique_finite_count、constant、forbidden_token、registered、status。

Valid-count schema 固定為：

~~~python
VALID_COUNT_COLUMNS = [
    "image_key",
    "feature",
    "roster_cell_count",
    "finite_cell_count",
    "required_minimum",
    "status",
]
~~~

不要用 placeholder 常數填補 exception 或 non-finite；exception 寫入 extraction_qc 並讓 preflight fail。

- [ ] **Step 4: Implement the preflight gate**

require_paper93_preflight 必須檢查 exact 93、33+60 identity、順序唯一、registered whitelist、全部 FOV predictors finite、valid counts、roster identity、basic/target tolerance、無 extraction failure。formal=True 時另外要求 693 FOV；smoke 只要求 subset 非空與 schema 一致。

- [ ] **Step 5: Run feature and existing registry regression tests**

~~~powershell
conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_round2_features.py tests/test_immunity_exp3_feature_sets.py::test_paper_style_registry_is_exact_label_free_approximation tests/test_immunity_exp3_feature_sets.py::test_paper_style_extras_use_phase_regions_and_normalized_centroid tests/test_immunity_exp3_feature_sets.py::test_primary_path_does_not_compute_paper_features -q
~~~

Expected: PASS。

- [ ] **Step 6: Commit only feature files**

~~~powershell
git add immunity/exp3/round2_features.py tests/test_immunity_exp3_round2_features.py
git diff --cached --name-only
git commit -m "feat(immunity): 建立 93 特徵擷取與品質閘門"
~~~

---

### Task 4: Two-model 93-feature Frozen-split Benchmark

**Files:**
- Create: immunity/exp3/round2_benchmark.py
- Create: tests/test_immunity_exp3_round2_benchmark.py

**Interfaces:**
- Produces: run_paper93_benchmark(images: pd.DataFrame, splits: Sequence[OuterSplit], config: Mapping[str, Any]) -> BenchmarkResult（回傳已 normalized 的 Round 2 identity）
- Produces: normalize_round2_result(result: BenchmarkResult) -> BenchmarkResult
- Consumes: Task 3 Paper93Bundle.images、Task 2 frozen splits、既有 run_phase_feature_set_benchmark。

- [ ] **Step 1: Write adapter identity, no-refit, split, count, and preprocessing tests**

~~~python
def test_round2_benchmark_calls_secondary_adapter_once_with_only_two_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    raw_expected = _raw_benchmark_result()

    def fake_adapter(
        images: pd.DataFrame,
        feature_sets: Mapping[str, Sequence[str]],
        splits: Sequence[OuterSplit],
        config: Mapping[str, object],
        model_names: Sequence[str],
        feature_set_name: str,
    ) -> BenchmarkResult:
        calls.append(
            {
                "models": tuple(model_names),
                "feature_set": feature_set_name,
                "predictors": tuple(feature_sets[feature_set_name]),
            }
        )
        return raw_expected

    monkeypatch.setattr(round2_benchmark, "run_phase_feature_set_benchmark", fake_adapter)
    result = run_paper93_benchmark(_paper93_images(), _frozen_splits(), _config())

    assert result is not raw_expected
    assert set(result.fold_metrics["source_round"]) == {"round2_paper93"}
    assert set(result.fold_metrics["feature_set"]) == {"paper_style_median"}
    pd.testing.assert_series_equal(
        result.fold_metrics["mae"],
        raw_expected.fold_metrics["mae"],
    )
    assert calls == [
        {
            "models": ("extra_trees", "random_forest"),
            "feature_set": "paper_style_median",
            "predictors": PAPER_STYLE_FOV_FEATURES,
        }
    ]


def test_round2_benchmark_never_refits_33_feature_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = Mock(side_effect=AssertionError("Round 1 baseline must stay frozen"))
    monkeypatch.setattr(round2_benchmark, "run_nested_benchmark", forbidden, raising=False)

    run_paper93_benchmark(_paper93_images(), _frozen_splits(), _config())

    forbidden.assert_not_called()
~~~

另加入：

- test_round2_benchmark_uses_exact_frozen_split_ids_and_membership
- test_round2_full_shape_has_46_metrics_5544_oof_and_46_hyperparameter_rows
- test_round2_fold_failure_is_retained_and_never_substitutes_a_split
- test_round2_result_labels_every_row_source_round2_and_paper_style_median
- test_round2_uses_same_seed_tree_search_space_and_fold_local_pipeline

- [ ] **Step 2: Run tests and verify RED**

~~~powershell
conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_round2_benchmark.py -q
~~~

Expected: FAIL with ModuleNotFoundError for immunity.exp3.round2_benchmark。

- [ ] **Step 3: Implement the thin public adapter**

~~~python
ROUND2_MODELS = ("extra_trees", "random_forest")


def run_paper93_benchmark(
    images: pd.DataFrame,
    splits: Sequence[OuterSplit],
    config: Mapping[str, Any],
) -> BenchmarkResult:
    raw = run_phase_feature_set_benchmark(
        images,
        {"paper_style_median": list(PAPER_STYLE_FOV_FEATURES)},
        splits,
        config,
        model_names=list(ROUND2_MODELS),
        feature_set_name="paper_style_median",
    )
    return normalize_round2_result(raw)
~~~

normalize_round2_result 必須對 predictions、fold_metrics、hyperparameters、feature_importance、failures 加上 source_round=round2_paper93、feature_set=paper_style_median 與 configuration_id={model}__paper_style_median；不得改 model name 或 numerical evidence。run_paper93_benchmark 是唯一 public training entry，必須在回傳前呼叫 normalization，runner 不得漏接另一個隱含 stage。

- [ ] **Step 4: Run new and existing benchmark contract tests**

~~~powershell
conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_round2_benchmark.py -q
conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_benchmark.py::test_registry_keeps_preprocessing_and_target_scaling_inside_estimators tests/test_immunity_exp3_benchmark.py::test_phase_feature_set_adapter_records_actual_secondary_identity tests/test_immunity_exp3_benchmark.py::test_phase_feature_set_adapter_rejects_paper_and_diagnostics tests/test_immunity_exp3_benchmark.py::test_phase_feature_set_adapter_requires_exact_registered_whitelist -q
~~~

Expected: PASS。

- [ ] **Step 5: Commit the two-model benchmark**

~~~powershell
git add immunity/exp3/round2_benchmark.py tests/test_immunity_exp3_round2_benchmark.py
git diff --cached --name-only
git commit -m "feat(immunity): 執行兩個模型的 frozen-split benchmark"
~~~

---

### Task 5: Four-configuration Eligibility、Ranking 與 Recommendation

**Files:**
- Modify: immunity/exp3/round2_benchmark.py
- Modify: tests/test_immunity_exp3_round2_benchmark.py

**Interfaces:**
- Produces: Round2Comparison dataclass
- Produces: build_round2_comparison(evidence: Round1Evidence, paper93_result: BenchmarkResult, expected_splits: Mapping[str, Sequence[str]]) -> Round2Comparison
- Produces: rank_round2_configurations(comparison: Round2Comparison, predictor_sets: Mapping[str, Sequence[str]]) -> pd.DataFrame
- Produces: select_round2_recommendation(ranking: pd.DataFrame) -> str | None
- Consumes: Task 2 frozen ET/RF/Dummy rows與 Task 4 normalized results。

- [ ] **Step 1: Write exact four-config, Dummy, eligibility, OOF rank, and strict tie tests**

~~~python
def test_round2_comparison_contains_exactly_four_candidate_configurations() -> None:
    comparison = build_round2_comparison(
        _round1_evidence_fixture(),
        _round2_result_fixture(),
        _expected_split_ids(),
    )

    assert set(comparison.fold_metrics["configuration_id"]) == {
        "extra_trees__basic_median",
        "extra_trees__paper_style_median",
        "random_forest__basic_median",
        "random_forest__paper_style_median",
    }
    assert set(comparison.dummy_metrics["model"]) == {"dummy_median"}
    assert set(comparison.fold_metrics["source_round"]) == {
        "round1",
        "round2_paper93",
    }


def test_round2_tie_band_is_strict_and_prefers_33_for_same_algorithm() -> None:
    comparison = _eligible_comparison_with_family_ranks(
        {
            "extra_trees__basic_median": (1, 2, 1, 2),
            "extra_trees__paper_style_median": (2, 1, 2, 1),
            "random_forest__basic_median": (3, 3, 4, 4),
            "random_forest__paper_style_median": (4, 4, 3, 3),
        }
    )
    ranking = rank_round2_configurations(comparison, _predictor_sets())

    selected = select_round2_recommendation(ranking)

    assert selected == "extra_trees__basic_median"
    assert ranking.set_index("configuration_id").loc[
        "extra_trees__paper_style_median", "eliminated_by_simplicity"
    ]
    assert not ranking.set_index("configuration_id").loc[
        "random_forest__basic_median", "in_tie_band"
    ]
~~~

另加入：

- test_round2_eligibility_uses_frozen_dummy_fold_median_mae
- test_round2_positive_r2_requires_two_validation_families
- test_round2_complete_gate_requires_23_folds_and_exact_oof_coverage
- test_round2_prediction_sd_gate_requires_five_percent_in_every_family
- test_round2_phase_gate_checks_exact_33_or_93_whitelist
- test_round2_ranking_uses_pooled_oof_mae_not_fold_median_mae
- test_round2_family_ranks_use_method_min_and_equal_weights
- test_round2_strict_tie_band_excludes_average_rank_difference_exactly_025
- test_round2_tie_break_order_is_worst_lobo_spearman_feature_count_simplicity_name
- test_round2_no_eligible_configuration_returns_none
- test_round2_recommendation_never_changes_round1_final_model_json

- [ ] **Step 2: Run comparison tests and verify RED**

~~~powershell
conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_round2_benchmark.py -k "comparison or eligibility or ranking or tie or recommendation" -q
~~~

Expected: FAIL because comparison APIs do not exist。

- [ ] **Step 3: Implement the comparison dataclass and evidence normalization**

~~~python
@dataclass(frozen=True)
class Round2Comparison:
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    dummy_metrics: pd.DataFrame
    dummy_predictions: pd.DataFrame
    hyperparameters: pd.DataFrame
    feature_importance: pd.DataFrame
    failures: pd.DataFrame
~~~

Baseline ET/RF rows只可從 evidence 讀取；Dummy 只進 dummy tables。Failure-free candidate outputs 必須是：

- fold_metrics：92 rows（4 configurations × 23 folds）。
- predictions：11,088 rows（4 configurations × 693 images × 4 families）。
- hyperparameters：92 rows。
- feature_importance：最多 5,796 rows（46 × 33 + 46 × 93）；importance 是 diagnostic，缺失必須在 completeness/QC 與 record 明示，但不單獨改變 eligibility。
- dummy_metrics：23 rows。
- dummy_predictions：2,772 rows。

Model fold fit failure 仍保留一列 status=failed 的 fold_metrics 與對應
model_failures row；該 configuration 因 complete_outer_folds_gate=False 而 ineligible，但其他
configuration 仍可形成可發布 evidence。Failed fold 不得有 OOF 或 hyperparameter rows；validator
必須以 failed identity 精確解釋缺少的 rows，不可用全域固定 OOF count 把整個 generation 誤判為
損壞。Feature-importance diagnostic 缺失則另外列入 QC／record，不視為 model fold failure。

- [ ] **Step 4: Implement gates and the unique decision algorithm**

Eligibility 與 ranking 使用不同且明確命名的 metrics：

- Gate 用 validation_median_fold_mae 與 median_fold_r2，完全沿用 Round 1 規則。
- Rank 用 validation_pooled_oof_mae，由完整 OOF rows 重新計算。

Ranking 固定 4 rows，至少包含：

~~~text
configuration_id,model,feature_set,source_round,feature_count,
leave_one_b_out_oof_mae,leave_one_passage_out_oof_mae,
leave_one_group_out_oof_mae,leave_one_condition_out_oof_mae,
各 family rank,validations_beating_dummy,validations_with_positive_r2,
mae_beats_dummy_gate,positive_r2_gate,complete_outer_folds_gate,
phase_only_feature_gate,prediction_sd_gate,eligible,reasons_json,
average_rank,worst_validation_rank,overall_oof_spearman,
in_tie_band,eliminated_by_simplicity,recommended
~~~

Decision 必須依書面規格：

1. 只從 eligible rows 找 R*。
2. average_rank - R* < 0.25 才進 band；等於 0.25 排除。
3. 同 algorithm 的 33／93 同在 band 時淘汰 93。
4. 依 worst rank、LOBO OOF MAE、overall OOF Spearman descending、feature count、既有 simplicity rank、model name 排序。
5. 至多一列 recommended=True；無 eligible 則全 False。

因四個 family rank 都是整數且等權平均，average_rank 只會以 0.25 為單位變化；因此嚴格
< 0.25 band 在目前 contract 下只包含 average_rank 完全相同者。測試不得捏造 1.10／1.15
這類不可能由四個 method="min" ranks 產生的平均值。Strict-boundary fixture 使用合法的
四-family permutations 產生 2.00 與 2.25，確認差值恰為 0.25 時排除。

- [ ] **Step 5: Run all Round 2 benchmark tests**

~~~powershell
conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_round2_benchmark.py -q
~~~

Expected: PASS。

- [ ] **Step 6: Commit comparison logic separately**

~~~powershell
git add immunity/exp3/round2_benchmark.py tests/test_immunity_exp3_round2_benchmark.py
git diff --cached --name-only
git commit -m "feat(immunity): 新增 33 與 93 特徵比較及 eligibility"
~~~

---

### Task 6: Atomic Reporting、Bundle Validation 與繁體中文 Record

**Files:**
- Create: immunity/exp3/round2_reporting.py
- Create: tests/test_immunity_exp3_round2_reporting.py

**Interfaces:**
- Produces: Round2Generation dataclass
- Produces: begin_round2_generation(output_dir: Path) -> Round2Generation
- Produces: publish_round2_generation(generation: Round2Generation) -> None
- Produces: quarantine_round2_generation(generation: Round2Generation) -> Path
- Produces: write_round2_bundle(staging_dir: Path, tables: Mapping[str, pd.DataFrame], json_payloads: Mapping[str, Mapping[str, Any]], record_context: Mapping[str, Any]) -> Path
- Produces: validate_round2_bundle(directory: Path, *, smoke: bool) -> None
- Produces: build_round2_experiment_record(context: Mapping[str, Any]) -> str

- [ ] **Step 1: Write exact artifact, atomicity, safety, completeness, and record tests**

固定 formal CSV names：

~~~python
ROUND2_TABLE_NAMES = (
    "data_snapshot.csv",
    "outer_splits.csv",
    "mask_provenance_qc.csv",
    "feature_valid_counts.csv",
    "extraction_qc.csv",
    "feature_qc.csv",
    "fold_metrics.csv",
    "oof_predictions.csv",
    "dummy_fold_metrics.csv",
    "dummy_oof_predictions.csv",
    "hyperparameters.csv",
    "feature_importance.csv",
    "model_failures.csv",
    "feature_set_comparison.csv",
    "eligibility.csv",
)
~~~

Caller 提供的固定 JSON names 只有 feature_sets.json、baseline_provenance.json、
run_metadata.json；writer 最後另外產生 artifact_hashes.json。另有 EXPERIMENT_RECORD.md。
run.log 由 run_round2 orchestration 擁有，不是 write_round2_bundle 的 input。

建立：

~~~python
def test_round2_writer_publishes_exact_required_artifacts_atomically(
    tmp_path: Path,
) -> None:
    generation = begin_round2_generation(_allowed_output(tmp_path))
    (generation.staging_dir / "run.log").write_text("closed\n", encoding="utf-8")
    record = write_round2_bundle(
        generation.staging_dir,
        _formal_tables(),
        _formal_json_payloads(),
        _formal_record_context(),
    )

    validate_round2_bundle(generation.staging_dir, smoke=False)
    publish_round2_generation(generation)

    assert record.name == "EXPERIMENT_RECORD.md"
    assert set(path.name for path in generation.output_dir.iterdir() if path.is_file()) == {
        *ROUND2_TABLE_NAMES,
        "feature_sets.json",
        "baseline_provenance.json",
        "run_metadata.json",
        "artifact_hashes.json",
        "EXPERIMENT_RECORD.md",
        "run.log",
    }


def test_round2_record_states_target_scope_and_approximation() -> None:
    text = build_round2_experiment_record(_formal_record_context())

    assert "IDO 螢光亮度 proxy" in text
    assert "不是免疫力" in text
    assert "phase-only paper-style approximation" in text
    assert "不改寫 Round 1 winner" in text
    assert "donor" not in _predictor_section(text)
~~~

另加入：

- test_round2_writer_preflights_all_destinations_before_first_publish
- test_round2_writer_rolls_back_staged_files_when_later_write_fails
- test_round2_publication_archives_prior_round2_generation_only
- test_round2_publication_never_moves_or_overwrites_round1_artifacts
- test_round2_failure_quarantines_staging_and_keeps_qc_log
- test_round2_validator_requires_693_rows_93_predictors_and_92_metric_identities
- test_round2_validator_requires_11088_oof_when_all_model_folds_succeed
- test_round2_validator_reconciles_failed_fold_to_missing_oof_and_marks_configuration_ineligible
- test_round2_validator_requires_23_dummy_metrics_2772_dummy_oof_and_four_rank_rows
- test_round2_validator_rejects_duplicate_oof_missing_fold_or_hash_mismatch
- test_smoke_validator_accepts_subset_shape_but_requires_schema_and_no_recommendation
- test_round2_json_writer_rejects_nan_infinity_and_non_mapping_payload
- test_round2_artifact_hash_manifest_excludes_itself_and_run_log

- [ ] **Step 2: Run tests and verify RED**

~~~powershell
conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_round2_reporting.py -q
~~~

Expected: FAIL with ModuleNotFoundError for immunity.exp3.round2_reporting。

- [ ] **Step 3: Implement staged generation boundaries**

Round2Generation 保存 output_dir 與 unique staging_dir。所有 run artifacts 先寫 staging；成功時才 archive 舊 Round 2 formal generation並發布。失敗時將 staging 搬到
<resolver 回傳的 exact target>/_generations/failed-{UTC}-{uuid}；因此 formal failure 留在
round2_paper93/，smoke failure 留在 round2_paper93/smoke/。既有成功 generation 與 Round 1
parent 完全不動。

所有 resolved paths 必須位於 resolver 回傳的 exact target（formal root 或 smoke child）；建立
staging／archive／failed generation 前後都重驗 boundary，拒絕任一 component 的
symlink／junction escape。搬移或 rollback 只能針對明確列出的 Round 2 filenames 與 staging
directory。

- [ ] **Step 4: Implement strict writers and bundle validator**

write_round2_bundle 要求 table keys 與 caller JSON keys exact match；所有 CSV／JSON／Markdown
先寫 sibling temp，再用 os.replace。artifact_hashes.json 由 writer 單獨擁有，在其他 staged
artifacts 關閉後才產生，列出所有 formal CSV、三個 caller JSON 與 EXPERIMENT_RECORD.md 的
SHA-256 與 byte length；它明確排除自身與仍由 orchestration 管理的 run.log，避免 self-reference 與
未關閉檔案 hash。

Formal validation 固定檢查 93 predictor identity、693 keys、23 folds、92 個 metric identities、
source_round、feature_set、四 ranking rows與最多一個 recommendation。若所有 fold status=ok，
再要求 11,088 OOF 與 92 hyperparameters 的 failure-free counts；5,796 importance 是完整
diagnostic 的上限，不是 publication hard gate。若有 failed
fold，必須與 model_failures 一對一對齊，且缺少的 OOF／hyperparameter rows只能來自該 failed
identity，受影響 configuration 必須 ineligible。Importance 缺失只作 diagnostic completeness
警告並寫入 record。Smoke validation 只要求非空 subset、93 schema、兩個新 models、無
recommendation與 smoke status。兩種 validator 都要求 staging/run.log 是 boundary 內的 regular
file，但不讀取或雜湊仍由 tee context 開啟的內容。

- [ ] **Step 5: Implement answer-first Traditional Chinese record**

Record 固定順序：

1. 結論與 recommendation／no eligible status。
2. 白話摘要：93 是否優於 33。
3. Target 定義與不可解讀成免疫力。
4. Data lock、693 FOV、23,976 cells、26 exclusions。
5. 33 vs 93 四 configuration metrics。
6. Eligibility gates 與 tie decision。
7. Feature QC／constant features／extraction failures。
8. Provenance、hashes、runtime 與 limitations。

Smoke record 的第一段必須寫「僅驗證流程，不是正式實驗結果」且不產生 33 vs 93 結論。

- [ ] **Step 6: Run reporting tests**

~~~powershell
conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_round2_reporting.py -q
~~~

Expected: PASS。

- [ ] **Step 7: Commit reporting separately**

~~~powershell
git add immunity/exp3/round2_reporting.py tests/test_immunity_exp3_round2_reporting.py
git diff --cached --name-only
git commit -m "feat(immunity): 原子發布 Round 2 結果與實驗記錄"
~~~

---

### Task 7: End-to-end CLI、Smoke 與 Failure Evidence

**Files:**
- Modify: immunity/exp3/run_round2_paper93.py
- Modify: tests/test_immunity_exp3_round2_cli.py

**Interfaces:**
- Produces: run_round2(config: Mapping[str, Any], smoke_fovs_per_condition: int | None = None) -> Path
- Produces: main() -> int
- Consumes: Tasks 2–6 public interfaces。

- [ ] **Step 1: Write orchestration order, smoke isolation, failure, and exit-code tests**

~~~python
def test_round2_orchestrator_runs_locked_stages_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _patch_round2_stages(monkeypatch, tmp_path, calls)

    record = run_module.run_round2(_synthetic_config(tmp_path))

    assert calls == [
        "load_round1_evidence",
        "require_formal_round1_evidence",
        "validate_frozen_masks",
        "extract_locked_paper93",
        "require_paper93_preflight",
        "restore_frozen_outer_splits",
        "run_paper93_benchmark",
        "build_round2_comparison",
        "rank_round2_configurations",
        "write_round2_bundle",
        "validate_round2_bundle",
        "publish_round2_generation",
    ]
    assert record.name == "EXPERIMENT_RECORD.md"


def test_round2_preflight_failure_quarantines_paths_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = Mock(side_effect=AssertionError("model fit must not run"))
    monkeypatch.setattr(run_module, "run_paper93_benchmark", train)
    _patch_failed_feature_preflight(monkeypatch, tmp_path, image_key="B8_P7_C08_F03")

    with pytest.raises(ValueError, match="B8_P7_C08_F03"):
        run_module.run_round2(_synthetic_config(tmp_path))

    train.assert_not_called()
    failed = list((_formal_output(tmp_path) / "_generations").glob("failed-*"))
    assert len(failed) == 1
    assert (failed[0] / "run.log").is_file()
    assert (failed[0] / "extraction_qc.csv").is_file()
~~~

另加入：

- test_round2_smoke_uses_only_smoke_child_and_marks_nonformal
- test_round2_smoke_skips_693_and_scientific_comparison_gates
- test_round2_smoke_uses_diagnostic_splits_without_frozen_membership_comparison
- test_round2_formal_never_calls_segmentation_or_round1_runner
- test_round2_cli_returns_zero_only_after_bundle_validation_and_record_publish
- test_round2_cli_success_publishes_closed_run_log
- test_round2_publish_and_quarantine_close_run_log_before_move
- test_round2_cli_returns_nonzero_and_logs_named_paths_on_hash_mask_or_feature_failure
- test_round2_run_does_not_change_any_pinned_round1_artifact_hash
- test_round2_import_does_not_expand_main_or_legacy_schema

- [ ] **Step 2: Run tests and verify RED**

~~~powershell
conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_round2_cli.py -q
~~~

Expected: orchestration tests FAIL because run_round2／main 尚未串接。

- [ ] **Step 3: Implement the exact runner flow**

run_round2 固定順序：

1. resolve exact target，再 begin_round2_generation。
2. 由 run_round2 自己開啟 staging/run.log 的 tee context。
3. load_round1_evidence，隨即對完整 evidence 呼叫 require_formal_round1_evidence；smoke 也不可略過 source pins。
4. 若 smoke，從每個 group_id × condition_index 依 fov stable sort 取 N 張；formal 使用 693。
5. validate_frozen_masks subset；先把 mask_provenance_qc.csv 寫 staging，failure 就 raise。
6. extract_locked_paper93；先寫 valid counts、extraction QC、feature QC，preflight failure 就 raise。
7. Formal 呼叫 restore_frozen_outer_splits 做 exact membership；smoke 呼叫 make_smoke_outer_splits，只建立 diagnostic splits，不套 frozen membership gate。
8. 只呼叫一次 run_paper93_benchmark；回傳值已 normalized。
9. Formal 才 build comparison、rank 與 recommendation；smoke 建立空 comparison／eligibility 並標記 no_scientific_conclusion。
10. 建立 tables、caller JSON、record與 artifact hash manifest，validate staging。
11. 離開 tee context，flush／close run.log 後才 publish generation。
12. 任一 exception 都先關閉 log，再以短暫 append handle 寫入 exception evidence並關閉，最後 quarantine staging、重新 raise。

Smoke benchmark config 必須覆寫成 max_hyperparameter_candidates=1、permutation_repeats=2、tree_estimators=10；metadata 同時保留原始與 effective config。

- [ ] **Step 4: Implement main and run.log**

main 只負責讀 --config、optional --smoke-fovs-per-condition、呼叫 run_round2 與轉換 exit
code；run_round2 擁有 generation 與 tee lifecycle。只有 validated bundle 已在 log handle 關閉後
成功 publish，且 published record 存在，main 才回傳 0。任何 exception 回傳非零；failed
generation 的 run.log 必須含 exception type、message、traceback、image key 與可用 pc／mask
path。成功 CLI test 另確認 published run.log 存在，但 run.log 不列入 artifact_hashes.json。

- [ ] **Step 5: Run CLI and focused Round 2 suite**

~~~powershell
conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_round2_evidence.py tests/test_immunity_exp3_round2_features.py tests/test_immunity_exp3_round2_benchmark.py tests/test_immunity_exp3_round2_reporting.py tests/test_immunity_exp3_round2_cli.py -q
~~~

Expected: PASS。

- [ ] **Step 6: Commit orchestration**

~~~powershell
git add immunity/exp3/run_round2_paper93.py tests/test_immunity_exp3_round2_cli.py
git diff --cached --name-only
git commit -m "feat(immunity): 串接 Exp3 Round 2 獨立執行流程"
~~~

---

### Task 8: Documentation 與完整程式驗證

**Files:**
- Modify: immunity/exp3/README.md:20

**Interfaces:**
- Documents: formal／smoke commands、isolated outputs、93 approximation、33 baseline reuse、failure policy。

- [ ] **Step 1: Add the Round 2 README section**

在 Round 1 正式／smoke commands 後新增：

~~~markdown
## Round 2：Paper-style 93 特徵

Round 2 只從 phase 影像建立 93 個 paper-style approximation predictors，並只新訓練
extra_trees 與 random_forest。Round 1 的 33-feature 結果與 Dummy evidence 只讀引用，
不會重跑七個模型。

正式執行：

    conda run --no-capture-output -n ki67dtc python -m immunity.exp3.run_round2_paper93 --config immunity/configs/exp3_round2_paper93.yaml

流程 smoke：

    conda run --no-capture-output -n ki67dtc python -m immunity.exp3.run_round2_paper93 --config immunity/configs/exp3_round2_paper93.yaml --smoke-fovs-per-condition 1

正式結果寫入 immunity/outputs/exp3/round2_paper93/，smoke 寫入其 smoke/ 子目錄。
IDO 只作亮度 target；結果不是免疫力 label，也不建立 donor label。Mask 或 frozen evidence
不一致時流程會停止並保留具名 QC，不會自動重新 segmentation 或增加 exclusion。
~~~

- [ ] **Step 2: Run Round 2 focused tests**

~~~powershell
conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_round2_evidence.py tests/test_immunity_exp3_round2_features.py tests/test_immunity_exp3_round2_benchmark.py tests/test_immunity_exp3_round2_reporting.py tests/test_immunity_exp3_round2_cli.py -q
~~~

Expected: PASS。

- [ ] **Step 3: Run all Exp3 regression tests**

~~~powershell
conda run --no-capture-output -n ki67dtc python -m pytest tests/test_immunity_exp3_manifest.py tests/test_immunity_exp3_phase_features.py tests/test_immunity_exp3_feature_sets.py tests/test_immunity_exp3_benchmark.py tests/test_immunity_exp3_reporting.py tests/test_immunity_exp3_cli.py tests/test_immunity_exp3_round2_evidence.py tests/test_immunity_exp3_round2_features.py tests/test_immunity_exp3_round2_benchmark.py tests/test_immunity_exp3_round2_reporting.py tests/test_immunity_exp3_round2_cli.py -q
~~~

Expected: PASS。tests/test_immunity_exp3_cli.py 可以執行，但不得 stage。

- [ ] **Step 4: Run the full project suite**

~~~powershell
conda run --no-capture-output -n ki67dtc python -m pytest -q
~~~

Expected: PASS。

- [ ] **Step 5: Verify formatting and preserve pre-existing dirty files**

~~~powershell
git diff --check
git status --short
git diff -- immunity/configs/exp3.yaml tests/test_immunity_exp3_cli.py
~~~

Expected: diff check 無輸出；兩個既存 dirty files 的 diff 與實作前一致。

- [ ] **Step 6: Commit documentation only**

~~~powershell
git add immunity/exp3/README.md
git diff --cached --name-only
git commit -m "docs(immunity): 補上 Exp3 Round 2 執行與驗證方式"
~~~

---

### Task 9: Execute and Validate the Isolated Smoke Run

**Files:**
- Runtime output: immunity/outputs/exp3/round2_paper93/smoke/
- Read-only source: immunity/outputs/exp3/

**Interfaces:**
- Consumes: completed Task 1–8 runner。
- Produces: non-formal smoke bundle and smoke EXPERIMENT_RECORD.md。

- [ ] **Step 1: Record Round 1 artifact hashes before smoke**

~~~powershell
Get-FileHash -Algorithm SHA256 immunity/outputs/exp3/data_manifest.csv,immunity/outputs/exp3/segmentation_qc.csv,immunity/outputs/exp3/outer_splits.csv,immunity/outputs/exp3/fold_metrics.csv,immunity/outputs/exp3/oof_predictions.csv,immunity/outputs/exp3/feature_cache/cell_level_basic.csv,immunity/outputs/exp3/feature_cache/image_level_basic.csv
~~~

Expected: hashes match the Frozen Round 1 Evidence table。

- [ ] **Step 2: Run smoke**

~~~powershell
conda run --no-capture-output -n ki67dtc python -m immunity.exp3.run_round2_paper93 --config immunity/configs/exp3_round2_paper93.yaml --smoke-fovs-per-condition 1
~~~

Expected: exit code 0；artifacts 只出現在 round2_paper93/smoke/。

- [ ] **Step 3: Validate smoke semantics**

~~~powershell
conda run --no-capture-output -n ki67dtc python -c "from pathlib import Path; from immunity.exp3.round2_reporting import validate_round2_bundle; validate_round2_bundle(Path('immunity/outputs/exp3/round2_paper93/smoke'), smoke=True); print('smoke bundle valid')"
~~~

Expected: smoke bundle valid。

人工確認：

- data snapshot 非空且通常為 72 FOV（9 groups × 8 conditions × 1）。
- feature_sets.json 是 93 predictors。
- 新 fit models 只有 extra_trees／random_forest。
- eligibility.csv 沒有正式 recommendation。
- EXPERIMENT_RECORD.md 明寫 smoke 不是正式結論。

- [ ] **Step 4: Recheck Round 1 hashes**

重跑 Step 1 command。Expected: 所有 hashes 完全相同。

---

### Task 10: Execute Full Round 2 and Produce the Experiment Record

**Files:**
- Runtime output: immunity/outputs/exp3/round2_paper93/
- Read-only source: immunity/outputs/exp3/

**Interfaces:**
- Produces: 完整 33 vs 93 comparison、eligibility、feature importance、metadata 與繁體中文 EXPERIMENT_RECORD.md。

- [ ] **Step 1: Run the complete 693-FOV experiment**

~~~powershell
conda run --no-capture-output -n ki67dtc python -m immunity.exp3.run_round2_paper93 --config immunity/configs/exp3_round2_paper93.yaml
~~~

Expected: exit code 0；只新增 46 個 outer fold-model evaluations（23 folds × 2 models；nested inner search 另計），不重跑 33-feature baseline。

- [ ] **Step 2: Validate the formal bundle**

~~~powershell
conda run --no-capture-output -n ki67dtc python -c "from pathlib import Path; from immunity.exp3.round2_reporting import validate_round2_bundle; validate_round2_bundle(Path('immunity/outputs/exp3/round2_paper93'), smoke=False); print('formal bundle valid')"
~~~

Expected: formal bundle valid。

- [ ] **Step 3: Verify exact formal evidence counts**

確認：

- data_snapshot.csv：693 unique image keys、93 predictors。
- mask_provenance_qc.csv：693 passed。
- feature_valid_counts.csv：41,580 rows（693 × 60 extras），全部 status=passed。
- outer_splits.csv：15,939 rows、23 folds。
- fold_metrics.csv：92 rows。
- dummy_fold_metrics.csv：23 rows。
- dummy_oof_predictions.csv：2,772 rows。
- feature_set_comparison.csv 與 eligibility.csv：各 4 rows。
- 只有一個 recommendation，或全部不合格且 recommendation 為空；不得強迫選 winner。

先檢查 model_failures.csv 與 fold_metrics.status。若 46 個新 fold-model evaluations 全部
成功，則預期 model_failures=0、oof_predictions=11,088、hyperparameters=92；完整 importance
diagnostic 時 feature_importance=5,796。若存在 model fold failure，bundle 仍可發布，但每個
failure 必須與 failed metric identity 及缺少的 OOF／hyperparameter rows 精確對齊，受影響
configuration 必須 ineligible；不得把失敗 fold 補值或把全 generation 誤報成功。Importance
少於 5,796 時，record 必須列出缺少的 diagnostic identities，不可靜默略過。

- [ ] **Step 4: Review the Traditional Chinese experiment record**

開啟 immunity/outputs/exp3/round2_paper93/EXPERIMENT_RECORD.md，人工確認：

- 第一段直接回答 93 是否優於 33。
- MAE 為 primary；RMSE、R²、Spearman 為 secondary。
- 說明 IDO brightness proxy 不是免疫力。
- 說明 93 是 phase-only paper-style approximation。
- 說明沒有 donor label、dose predictor 或 IDO predictor。
- 說明 Round 2 不改寫 Round 1 winner。
- 若 93 與 33 在 strict tie band，保留 33。

- [ ] **Step 5: Re-run final verification**

~~~powershell
conda run --no-capture-output -n ki67dtc python -m pytest -q
git diff --check
git status --short
~~~

Expected: tests PASS、diff check 無輸出；只保留已知未提交的 exp3.yaml／test_immunity_exp3_cli.py changes，以及任何刻意未 commit 的 runtime outputs。

## Standalone Validator Acceptance Checklist（Final Fix Wave）

- [x] Published root 在首次 rename 前枚舉所有 children；只接受固定 bundle files、
  安全 `_generations/`，以及 formal root 的安全 regular `smoke/`。
- [x] Publication 完成全部 staging replaces 後、刪除 staging 前，重新驗證 published root；
  `BaseException` 走 best-effort rollback 並重拋原始物件。
- [x] `run_metadata.json.reproducibility` 精確驗證 Git commit／branch／porcelain status／dirty、
  Python、五個主要 package versions、UTC start/completion 與 finite runtime/duration 一致性；
  dirty worktree 是合法 provenance，不要求 clean Git。
- [x] `baseline_provenance.json` 精確驗證 schema、`read_only=true`、Round 1 root、roster hash、
  artifact hash identities，並與 original/effective config evidence 對帳。
- [x] `data_snapshot.csv` 的 target 與全部 93 predictors 都必須 numeric finite；
  `mask_provenance_qc.csv` 必須 exact image coverage、無重複且全部 passed。
- [x] `feature_valid_counts.csv` 必須 exact data FOV × 60 extras；所有 count 使用 strict integer
  parser，固定 minimum=3，並重算 status、finite≤roster 與逐 FOV roster 一致性。
- [x] `extraction_qc.csv` 必須 exact image coverage、全部 passed、extracted=roster、固定 pair
  aggregation unit、pair topology count consistency，以及 frozen outside fraction ≤0.05；所有 totals
  與 `run_metadata.json`／`pair_mapping_summary` 重新對帳。
- [x] Candidate 與 Dummy OOF observed/predicted 必須 numeric finite，observed 必須逐列等於
  data target；formal successful identities 必須精確符合 frozen test membership，每個
  configuration × validation family × image 恰好一次，failed folds 與 metrics／failures／OOF／
  hyperparameters 一致。
- [x] Formal ranking 由 raw folds、candidate/Dummy OOF 與 failures 重新計算 pooled family MAE、
  `method=min` ranks、average/worst rank、五個 eligibility gates、strict `<0.25` tie band、
  same-algorithm 33-feature elimination 與 deterministic recommendation；
  `feature_set_comparison.csv`、`eligibility.csv` 的每個 derived field 都逐欄比對。
- [x] Smoke 只接受非空 strict subset、Paper93 ET/RF 流程、無 recommendation，且 metadata／
  record 必須同時明示 diagnostic、non-formal 與不形成科學結論。

## Execution Handoff

推薦用 superpowers:subagent-driven-development：每個 Task 使用 fresh subagent，完成後依序做 spec compliance 與 code quality review，再進下一 Task。若改用 superpowers:executing-plans，則以 Task 1–3、Task 4–7、Task 8–10 三個 checkpoints 分批執行；任何 preflight／smoke failure 都先停下來檢查 evidence，不直接進 formal full run。
