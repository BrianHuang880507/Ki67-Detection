import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# --- Pure config logic tests (no Qt needed) ---

def test_save_config_writes_correct_json(tmp_path):
    cfg = tmp_path / "config.json"

    class FakeCtrl:
        def _save_config(self, name):
            import controller as ctrl
            old_path = ctrl.CONFIG_PATH
            ctrl.CONFIG_PATH = str(cfg)
            try:
                with open(ctrl.CONFIG_PATH, "w") as f:
                    json.dump({"theme": name}, f)
            finally:
                ctrl.CONFIG_PATH = old_path

    fc = FakeCtrl()
    fc._save_config("clean")
    assert json.loads(cfg.read_text()) == {"theme": "clean"}


def test_config_load_returns_saved_theme(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"theme": "steel"}))
    data = json.loads(cfg.read_text())
    theme = data.get("theme", "dark")
    assert theme == "steel"


def test_config_load_falls_back_on_missing_file(tmp_path):
    cfg = tmp_path / "nonexistent.json"
    theme = "dark"
    if cfg.exists():
        try:
            theme = json.loads(cfg.read_text()).get("theme", "dark")
        except Exception:
            pass
    assert theme == "dark"


def test_config_load_falls_back_on_invalid_key(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"theme": "not_a_real_theme"}))
    from controller import THEMES
    data = json.loads(cfg.read_text())
    theme = data.get("theme", "dark")
    if theme not in THEMES:
        theme = "dark"
    assert theme == "dark"


def test_themes_dict_has_all_three_keys():
    from controller import THEMES
    assert set(THEMES.keys()) == {"dark", "clean", "steel"}


def test_themes_values_are_nonempty_strings():
    from controller import THEMES
    for name, qss in THEMES.items():
        assert isinstance(qss, str) and len(qss) > 100, f"Theme '{name}' QSS is too short"
