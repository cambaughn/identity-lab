"""Settings persistence: defaults, round-trip, and malformed-input fallback."""

from identity_lab.config.settings import AppSettings, load_settings, save_settings


def test_defaults_when_file_missing(tmp_path):
    settings = load_settings(tmp_path / "nope.json")
    assert settings == AppSettings()
    assert settings.camera_index == 0
    assert settings.window_geometry is None


def test_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    original = AppSettings(camera_index=2, window_geometry="AAEC")
    save_settings(original, path)
    assert load_settings(path) == original


def test_save_creates_parent_directory(tmp_path):
    path = tmp_path / "deep" / "nested" / "settings.json"
    save_settings(AppSettings(camera_index=1), path)
    assert load_settings(path).camera_index == 1


def test_corrupted_json_falls_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not valid json!!")
    assert load_settings(path) == AppSettings()


def test_non_dict_json_falls_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('["a", "list"]')
    assert load_settings(path) == AppSettings()


def test_wrong_types_fall_back_per_field(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"camera_index": "two", "window_geometry": "AAEC"}')
    loaded = load_settings(path)
    assert loaded.camera_index == 0          # bad value -> default
    assert loaded.window_geometry == "AAEC"  # good value kept


def test_negative_and_bool_camera_index_rejected(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"camera_index": -3}')
    assert load_settings(path).camera_index == 0
    path.write_text('{"camera_index": true}')
    assert load_settings(path).camera_index == 0


def test_unknown_keys_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"camera_index": 1, "surprise": 42}')
    assert load_settings(path).camera_index == 1
