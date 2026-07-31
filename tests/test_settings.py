"""Settings persistence: defaults, round-trip, and malformed-input fallback."""

from identity_lab.config.settings import AppSettings, load_settings, save_settings


def test_defaults_when_file_missing(tmp_path):
    settings = load_settings(tmp_path / "nope.json")
    assert settings == AppSettings()
    assert settings.camera_device_id is None
    assert settings.infer_every_n == 2
    assert settings.min_face_px == 60


def test_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    original = AppSettings(
        camera_device_id="ABC-123",
        show_landmarks=True,
        debug_mode=True,
        infer_every_n=5,
        min_face_px=120,
        window_geometry="AAEC",
    )
    save_settings(original, path)
    assert load_settings(path) == original


def test_save_creates_parent_directory(tmp_path):
    path = tmp_path / "deep" / "nested" / "settings.json"
    save_settings(AppSettings(camera_device_id="X"), path)
    assert load_settings(path).camera_device_id == "X"


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
    path.write_text(
        '{"camera_device_id": 7, "show_landmarks": "yes", "debug_mode": 1,'
        ' "infer_every_n": "two", "min_face_px": 60.5, "window_geometry": "AAEC"}'
    )
    loaded = load_settings(path)
    assert loaded.camera_device_id is None   # bad -> default
    assert loaded.show_landmarks is False    # bad -> default
    assert loaded.debug_mode is False        # bad -> default
    assert loaded.infer_every_n == 2         # bad -> default
    assert loaded.min_face_px == 60          # bad -> default
    assert loaded.window_geometry == "AAEC"  # good value kept


def test_out_of_range_numbers_rejected(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"infer_every_n": 0, "min_face_px": -5}')
    loaded = load_settings(path)
    assert loaded.infer_every_n == 2
    assert loaded.min_face_px == 60
    path.write_text('{"infer_every_n": 31, "min_face_px": 1001}')
    loaded = load_settings(path)
    assert loaded.infer_every_n == 2
    assert loaded.min_face_px == 60


def test_bool_rejected_for_ints(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"infer_every_n": true, "min_face_px": false}')
    loaded = load_settings(path)
    assert loaded.infer_every_n == 2
    assert loaded.min_face_px == 60


def test_unknown_and_legacy_keys_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"camera_index": 2, "surprise": 42, "min_face_px": 100}')
    loaded = load_settings(path)
    assert loaded.min_face_px == 100
    assert loaded == AppSettings(min_face_px=100)
