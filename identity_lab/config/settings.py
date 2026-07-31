"""Persisted user settings — JSON in the macOS app-data directory.

Settings live at ~/Library/Application Support/IdentityLab/settings.json.
Loading is tolerant: a missing, corrupted, or wrongly-typed file falls back
to defaults (per-field where possible) instead of crashing the app.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

APP_DATA_DIR = Path.home() / "Library" / "Application Support" / "IdentityLab"
SETTINGS_FILENAME = "settings.json"

INFER_EVERY_N_RANGE = (1, 30)
MIN_FACE_PX_RANGE = (0, 1000)


@dataclass
class AppSettings:
    camera_device_id: str | None = None  # Qt unique device id
    show_landmarks: bool = False
    debug_mode: bool = False
    infer_every_n: int = 2               # run inference every Nth frame
    min_face_px: int = 60                # ignore faces smaller than this (px)
    window_geometry: str | None = None   # base64-encoded Qt geometry blob


def settings_path(data_dir: Path | None = None) -> Path:
    return (data_dir or APP_DATA_DIR) / SETTINGS_FILENAME


def _valid_int(value, lo: int, hi: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and lo <= value <= hi


def load_settings(path: Path | None = None) -> AppSettings:
    """Load settings, falling back to defaults on any malformed input."""
    path = path or settings_path()
    defaults = AppSettings()
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return defaults
    if not isinstance(raw, dict):
        return defaults

    loaded = AppSettings()
    v = raw.get("camera_device_id")
    if isinstance(v, str) or v is None:
        loaded.camera_device_id = v
    v = raw.get("show_landmarks")
    if isinstance(v, bool):
        loaded.show_landmarks = v
    v = raw.get("debug_mode")
    if isinstance(v, bool):
        loaded.debug_mode = v
    v = raw.get("infer_every_n")
    if _valid_int(v, *INFER_EVERY_N_RANGE):
        loaded.infer_every_n = v
    v = raw.get("min_face_px")
    if _valid_int(v, *MIN_FACE_PX_RANGE):
        loaded.min_face_px = v
    v = raw.get("window_geometry")
    if isinstance(v, str) or v is None:
        loaded.window_geometry = v
    return loaded


def save_settings(settings: AppSettings, path: Path | None = None) -> None:
    """Atomically write settings as JSON, creating the directory if needed."""
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(settings), indent=2))
    tmp.replace(path)
