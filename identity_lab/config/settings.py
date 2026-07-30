"""Persisted user settings — JSON in the macOS app-data directory.

Settings live at ~/Library/Application Support/IdentityLab/settings.json.
Loading is tolerant: a missing, corrupted, or wrongly-typed file falls back
to defaults (per-field where possible) instead of crashing the app.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

APP_DATA_DIR = Path.home() / "Library" / "Application Support" / "IdentityLab"
SETTINGS_FILENAME = "settings.json"


@dataclass
class AppSettings:
    camera_index: int = 0
    window_geometry: str | None = None  # base64-encoded Qt geometry blob


def settings_path(data_dir: Path | None = None) -> Path:
    return (data_dir or APP_DATA_DIR) / SETTINGS_FILENAME


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
    for f in fields(AppSettings):
        if f.name not in raw:
            continue
        value = raw[f.name]
        if f.name == "camera_index":
            # bool is an int subclass; reject it explicitly
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                loaded.camera_index = value
        elif f.name == "window_geometry":
            if isinstance(value, str) or value is None:
                loaded.window_geometry = value
    return loaded


def save_settings(settings: AppSettings, path: Path | None = None) -> None:
    """Atomically write settings as JSON, creating the directory if needed."""
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(settings), indent=2))
    tmp.replace(path)
