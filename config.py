"""Loads config.toml (falls back to defaults) and the .env file with the API key."""

import copy
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PHOTOS_DIR = DATA_DIR / "photos"
CAPTURES_DIR = DATA_DIR / "captures"
TRAINING_PHOTOS = DATA_DIR / "training_photos"
WARDROBE_FILE = DATA_DIR / "wardrobe.json"
STATE_FILE = DATA_DIR / "today.json"
HISTORY_FILE = DATA_DIR / "history.json"
SAMPLE_WARDROBE = ROOT / "sample_wardrobe.json"

DEFAULTS = {
    "location": {
        "latitude": 37.77,
        "longitude": -122.42,
        "units": "fahrenheit",
    },
    "camera": {
        "device": 0,
        "width": 1920,
        "height": 1080,
        "warmup_frames": 20,
        "light_pin": 0,
        "light_settle": 1.5,
        "collect_min_change": 1.0,
    },
    "detect": {
        "method": "api",
        "model": "claude-sonnet-5-5",
        "max_image_px": 1568,
        "timeout": 60,
    },
    "yolo": {
        "model": "models/closet.onnx",
        "present": 0.5,
        "unsure": 0.25,
        "api_fallback": True,
    },
    "outfits": {
        "count": 3,
        "repeat_days": 4,
        "weekday_formality": 2,
        "weekend_formality": 1,
    },
    "display": {
        "width": 1080,
        "height": 1920,
        "rotate": 0,
        "panel_width": 0.8,
        "align": "center",
        "show_from": "06:00",
        "show_until": "11:00",
        "cycle_seconds": 0,
        "button_pin": 0,
    },
}


def _merge(base, extra):
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


def load(path=None):
    path = Path(path or os.environ.get("MIRROR_CONFIG", ROOT / "config.toml"))
    cfg = copy.deepcopy(DEFAULTS)
    if path.exists():
        with open(path, "rb") as f:
            _merge(cfg, tomllib.load(f))
    return cfg


def load_env(path=ROOT / ".env"):
    # Small .env reader so the Pi doesn't need python-dotenv.
    # Real environment variables win over the file.
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
