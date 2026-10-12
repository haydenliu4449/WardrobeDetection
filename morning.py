"""The morning run: photo -> what's in the closet -> weather -> outfits -> data/today.json

display.py watches today.json and redraws when it changes.

    python3 morning.py                    normal run (the systemd timer does this)
    python3 morning.py --photo shot.jpg   use an existing photo instead of the camera
    python3 morning.py --no-detect        skip detection and assume everything is in the closet
    python3 morning.py --demo             sample wardrobe, no camera or API, for showing it on a laptop
"""

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import capture
import closet
import config
import detect
import outfits
import weather

log = logging.getLogger("morning")

LAST_INVENTORY = config.DATA_DIR / "last_inventory.json"
MIN_DETECTED = 3


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, data):
    # write then rename, so the display never reads a half-written file
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def run_detection(photo, items, cfg):
    if cfg.get("detect", {}).get("method") == "yolo":
        import yolo_detect  # needs onnxruntime, only installed when using it
        return yolo_detect.detect(photo, items, cfg)
    return detect.detect(photo, items, cfg)


def check_closet(items, cfg, photo=None):
    """Returns (items that are in the closet, extra info for today.json)."""
    try:
        photo = Path(photo) if photo else capture.take_photo(cfg)
        capture.cleanup()
        result = run_detection(photo, items, cfg)
    except Exception as e:
        log.error("couldn't check the closet: %s", e)
        result = None

    # a handful of items usually means a bad photo (door shut, light off),
    # not that the closet is actually empty
    if result:
        found = len(result["present"]) + len(result["unsure"])
        if found < MIN_DETECTED:
            log.warning("only %d items detected, not trusting this photo", found)
            result = None

    if result is None:
        last = load_json(LAST_INVENTORY, None)
        if last:
            log.info("falling back to the inventory from %s", last["date"])
            ids = set(last["ids"])
        else:
            log.info("no previous inventory, assuming everything is there")
            return items, {}
        return [i for i in items if i["id"] in ids], {"stale_inventory": last["date"]}

    # unsure items count as present: suggesting something that turns out to be
    # in the wash is less annoying than never suggesting it
    ids = set(result["present"]) | set(result["unsure"])
    write_json(LAST_INVENTORY, {"date": date.today().isoformat(), "ids": sorted(ids)})
    info = {
        "missing": [i["id"] for i in items if i["id"] not in ids],
        "unknown": result["unknown"],
    }
    return [i for i in items if i["id"] in ids], info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", help="closet photo to use instead of taking one")
    parser.add_argument("--no-detect", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.load_env()
    cfg = config.load()
    today = date.today()

    wardrobe_path = config.SAMPLE_WARDROBE if args.demo else config.WARDROBE_FILE
    items = closet.load(wardrobe_path)
    state = {
        "date": today.isoformat(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "wardrobe": str(wardrobe_path),
    }
    if not items:
        state.update(outfits=[], error="wardrobe is empty, add clothes with closet.py")
        write_json(config.STATE_FILE, state)
        sys.exit("wardrobe is empty, add clothes with: python3 closet.py add photo.jpg")

    forecast = weather.get(cfg)

    if args.demo or args.no_detect:
        available, info = items, {}
    else:
        available, info = check_closet(items, cfg, args.photo)

    history = [] if args.demo else load_json(config.HISTORY_FILE, [])
    opts = cfg["outfits"]
    picks = outfits.suggest(
        available, forecast, history,
        count=opts["count"],
        repeat_days=opts["repeat_days"],
        weekday_formality=opts["weekday_formality"],
        weekend_formality=opts["weekend_formality"],
        today=today,
    )

    state.update(weather=forecast, outfits=picks, **info)
    if not picks:
        state["error"] = "not enough clothes in the closet to make an outfit"
    write_json(config.STATE_FILE, state)

    names = {i["id"]: i["name"] for i in items}
    for n, pick in enumerate(picks, 1):
        log.info("outfit %d (%.2f): %s", n, pick["score"], " + ".join(names[i] for i in pick["items"]))

    if picks and not args.demo:
        # remember the first suggestion so tomorrow leans toward other pieces
        history = [h for h in history if h["date"] != today.isoformat()]
        history.append({"date": today.isoformat(), "items": picks[0]["items"]})
        write_json(config.HISTORY_FILE, history[-30:])


if __name__ == "__main__":
    main()
