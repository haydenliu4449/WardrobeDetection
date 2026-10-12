"""Manage the wardrobe catalog.

    python3 closet.py add photo.jpg
    python3 closet.py list
    python3 closet.py remove top-03
"""

import argparse
import json
import sys

from PIL import Image, ImageOps

import config
from colors import COLORS

CATEGORIES = ["top", "bottom", "dress", "outerwear", "shoes"]
PHOTO_SIZE = 800


def load(path=config.WARDROBE_FILE):
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def save(items, path=config.WARDROBE_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(items, f, indent=2)
    tmp.replace(path)


def next_id(items, category):
    prefix = {"outerwear": "outer"}.get(category, category)
    used = [int(i["id"].split("-")[-1]) for i in items if i["id"].startswith(prefix + "-")]
    return f"{prefix}-{max(used, default=0) + 1:02d}"


def ask(prompt, default=None, choices=None, cast=str):
    hint = f" [{default}]" if default is not None else ""
    if choices:
        hint = f" ({'/'.join(choices)})" + hint
    while True:
        answer = input(f"{prompt}{hint}: ").strip() or default
        if answer is None:
            continue
        if choices and answer not in choices:
            print("  pick one of:", ", ".join(choices))
            continue
        try:
            return cast(answer)
        except ValueError:
            print("  that doesn't look right, try again")


def add(args):
    items = load()
    category = args.category or ask("category", choices=CATEGORIES)
    item = {
        "id": next_id(items, category),
        "name": args.name or ask("name (e.g. 'white linen shirt')"),
        "category": category,
        "color": args.color or ask("main color", default="black"),
        "warmth": args.warmth or ask("warmth 1-5 (1 = summer, 5 = winter coat)", default=3, cast=int),
        "formality": args.formality or ask("formality 1-3 (1 = casual, 3 = dressy)", default=1, cast=int),
        "patterned": args.patterned,
        "tags": args.tags or [],
    }
    if item["color"].lower() not in COLORS:
        print(f"note: '{item['color']}' isn't in colors.py, it'll be treated as a neutral")

    if args.photo:
        img = ImageOps.exif_transpose(Image.open(args.photo)).convert("RGB")
        img.thumbnail((PHOTO_SIZE, PHOTO_SIZE))
        config.PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{item['id']}.jpg"
        img.save(config.PHOTOS_DIR / filename, quality=88)
        item["photo"] = filename

    items.append(item)
    save(items)
    print(f"added {item['id']}: {item['name']}")


def list_items(args):
    items = load()
    if not items:
        print("wardrobe is empty, add something with: python3 closet.py add photo.jpg")
        return
    for cat in CATEGORIES:
        group = [i for i in items if i["category"] == cat]
        if not group:
            continue
        print(f"\n{cat} ({len(group)})")
        for i in group:
            extra = " patterned" if i.get("patterned") else ""
            print(f"  {i['id']:<10} {i['name']:<32} {i['color']:<12} w{i['warmth']} f{i['formality']}{extra}")


def remove(args):
    items = load()
    keep = [i for i in items if i["id"] != args.id]
    if len(keep) == len(items):
        sys.exit(f"no item with id {args.id}")
    gone = next(i for i in items if i["id"] == args.id)
    if gone.get("photo"):
        (config.PHOTOS_DIR / gone["photo"]).unlink(missing_ok=True)
    save(keep)
    print(f"removed {args.id}")


def main():
    parser = argparse.ArgumentParser(description="wardrobe catalog")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add", help="add an item (prompts for anything not given)")
    p.add_argument("photo", nargs="?", help="photo of the item, ideally on a dark background")
    p.add_argument("--name")
    p.add_argument("--category", choices=CATEGORIES)
    p.add_argument("--color")
    p.add_argument("--warmth", type=int, choices=range(1, 6))
    p.add_argument("--formality", type=int, choices=range(1, 4))
    p.add_argument("--patterned", action="store_true")
    p.add_argument("--tags", nargs="*", help="e.g. waterproof, avoid_rain")
    p.set_defaults(func=add)

    p = sub.add_parser("list")
    p.set_defaults(func=list_items)

    p = sub.add_parser("remove")
    p.add_argument("id")
    p.set_defaults(func=remove)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
