"""Asks a vision model which wardrobe items are in today's closet photo.

Instead of asking the model to identify clothes from scratch, it gets the
closet photo plus a contact sheet of the catalog and just has to match them up.

    python3 detect.py data/captures/test.jpg
"""

import base64
import hashlib
import io
import json
import logging
import math
import os
import sys
import time

import requests
from PIL import Image, ImageDraw, ImageOps

import config
from render import load_font

log = logging.getLogger(__name__)

API_URL = "https://api.anthropic.com/v1/messages"
THUMB = 220
RETRY_STATUS = {429, 500, 502, 503, 529}

TOOL = {
    "name": "report_closet",
    "description": "Report which catalog items are in the closet photo.",
    "input_schema": {
        "type": "object",
        "properties": {
            "present": {
                "type": "array", "items": {"type": "string"},
                "description": "IDs of catalog items you are fairly sure are in the closet",
            },
            "unsure": {
                "type": "array", "items": {"type": "string"},
                "description": "IDs that might be there but you can't tell",
            },
            "unknown": {
                "type": "array", "items": {"type": "string"},
                "description": "short descriptions of clothes that don't match any catalog item",
            },
        },
        "required": ["present"],
    },
}

PROMPT = """The first image is a photo of a closet taken this morning.{sheet_note}

Wardrobe catalog (id | name | category | color):
{catalog}

Work out which catalog items are physically in the closet right now. Clothes are often \
partly hidden: on a hanger you might only see a sleeve or an edge, and folded things \
only show the side of a stack, so match on color, fabric and shape. Put items you're \
fairly sure about in "present" and plausible matches you can't confirm in "unsure". \
Anything that is clearly clothing but doesn't match the catalog goes in "unknown"."""

SHEET_NOTE = ("\nThe second image is a contact sheet of every catalog item that has a "
              "photo, each labeled with its id.")


def encode(img, max_px):
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def image_block(data):
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def contact_sheet(items):
    """Grid of item photos labeled by id. Cached until the catalog changes."""
    with_photos = [i for i in items
                   if i.get("photo") and (config.PHOTOS_DIR / i["photo"]).exists()]
    if not with_photos:
        return None

    stamp = [(i["id"], (config.PHOTOS_DIR / i["photo"]).stat().st_mtime) for i in with_photos]
    key = hashlib.sha1(json.dumps(stamp).encode()).hexdigest()[:10]
    path = config.DATA_DIR / f"contact_sheet_{key}.jpg"
    if path.exists():
        return Image.open(path)

    cols = min(8, math.ceil(math.sqrt(len(with_photos))))
    rows = math.ceil(len(with_photos) / cols)
    label_h = 30
    sheet = Image.new("RGB", (cols * THUMB, rows * (THUMB + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(22, bold=True)

    for n, item in enumerate(with_photos):
        x, y = (n % cols) * THUMB, (n // cols) * (THUMB + label_h)
        thumb = Image.open(config.PHOTOS_DIR / item["photo"]).convert("RGB")
        thumb.thumbnail((THUMB - 10, THUMB - 10))
        sheet.paste(thumb, (x + (THUMB - thumb.width) // 2, y + (THUMB - thumb.height) // 2))
        draw.text((x + THUMB // 2, y + THUMB + 2), item["id"], fill="black", font=font, anchor="mt")

    for old in config.DATA_DIR.glob("contact_sheet_*.jpg"):
        old.unlink()
    sheet.save(path, quality=85)
    return sheet


def call_api(content, model, api_key, timeout):
    body = {
        "model": model,
        "max_tokens": 1024,
        "tools": [TOOL],
        "tool_choice": {"type": "tool", "name": TOOL["name"]},
        "messages": [{"role": "user", "content": content}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    for attempt in range(3):
        try:
            r = requests.post(API_URL, json=body, headers=headers, timeout=timeout)
        except requests.ConnectionError as e:
            # wifi on the Zero is sometimes still waking up at 6am
            log.warning("connection error (%s), retrying", e)
        else:
            if r.status_code not in RETRY_STATUS:
                if r.status_code != 200:
                    raise RuntimeError(f"API error {r.status_code}: {r.text[:300]}")
                return r.json()
            log.warning("API returned %s, retrying", r.status_code)
        time.sleep(5 * 2 ** attempt)
    raise RuntimeError("API still failing after 3 attempts")


def parse(response, items):
    known = {i["id"] for i in items}
    block = next((b for b in response.get("content", []) if b.get("type") == "tool_use"), None)
    if block is None:
        raise RuntimeError("model didn't return the closet report")
    data = block["input"]
    # drop anything the model made up that isn't actually in the catalog
    return {
        "present": [x for x in data.get("present", []) if x in known],
        "unsure": [x for x in data.get("unsure", []) if x in known],
        "unknown": data.get("unknown", []),
    }


def detect(photo, items, cfg):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY isn't set (put it in .env)")
    opts = cfg["detect"]

    sheet = contact_sheet(items)
    catalog = "\n".join(f"{i['id']} | {i['name']} | {i['category']} | {i['color']}" for i in items)
    content = [image_block(encode(photo, opts["max_image_px"]))]
    if sheet is not None:
        content.append(image_block(encode(sheet, opts["max_image_px"])))
    content.append({"type": "text", "text": PROMPT.format(
        catalog=catalog, sheet_note=SHEET_NOTE if sheet is not None else "")})

    response = call_api(content, opts["model"], api_key, opts["timeout"])
    result = parse(response, items)
    log.info("detected %d present, %d unsure, %d unknown",
             len(result["present"]), len(result["unsure"]), len(result["unknown"]))
    return result


if __name__ == "__main__":
    import closet

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) != 2:
        sys.exit("usage: python3 detect.py <closet photo>")
    config.load_env()
    print(json.dumps(detect(sys.argv[1], closet.load(), config.load()), indent=2))
