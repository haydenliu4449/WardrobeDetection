"""Draws the mirror screen with Pillow.

display.py only pushes the finished image to the screen. Doing the layout
here means it can be tested anywhere (python3 display.py --snapshot) and the
Zero only does the slow image resizing once per outfit, not every frame.
"""

from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

import colors
import config

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONTS = {"light": "DejaVuSans-ExtraLight.ttf", "regular": "DejaVuSans.ttf", "bold": "DejaVuSans-Bold.ttf"}

WHITE = (235, 235, 235)
GREY = (130, 130, 130)
BLACK = (0, 0, 0)

# relative height each category gets in a column
SLOT_WEIGHT = {"outerwear": 2.0, "top": 1.0, "dress": 1.9, "bottom": 1.1, "shoes": 0.55}


def load_font(size, bold=False, weight=None):
    weight = weight or ("bold" if bold else "regular")
    for name in (FONTS[weight], FONTS["regular"]):
        path = FONT_DIR / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1, install fonts-dejavu-core instead
        return ImageFont.load_default()


def fit(img, w, h):
    img = img.copy()
    img.thumbnail((w, h))
    return img


def item_image(item, w, h, photos_dir):
    photo = item.get("photo")
    if photo and (photos_dir / photo).exists():
        img = ImageOps.exif_transpose(Image.open(photos_dir / photo)).convert("RGB")
        return fit(img, w, h)

    # no photo yet, draw a color swatch instead
    sw, sh = int(w * 0.6), int(h * 0.75)
    side = min(sw, sh) if item["category"] == "shoes" else None
    if side:
        sw, sh = side, int(side * 0.6)
    swatch = Image.new("RGB", (sw, sh), BLACK)
    d = ImageDraw.Draw(swatch)
    d.rounded_rectangle((0, 0, sw - 1, sh - 1), radius=max(6, sw // 12),
                        fill=colors.rgb(item["color"]), outline=GREY, width=2)
    return swatch


def clip(draw, text, font, max_w):
    if draw.textlength(text, font=font) <= max_w:
        return text
    while text and draw.textlength(text + "…", font=font) > max_w:
        text = text[:-1]
    return text.rstrip() + "…"


def draw_column(canvas, draw, pieces, box, photos_dir, label_font):
    x, y, w, h = box
    label_h = int(label_font.size * 1.6)
    weights = [SLOT_WEIGHT[p["category"]] for p in pieces]
    unit = h / sum(weights)
    gap = int(h * 0.015)

    for piece, weight in zip(pieces, weights):
        slot_h = int(unit * weight)
        img_h = max(slot_h - label_h - gap, 10)
        img = item_image(piece, w, img_h, photos_dir)
        # center the picture + its label together so the label hugs the picture
        img_y = y + (slot_h - gap - img.height - label_h) // 2
        canvas.paste(img, (x + (w - img.width) // 2, img_y))
        draw.text((x + w // 2, img_y + img.height + 10), clip(draw, piece["name"], label_font, w),
                  fill=GREY, font=label_font, anchor="mt")
        y += slot_h


def weather_line(w):
    deg = "°"
    parts = [f"{w['high']}{deg} / {w['low']}{deg}"]
    summary, chance = w.get("summary", ""), w.get("rain_chance", 0)
    if summary in ("rain", "drizzle", "thunderstorms", "snow"):
        summary = f"{summary} {chance}%"
    elif chance >= 30:
        summary = f"{summary}, {chance}% chance of rain" if summary else f"{chance}% chance of rain"
    if summary:
        parts.append(summary)
    return "    ".join(parts)


def panel_box(size, disp):
    width, height = size
    pw = int(width * disp.get("panel_width", 1.0))
    align = disp.get("align", "center")
    px = {"left": 0, "right": width - pw}.get(align, (width - pw) // 2)
    return px, pw


def render(state, index, size, disp, items_by_id, photos_dir=config.PHOTOS_DIR):
    width, height = size
    canvas = Image.new("RGB", size, BLACK)
    draw = ImageDraw.Draw(canvas)
    px, pw = panel_box(size, disp)
    margin = int(pw * 0.06)
    cx = px + pw // 2

    day = date.fromisoformat(state["date"])
    big = load_font(int(height * 0.045), weight="light")
    mid = load_font(int(height * 0.018))
    small = load_font(int(height * 0.014))

    y = int(height * 0.04)
    draw.text((cx, y), day.strftime("%A"), fill=WHITE, font=big, anchor="mt")
    y += int(big.size * 1.25)
    draw.text((cx, y), f"{day:%B} {day.day}", fill=GREY, font=mid, anchor="mt")
    y += int(mid.size * 1.6)
    if state.get("weather"):
        draw.text((cx, y), weather_line(state["weather"]), fill=WHITE, font=mid, anchor="mt")
    top = y + int(mid.size * 2.8)

    outfits = state.get("outfits", [])
    footer = height - int(height * 0.06)
    dot_r = max(5, height // 220)
    for n in range(len(outfits)):
        dx = cx + (n - (len(outfits) - 1) / 2) * dot_r * 5
        bbox = (dx - dot_r, footer - dot_r, dx + dot_r, footer + dot_r)
        if n == index:
            draw.ellipse(bbox, fill=WHITE)
        else:
            draw.ellipse(bbox, outline=GREY, width=2)

    pieces = [items_by_id[i] for i in outfits[index]["items"] if i in items_by_id]
    area_h = footer - top - int(height * 0.04)
    inner_x, inner_w = px + margin, pw - 2 * margin

    outer = [p for p in pieces if p["category"] == "outerwear"]
    rest = [p for p in pieces if p["category"] != "outerwear"]
    if outer:
        col_w = inner_w // 2 - margin // 2
        draw_column(canvas, draw, outer, (inner_x, top + area_h // 6, col_w, area_h * 2 // 3),
                    photos_dir, small)
        draw_column(canvas, draw, rest, (inner_x + inner_w - col_w, top, col_w, area_h),
                    photos_dir, small)
    else:
        draw_column(canvas, draw, rest, (inner_x, top, inner_w, area_h), photos_dir, small)

    return canvas


def render_message(size, text, sub=""):
    width, height = size
    canvas = Image.new("RGB", size, BLACK)
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, height // 2), text, fill=WHITE,
              font=load_font(int(height * 0.025), weight="light"), anchor="mm")
    if sub:
        draw.text((width // 2, height // 2 + int(height * 0.035)), sub, fill=GREY,
                  font=load_font(int(height * 0.014)), anchor="mm")
    return canvas
