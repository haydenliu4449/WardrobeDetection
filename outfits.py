"""Builds outfits from whatever is in the closet today.

Every valid combination gets a score (color match, weather, formality,
what was suggested recently) and the best few that don't share pieces win.
"""

import random
from collections import defaultdict
from datetime import date, timedelta
from itertools import product

import colors

ORDER = {"outerwear": 0, "top": 1, "dress": 1, "bottom": 2, "shoes": 3}
CORE = ("top", "bottom", "dress")


def fahrenheit(temp, units):
    return temp if units == "fahrenheit" else temp * 9 / 5 + 32


def target_warmth(weather):
    """Map the day's feels-like high onto the same 1-5 scale the items use."""
    f = fahrenheit(weather["feels_like"], weather["units"])
    if f >= 85:
        return 1
    if f >= 75:
        return 2
    if f >= 62:
        return 3
    if f >= 50:
        return 4
    return 5


def needs_layer(weather):
    # mornings are cold even when the afternoon isn't, so look at the low too
    low = fahrenheit(weather["low"], weather["units"])
    return target_warmth(weather) >= 4 or low < 50 or weather.get("rain_chance", 0) >= 50


def recent_ids(history, today, days):
    cutoff = today - timedelta(days=days)
    ids = set()
    for entry in history:
        if date.fromisoformat(entry["date"]) >= cutoff and entry["date"] != today.isoformat():
            ids.update(entry["items"])
    return ids


def combinations(items, layer):
    by_cat = defaultdict(list)
    for item in items:
        by_cat[item["category"]].append(item)

    cores = [list(pair) for pair in product(by_cat["top"], by_cat["bottom"])]
    cores += [[d] for d in by_cat["dress"]]
    shoes = by_cat["shoes"] or [None]
    outers = by_cat["outerwear"] if layer and by_cat["outerwear"] else [None]

    for core, shoe, outer in product(cores, shoes, outers):
        pieces = core + [p for p in (shoe, outer) if p]
        yield sorted(pieces, key=lambda p: ORDER[p["category"]])


def score(pieces, target, formality, recent, rainy):
    s = 2.0 * colors.outfit_score([p["color"] for p in pieces])

    core = [p for p in pieces if p["category"] in CORE]
    core_warmth = sum(p["warmth"] for p in core) / len(core)
    # the coat handles the really cold days, so the base layer tops out at 4
    s -= 0.5 * abs(core_warmth - min(target, 4))

    outer = next((p for p in pieces if p["category"] == "outerwear"), None)
    if outer:
        s -= 0.3 * abs(outer["warmth"] - target)
        if rainy and "waterproof" in outer.get("tags", []):
            s += 0.4

    if rainy:
        s -= 0.5 * sum("avoid_rain" in p.get("tags", []) for p in pieces)

    levels = [p["formality"] for p in pieces]
    s -= 0.3 * sum(abs(lvl - formality) for lvl in levels) / len(levels)
    if max(levels) - min(levels) > 1:
        s -= 0.5  # sneakers with a blazer-and-slacks kind of mismatch

    if sum(bool(p.get("patterned")) for p in pieces) > 1:
        s -= 0.6

    s -= 0.6 * sum(p["id"] in recent for p in pieces)
    return s


def suggest(items, weather, history=(), count=3, repeat_days=4,
            weekday_formality=2, weekend_formality=1, today=None):
    today = today or date.today()
    target = target_warmth(weather)
    rainy = weather.get("rain_chance", 0) >= 50
    formality = weekend_formality if today.weekday() >= 5 else weekday_formality
    recent = recent_ids(history, today, repeat_days)

    # seeded by date: different picks each day, but the same picks if it reruns
    rng = random.Random(today.isoformat())

    scored = []
    for pieces in combinations(items, needs_layer(weather)):
        s = score(pieces, target, formality, recent, rainy) + rng.uniform(0, 0.25)
        scored.append((s, pieces))
    scored.sort(key=lambda x: x[0], reverse=True)

    picks, used = [], set()
    for s, pieces in scored:
        core_ids = {p["id"] for p in pieces if p["category"] in CORE}
        if core_ids & used:
            continue
        used |= core_ids
        picks.append({"items": [p["id"] for p in pieces], "score": round(s, 2)})
        if len(picks) == count:
            break
    return picks
