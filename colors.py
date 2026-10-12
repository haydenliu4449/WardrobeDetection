import colorsys
from itertools import combinations

# Approximate RGB for the color names used in the wardrobe file.
# Used for matching and for drawing a swatch when an item has no photo.
COLORS = {
    "black": (20, 20, 20),
    "white": (245, 245, 240),
    "ivory": (240, 234, 214),
    "cream": (238, 228, 200),
    "grey": (140, 140, 140),
    "gray": (140, 140, 140),
    "charcoal": (64, 64, 68),
    "beige": (215, 195, 160),
    "tan": (200, 165, 120),
    "camel": (193, 154, 107),
    "khaki": (190, 175, 130),
    "brown": (110, 75, 45),
    "navy": (30, 40, 80),
    "denim": (70, 95, 140),
    "red": (200, 35, 40),
    "burgundy": (115, 25, 45),
    "pink": (235, 150, 175),
    "coral": (245, 120, 100),
    "orange": (235, 120, 30),
    "rust": (180, 80, 40),
    "mustard": (215, 170, 40),
    "yellow": (245, 215, 60),
    "olive": (110, 115, 50),
    "sage": (155, 175, 140),
    "green": (40, 140, 70),
    "emerald": (20, 120, 90),
    "teal": (20, 125, 130),
    "light blue": (150, 190, 225),
    "blue": (40, 90, 190),
    "lavender": (185, 165, 215),
    "purple": (110, 50, 140),
}

# These go with pretty much anything, so they never count against an outfit.
NEUTRALS = {
    "black", "white", "ivory", "cream", "grey", "gray", "charcoal",
    "beige", "tan", "camel", "khaki", "brown", "navy", "denim",
}


def rgb(name):
    return COLORS.get(name.lower().strip(), (128, 128, 128))


def is_neutral(name):
    name = name.lower().strip()
    # unknown colors are treated as neutral rather than guessing
    return name in NEUTRALS or name not in COLORS


def hue(name):
    r, g, b = (c / 255 for c in rgb(name))
    return colorsys.rgb_to_hsv(r, g, b)[0] * 360


def pair_score(a, b):
    """0..1, how well two colors sit together."""
    if is_neutral(a) or is_neutral(b):
        return 1.0
    diff = abs(hue(a) - hue(b))
    diff = min(diff, 360 - diff)
    if diff <= 25:
        return 0.9   # same family / tonal
    if diff <= 60:
        return 0.7   # analogous
    if 150 <= diff <= 210:
        return 0.6   # complementary, works if it's intentional
    return 0.2


def outfit_score(color_names):
    bold = [c for c in color_names if not is_neutral(c)]
    if len(bold) < 2:
        return 1.0
    scores = [pair_score(a, b) for a, b in combinations(bold, 2)]
    score = sum(scores) / len(scores)
    if len(set(bold)) > 2:
        score -= 0.3
    return max(score, 0.0)
