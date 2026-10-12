import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import colors
import outfits

MONDAY = date(2026, 10, 12)


def item(id, category, color="black", warmth=3, formality=1, **extra):
    return {"id": id, "name": id, "category": category, "color": color,
            "warmth": warmth, "formality": formality, **extra}


def weather(feels_like, low=None, rain=0):
    return {"high": feels_like, "low": low if low is not None else feels_like - 8,
            "feels_like": feels_like, "rain_chance": rain, "units": "fahrenheit"}


WARDROBE = [
    item("top-01", "top", "white", warmth=2),
    item("top-02", "top", "cream", warmth=4),
    item("top-03", "top", "red", warmth=1),
    item("bottom-01", "bottom", "denim"),
    item("bottom-02", "bottom", "black", warmth=2),
    item("dress-01", "dress", "navy", warmth=2),
    item("outer-01", "outerwear", "camel", warmth=5),
    item("outer-02", "outerwear", "khaki", warmth=2, tags=["waterproof"]),
    item("shoes-01", "shoes", "white"),
]


class TestOutfits(unittest.TestCase):
    def test_every_outfit_is_complete(self):
        for pick in outfits.suggest(WARDROBE, weather(70), today=MONDAY):
            cats = [i.split("-")[0] for i in pick["items"]]
            self.assertIn("shoes", cats)
            if "dress" in cats:
                self.assertNotIn("top", cats)
                self.assertNotIn("bottom", cats)
            else:
                self.assertIn("top", cats)
                self.assertIn("bottom", cats)

    def test_cold_day_gets_a_coat(self):
        picks = outfits.suggest(WARDROBE, weather(38), today=MONDAY)
        self.assertTrue(picks)
        for pick in picks:
            self.assertTrue(any(i.startswith("outer") for i in pick["items"]))
        self.assertIn("outer-01", picks[0]["items"])

    def test_warm_day_skips_outerwear(self):
        for pick in outfits.suggest(WARDROBE, weather(80, low=62), today=MONDAY):
            self.assertFalse(any(i.startswith("outer") for i in pick["items"]))

    def test_rain_prefers_waterproof_layer(self):
        picks = outfits.suggest(WARDROBE, weather(70, low=60, rain=80), today=MONDAY)
        self.assertIn("outer-02", picks[0]["items"])

    def test_picks_dont_share_main_pieces(self):
        picks = outfits.suggest(WARDROBE, weather(65), count=3, today=MONDAY)
        seen = set()
        for pick in picks:
            core = {i for i in pick["items"] if not i.startswith(("shoes", "outer"))}
            self.assertFalse(core & seen)
            seen |= core

    def test_recent_items_are_avoided(self):
        first = outfits.suggest(WARDROBE, weather(65), count=1, today=MONDAY)[0]
        history = [{"date": "2026-10-11", "items": first["items"]}]
        second = outfits.suggest(WARDROBE, weather(65), history, count=1, today=MONDAY)[0]
        self.assertNotEqual(first["items"], second["items"])

    def test_same_day_is_repeatable(self):
        a = outfits.suggest(WARDROBE, weather(65), today=MONDAY)
        b = outfits.suggest(WARDROBE, weather(65), today=MONDAY)
        self.assertEqual(a, b)

    def test_nothing_to_wear(self):
        only_tops = [i for i in WARDROBE if i["category"] == "top"]
        self.assertEqual(outfits.suggest(only_tops, weather(65), today=MONDAY), [])

    def test_target_warmth_in_celsius(self):
        w = {"feels_like": 30, "low": 22, "units": "celsius"}
        self.assertEqual(outfits.target_warmth(w), 1)


class TestColors(unittest.TestCase):
    def test_neutrals_go_with_anything(self):
        self.assertEqual(colors.pair_score("navy", "red"), 1.0)
        self.assertEqual(colors.outfit_score(["black", "white", "denim"]), 1.0)

    def test_tonal_beats_complementary_beats_clash(self):
        tonal = colors.pair_score("red", "burgundy")
        complementary = colors.pair_score("red", "teal")
        clash = colors.pair_score("red", "green")
        self.assertGreater(tonal, complementary)
        self.assertGreater(complementary, clash)

    def test_three_bold_colors_is_too_many(self):
        self.assertLess(colors.outfit_score(["red", "yellow", "purple"]),
                        colors.outfit_score(["red", "burgundy", "black"]))

    def test_unknown_color_is_neutral(self):
        self.assertTrue(colors.is_neutral("chartreuse-ish"))


if __name__ == "__main__":
    unittest.main()
