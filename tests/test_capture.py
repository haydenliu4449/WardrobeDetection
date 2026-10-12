import random
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import capture


def closet_photo(path, shirt=None, noise=8, brightness=0, seed=0):
    """Grey closet with sensor noise, optionally a red shirt hanging in it."""
    rnd = random.Random(seed)
    img = Image.new("RGB", (640, 360), (95 + brightness,) * 3)
    noisy = Image.effect_noise((640, 360), noise).convert("RGB")
    img = Image.blend(img, noisy, 0.08)
    if shirt:
        ImageDraw.Draw(img).rectangle(shirt, fill=(200, 40, 40))
    img = img.rotate(rnd.uniform(-0.2, 0.2))  # the camera never sits perfectly still
    img.save(path)
    return path


class TestChangedShare(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.base = closet_photo(self.dir / "a.jpg", seed=1)

    def test_same_closet_is_unchanged(self):
        again = closet_photo(self.dir / "b.jpg", seed=2)
        self.assertLess(capture.changed_share(self.base, again), 1.0)

    def test_small_exposure_drift_is_ignored(self):
        brighter = closet_photo(self.dir / "b.jpg", brightness=8, seed=2)
        self.assertLess(capture.changed_share(self.base, brighter), 1.0)

    def test_one_shirt_counts_as_a_change(self):
        # a shirt is roughly 4% of the frame
        moved = closet_photo(self.dir / "b.jpg", shirt=(300, 60, 360, 300), seed=2)
        self.assertGreater(capture.changed_share(self.base, moved), 1.0)


if __name__ == "__main__":
    unittest.main()
