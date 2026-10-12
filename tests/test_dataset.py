import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

import prepare_dataset as prep

NAMES = ["bottom-01", "shoes-01", "top-01"]  # label studio sorts its classes.txt


def make_export(folder, days=8, per_day=3):
    """Something shaped like a Label Studio YOLO export."""
    (folder / "images").mkdir(parents=True)
    (folder / "labels").mkdir()
    (folder / "classes.txt").write_text("\n".join(NAMES) + "\n")
    (folder / "notes.json").write_text("{}")
    for d in range(1, days + 1):
        for h in range(per_day):
            stem = f"a1b2c3-2026-10-{d:02d}_{9 + h:02d}00"
            (folder / "images" / f"{stem}.jpg").write_bytes(b"jpg")
            (folder / "labels" / f"{stem}.txt").write_text("2 0.5 0.4 0.2 0.3\n0 0.5 0.8 0.2 0.3\n")
    (folder / "images" / "never-labeled.jpg").write_bytes(b"jpg")


class TestSplit(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.export = self.root / "export"
        self.out = self.root / "dataset"

    def run_split(self, photos=None):
        args = SimpleNamespace(export=self.export, out=self.out, val=0.15, photos=photos,
                               wardrobe=self.root / "missing.json")
        with redirect_stdout(StringIO()) as printed:
            prep.split(args)
        return printed.getvalue()

    def test_whole_days_go_to_val(self):
        make_export(self.export, days=14)
        self.run_split()
        days = lambda part: {prep.day_of(p) for p in (self.out / "images" / part).iterdir()}
        train, val = days("train"), days("val")
        self.assertTrue(val)
        self.assertFalse(train & val)
        self.assertEqual(len(train | val), 14)

    def test_yaml_and_counts(self):
        make_export(self.export, days=6, per_day=2)
        printed = self.run_split()
        yaml = (self.out / "closet.yaml").read_text()
        self.assertIn('0: "bottom-01"', yaml)
        self.assertIn('2: "top-01"', yaml)
        info = json.loads((self.out / "classes.json").read_text())
        self.assertEqual(info["names"], NAMES)
        total = info["train_counts"]["top-01"] + info["val_counts"].get("top-01", 0)
        self.assertEqual(total, 12)
        self.assertNotIn("shoes-01", info["train_counts"])
        self.assertIn("skipped 1 images with no label file", printed)

    def test_bad_label_lines_are_dropped(self):
        make_export(self.export, days=5, per_day=1)
        label = next((self.export / "labels").iterdir())
        label.write_text("7 0.5 0.5 0.1 0.1\n0 0.1 0.1 0.2 0.2 0.3 0.3\n1 0.5 0.5 0.1 0.1\n")
        printed = self.run_split()
        self.assertIn("skipped 2 label lines", printed)

    def test_wont_delete_an_unrelated_folder(self):
        make_export(self.export, days=5)
        self.out.mkdir()
        (self.out / "homework.docx").write_text("important")
        with self.assertRaises(SystemExit):
            self.run_split()
        self.assertTrue((self.out / "homework.docx").exists())

    def test_labels_only_export_with_original_photos(self):
        # Label Studio style: labels named "<id>-<original name>", images not included
        self.export.mkdir()
        photos = self.root / "training_photos"
        photos.mkdir()
        (self.export / "classes.txt").write_text("\n".join(NAMES))
        for d in range(1, 7):
            (photos / f"2026-10-{d:02d}_0900.jpg").write_bytes(b"jpg")
            (self.export / f"9f8e7d6c-2026-10-{d:02d}_0900.txt").write_text("1 0.5 0.5 0.1 0.1\n")

        with self.assertRaises(SystemExit):
            self.run_split()
        self.run_split(photos=photos)
        copied = list((self.out / "images").rglob("*.jpg"))
        self.assertEqual(len(copied), 6)
        for image in copied:
            label = self.out / "labels" / image.parent.name / f"{image.stem}.txt"
            self.assertEqual(label.read_text(), "1 0.500000 0.500000 0.100000 0.100000\n")

    def test_rerun_replaces_old_dataset(self):
        make_export(self.export, days=5)
        self.run_split()
        (self.out / "images" / "train" / "stale.jpg").write_bytes(b"old")
        self.run_split()
        self.assertFalse((self.out / "images" / "train" / "stale.jpg").exists())


class TestSpread(unittest.TestCase):
    def test_always_holds_something_out(self):
        self.assertEqual(len(prep.spread(list(range(5)), 0.15)), 1)
        self.assertEqual(prep.spread(list(range(14)), 0.15), [3, 10])


if __name__ == "__main__":
    unittest.main()
