import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import detect
import morning

ITEMS = [{"id": f"top-0{n}", "name": "shirt", "category": "top", "color": "white"} for n in range(1, 6)]


def api_reply(present, unsure=(), unknown=()):
    return {"content": [
        {"type": "text", "text": "Here's what I found."},
        {"type": "tool_use", "name": "report_closet",
         "input": {"present": list(present), "unsure": list(unsure), "unknown": list(unknown)}},
    ]}


class TestParse(unittest.TestCase):
    def test_drops_ids_not_in_catalog(self):
        result = detect.parse(api_reply(["top-01", "top-99"], ["shoes-07"], ["grey hoodie"]), ITEMS)
        self.assertEqual(result["present"], ["top-01"])
        self.assertEqual(result["unsure"], [])
        self.assertEqual(result["unknown"], ["grey hoodie"])

    def test_missing_tool_block(self):
        with self.assertRaises(RuntimeError):
            detect.parse({"content": [{"type": "text", "text": "hmm"}]}, ITEMS)


class TestCheckCloset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.inventory = Path(self.tmp.name) / "last_inventory.json"
        patcher = mock.patch.object(morning, "LAST_INVENTORY", self.inventory)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def check(self, result=None, error=None):
        with mock.patch.object(morning.detect, "detect", side_effect=error, return_value=result):
            available, info = morning.check_closet(ITEMS, {}, photo="closet.jpg")
        return [i["id"] for i in available], info

    def test_unsure_counts_as_present(self):
        ids, info = self.check({"present": ["top-01", "top-02"], "unsure": ["top-03"], "unknown": []})
        self.assertEqual(ids, ["top-01", "top-02", "top-03"])
        self.assertEqual(info["missing"], ["top-04", "top-05"])

    def test_api_failure_uses_last_inventory(self):
        self.inventory.write_text(json.dumps({"date": "2026-10-11", "ids": ["top-02", "top-04"]}))
        ids, info = self.check(error=RuntimeError("API error 500"))
        self.assertEqual(ids, ["top-02", "top-04"])
        self.assertEqual(info["stale_inventory"], "2026-10-11")

    def test_api_failure_with_no_history_assumes_everything(self):
        ids, _ = self.check(error=RuntimeError("no network"))
        self.assertEqual(len(ids), len(ITEMS))

    def test_suspiciously_empty_photo_is_ignored(self):
        self.inventory.write_text(json.dumps({"date": "2026-10-11", "ids": ["top-01", "top-02", "top-03"]}))
        ids, _ = self.check({"present": ["top-05"], "unsure": [], "unknown": []})
        self.assertEqual(ids, ["top-01", "top-02", "top-03"])


if __name__ == "__main__":
    unittest.main()
