"""Gets labeled closet photos into the layout YOLO trains on.

    python training/prepare_dataset.py classes
        writes training/classes.txt and training/label_studio.xml from the wardrobe,
        so every box gets labeled with an item id (top-01, shoes-03, ...)

    python training/prepare_dataset.py split path/to/export [--photos data/training_photos]
        takes a YOLO-format export (Label Studio, CVAT "Ultralytics YOLO", Roboflow)
        and builds training/dataset/ with a train/val split and closet.yaml.
        --photos is for exports that only contain the label files
"""

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WARDROBE = ROOT / "data" / "wardrobe.json"
DATASET = HERE / "dataset"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAME_FILES = ("classes.txt", "obj.names")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
FEW = 15  # fewer labeled boxes than this and the model won't learn the item well

LABEL_STUDIO = """<View>
  <Image name="image" value="$image" zoom="true"/>
  <RectangleLabels name="label" toName="image">
{labels}
  </RectangleLabels>
</View>
"""


def load_wardrobe(path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        sys.exit(f"no wardrobe at {path}, copy data/ over from the Pi first")


def write_classes(args):
    items = load_wardrobe(args.wardrobe)
    ids = [i["id"] for i in items]
    (HERE / "classes.txt").write_text("\n".join(ids) + "\n")
    labels = "\n".join(f'    <Label value="{i}"/>' for i in ids)
    (HERE / "label_studio.xml").write_text(LABEL_STUDIO.format(labels=labels))

    print("wrote training/classes.txt and training/label_studio.xml\n")
    for item in items:
        print(f"  {item['id']:<10} {item['name']}")


def read_names(export):
    for name in NAME_FILES:
        for path in export.rglob(name):
            return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    # Roboflow / CVAT Ultralytics exports put the names in data.yaml
    for path in export.rglob("*.yaml"):
        import yaml  # comes with ultralytics
        names = (yaml.safe_load(path.read_text()) or {}).get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names)]
        if isinstance(names, list):
            return names

    sys.exit("couldn't find the class names (classes.txt, obj.names or data.yaml) in the export")


def find_pairs(export, photos=None):
    """(image, label file) pairs matched by file name, wherever the export put them.
    `photos` is a folder of the original photos, for exports that only contain labels."""
    images = {p.stem: p for p in export.rglob("*") if p.suffix.lower() in IMAGE_EXT}
    exported = set(images)
    if photos:
        for p in Path(photos).iterdir():
            if p.suffix.lower() in IMAGE_EXT:
                images.setdefault(p.stem, p)
    labels = {p.stem: p for p in export.rglob("*.txt") if p.name not in NAME_FILES}

    pairs, no_image = [], 0
    for stem, label in sorted(labels.items()):
        # Label Studio puts a short id in front of uploaded files: 1a2b3c4d-2026-10-12_0900
        image = images.get(stem) or images.get(stem.split("-", 1)[-1])
        if image is None:
            no_image += 1
        else:
            pairs.append((image, label))
    unlabeled = len(exported - set(labels))
    return pairs, unlabeled, no_image


def read_boxes(label, num_classes):
    boxes, bad = [], 0
    for line in label.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            bad += 1
            continue
        # anything other than 4 numbers is a polygon, export as boxes instead
        if len(coords) != 4 or not 0 <= cls < num_classes or not all(0 <= v <= 1.0001 for v in coords):
            bad += 1
            continue
        boxes.append((cls, coords))
    return boxes, bad


def day_of(path):
    # collected photos are named 2026-10-12_0900.jpg, labeling tools usually keep the date somewhere in the name
    match = DATE.search(path.stem)
    return match.group(0) if match else path.stem


def spread(seq, fraction):
    """Roughly `fraction` of seq, picked evenly across it (at least one)."""
    n = max(1, round(len(seq) * fraction))
    return [seq[int((k + 0.5) * len(seq) / n)] for k in range(n)]


def choose_val(pairs, fraction):
    """Hold out whole days. Photos from the same day are nearly identical, so
    splitting them randomly would put copies of the same closet in both sets
    and make the validation score look better than it is."""
    by_day = defaultdict(list)
    for pair in pairs:
        by_day[day_of(pair[0])].append(pair)

    days = sorted(by_day)
    if len(days) >= 5:
        return {p[0] for d in spread(days, fraction) for p in by_day[d]}

    print(f"only {len(days)} different days in the export, splitting by photo instead")
    return {p[0] for p in spread(pairs, fraction)}


def split(args):
    export = Path(args.export).resolve()
    out = Path(args.out).resolve()
    names = read_names(export)
    pairs, unlabeled, no_image = find_pairs(export, args.photos)
    if no_image and not pairs:
        sys.exit("the export has labels but no images, add --photos data/training_photos")
    if len(pairs) < 2:
        sys.exit(f"found {len(pairs)} labeled images in {export}, need a lot more than that")

    if out.exists():
        if any(out.iterdir()) and not (out / "closet.yaml").exists():
            sys.exit(f"{out} has other files in it, pick an empty folder")
        shutil.rmtree(out)

    val = choose_val(pairs, args.val)
    counts = {"train": Counter(), "val": Counter()}
    bad_lines = 0
    for image, label in pairs:
        part = "val" if image in val else "train"
        boxes, bad = read_boxes(label, len(names))
        bad_lines += bad
        for sub in ("images", "labels"):
            (out / sub / part).mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, out / "images" / part / image.name)
        rows = [f"{cls} {' '.join(f'{v:.6f}' for v in coords)}" for cls, coords in boxes]
        (out / "labels" / part / f"{image.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        counts[part].update(names[cls] for cls, _ in boxes)

    yaml_lines = [f"path: {json.dumps(out.as_posix())}", "train: images/train", "val: images/val", "names:"]
    yaml_lines += [f"  {n}: {json.dumps(name)}" for n, name in enumerate(names)]
    (out / "closet.yaml").write_text("\n".join(yaml_lines) + "\n")
    # train.py reads this to know which items actually have enough examples
    (out / "classes.json").write_text(json.dumps(
        {"names": names, "train_counts": dict(counts["train"]), "val_counts": dict(counts["val"])}, indent=2))

    n_val = len(val)
    print(f"{len(pairs) - n_val} train / {n_val} val images -> {out}")
    if unlabeled:
        print(f"skipped {unlabeled} images with no label file")
    if no_image:
        print(f"skipped {no_image} label files with no matching image (try --photos data/training_photos)")
    if bad_lines:
        print(f"skipped {bad_lines} label lines that weren't valid boxes")

    print(f"\n{'item':<12}{'train':>7}{'val':>6}")
    for name in names:
        tr, va = counts["train"][name], counts["val"][name]
        note = "  <- needs more labels" if tr < FEW else ("  <- none in val" if va == 0 else "")
        print(f"{name:<12}{tr:>7}{va:>6}{note}")

    wardrobe_path = Path(args.wardrobe)
    if wardrobe_path.exists():
        wardrobe = {i["id"] for i in load_wardrobe(wardrobe_path)}
        stray = [n for n in names if n not in wardrobe]
        if stray:
            print(f"\nthese labels aren't in the wardrobe (typo, or removed item?): {', '.join(stray)}")


def main():
    parser = argparse.ArgumentParser(description="prepare the YOLO dataset")
    parser.add_argument("--wardrobe", default=WARDROBE)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("classes", help="write the label list for the labeling tool")
    p.set_defaults(func=write_classes)

    p = sub.add_parser("split", help="build training/dataset from a labeled export")
    p.add_argument("export")
    p.add_argument("--photos", help="original photos, if the export only has labels")
    p.add_argument("--out", default=DATASET)
    p.add_argument("--val", type=float, default=0.15, help="rough share of days held out for validation")
    p.set_defaults(func=split)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
