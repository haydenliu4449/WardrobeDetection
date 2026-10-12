"""Fine-tunes a small YOLO model on the closet photos and exports it for the Pi.

Run this on a PC, not the Pi:
    python training/train.py
    python training/train.py --epochs 150 --model yolo26s.pt

Writes models/closet.onnx and models/closet.json. Copy the models folder to the Pi.
"""

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = ROOT / "models"


def evaluate(model, data, imgsz, names):
    """mAP50 overall and per item on the held-out days. Only used for the printout,
    so a change in the ultralytics API shouldn't throw away an hour of training."""
    try:
        metrics = model.val(data=str(data), imgsz=imgsz, plots=False)
        per_item = {names[int(c)]: float(ap) for c, ap in zip(metrics.box.ap_class_index, metrics.box.ap50)}
        return round(float(metrics.box.map50), 3), per_item
    except Exception as e:
        print(f"couldn't evaluate the model ({e}), the export is still fine")
        return None, {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=HERE / "dataset" / "closet.yaml")
    parser.add_argument("--model", default="yolo26n.pt", help="pretrained weights to start from")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="lower it if you run out of GPU memory")
    parser.add_argument("--min-boxes", type=int, default=10,
                        help="items with fewer labeled boxes than this are left to the API fallback")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        print("no CUDA GPU found, this will be slow on the CPU (see training/README.md)\n")

    data = Path(args.data).resolve()
    try:
        info = json.loads((data.parent / "classes.json").read_text())
    except FileNotFoundError:
        raise SystemExit("no dataset yet, run training/prepare_dataset.py split first")
    names = info["names"]

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=30,
        project=str(HERE / "runs"),
        name="closet",
        # a lot of her clothes are only told apart by color (white vs cream shirt,
        # navy vs black pants), so keep the color augmentation gentle
        hsv_h=0.0,
        hsv_s=0.3,
    )
    best = Path(model.trainer.best)
    trained = YOLO(best)

    # default export is the plain (4 + classes, anchors) output, yolo_detect.py decodes it
    exported = trained.export(format="onnx", imgsz=args.imgsz)
    exported = Path(exported) if exported else best.with_suffix(".onnx")
    MODELS.mkdir(exist_ok=True)
    shutil.copy2(exported, MODELS / "closet.onnx")

    map50, ap50 = evaluate(trained, data, args.imgsz, names)
    counts = info["train_counts"]
    covered = [n for n in names if counts.get(n, 0) >= args.min_boxes]
    meta = {
        "names": names,
        "trained": covered,
        "imgsz": args.imgsz,
        "base_model": args.model,
        "trained_on": date.today().isoformat(),
        "map50": map50,
    }
    (MODELS / "closet.json").write_text(json.dumps(meta, indent=2))

    if map50 is not None:
        print(f"\nmAP50 on held-out days: {map50:.2f}")
    weak = sorted((ap, n) for n, ap in ap50.items() if ap < 0.7)
    if weak:
        print("weakest items, label more photos of these:")
        for ap, n in weak:
            print(f"  {n:<12} {ap:.2f}")
    left_out = [n for n in names if n not in covered]
    if left_out:
        print(f"not enough examples yet, the Pi will use the API for: {', '.join(left_out)}")
    print(f"\nsaved {MODELS / 'closet.onnx'} and closet.json, copy the models folder to the Pi")


if __name__ == "__main__":  # the guard matters on Windows, the data loader starts worker processes
    main()
