"""Finds wardrobe items in the closet photo with the model from training/train.py.

Returns the same thing as detect.py, so morning.py doesn't care which one runs
([detect] method in config.toml). Runs with onnxruntime, no PyTorch on the Pi.

    python3 yolo_detect.py data/captures/test.jpg                  score for every item
    python3 yolo_detect.py data/captures/test.jpg --save boxes.jpg draw what it found
"""

import argparse
import ast
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)

PAD = 114  # grey ultralytics pads with during training, so use the same


class Model:
    def __init__(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = config.ROOT / path
        if not path.exists():
            raise RuntimeError(f"no model at {path}, see training/README.md")

        import onnxruntime as ort
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        meta = load_meta(path, self.session)
        self.names = meta["names"]
        # items the model saw enough of during training, the rest go to the fallback
        self.trained = set(meta.get("trained", self.names))
        size = inp.shape[2]
        self.size = size if isinstance(size, int) else meta.get("imgsz", 640)

    def run(self, img):
        x, scale, pad = letterbox(img, self.size)
        blob = x[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255  # BGR->RGB, HWC->NCHW
        out = self.session.run(None, {self.input_name: np.ascontiguousarray(blob)})[0]
        return decode(out, len(self.names), scale, pad)


def load_meta(path, session):
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    # no closet.json, fall back to the names ultralytics writes into the ONNX file
    # (stored as a python dict literal like "{0: 'top-01', 1: 'top-02'}")
    raw = session.get_modelmeta().custom_metadata_map.get("names")
    if not raw:
        raise RuntimeError("model has no class names, export it with training/train.py")
    names = ast.literal_eval(raw)
    return {"names": [names[i] for i in sorted(names)]}


def letterbox(img, size):
    """Resize to fit a size x size square without stretching, pad the rest."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = round(w * scale), round(h * scale)
    left, top = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), PAD, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, scale, (left, top)


def decode(out, num_classes, scale, pad):
    """Raw model output -> boxes (x1, y1, x2, y2 in photo pixels), scores, class indices."""
    out = out[0]
    if out.shape[0] == 4 + num_classes:
        # (4 + classes, anchors): one candidate box per anchor, centre x/y + w/h then a score per class
        out = out.T
        classes = out[:, 4:].argmax(1)
        scores = out[np.arange(len(out)), 4 + classes]
        xy, wh = out[:, :2], out[:, 2:4]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1)
    elif out.ndim == 2 and out.shape[1] == 6:
        # (detections, 6): models exported with the NMS-free head, already x1 y1 x2 y2 score class
        boxes, scores, classes = out[:, :4], out[:, 4], out[:, 5].astype(int)
    else:
        raise RuntimeError(f"unexpected model output shape {out.shape} for {num_classes} classes")

    left, top = pad
    boxes = (boxes - [left, top, left, top]) / scale
    return boxes, scores, classes


def item_scores(scores, classes, names):
    """Best score per item. For deciding what's in the closet the boxes don't
    matter, only whether each item shows up anywhere, so no NMS needed here."""
    best = np.zeros(len(names))
    np.maximum.at(best, classes, scores)
    return dict(zip(names, best.tolist()))


def cover_untrained(result, untrained, photo, items, cfg):
    """Items the model doesn't know yet (new clothes, or too few examples)."""
    if cfg["yolo"]["api_fallback"] and os.environ.get("ANTHROPIC_API_KEY"):
        import detect
        try:
            # the full catalog, not just the new items, so the model can tell them apart from the rest
            api = detect.detect(photo, items, cfg)
        except Exception as e:
            log.warning("API fallback failed (%s)", e)
        else:
            result["present"] += [i for i in api["present"] if i in untrained]
            result["unsure"] += [i for i in api["unsure"] if i in untrained]
            result["unknown"] = api["unknown"]
            return result

    # no way to check them, assume they're there so they still get suggested
    result["unsure"] += untrained
    return result


def detect(photo, items, cfg):
    opts = cfg["yolo"]
    model = Model(opts["model"])
    img = cv2.imread(str(photo))
    if img is None:
        raise RuntimeError(f"can't read {photo}")

    _, scores, classes = model.run(img)
    best = item_scores(scores, classes, model.names)
    result = {"present": [], "unsure": [], "unknown": []}
    untrained = []
    for item in items:
        if item["id"] not in model.trained:
            untrained.append(item["id"])
        elif best.get(item["id"], 0) >= opts["present"]:
            result["present"].append(item["id"])
        elif best.get(item["id"], 0) >= opts["unsure"]:
            result["unsure"].append(item["id"])

    if untrained:
        log.info("%d items aren't in the model yet: %s", len(untrained), ", ".join(untrained))
        cover_untrained(result, untrained, photo, items, cfg)

    log.info("yolo: %d present, %d unsure", len(result["present"]), len(result["unsure"]))
    return result


def draw(img, boxes, scores, classes, names, labels, min_score):
    """Boxes with item names on the photo. Overlapping boxes for the same item
    get merged here (NMS) since this is the one place the boxes are looked at."""
    keep = scores >= min_score
    for c in np.unique(classes[keep]):
        idx = np.flatnonzero(keep & (classes == c))
        rects = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in boxes[idx]]
        for j in np.array(cv2.dnn.NMSBoxes(rects, scores[idx].tolist(), min_score, 0.5)).flatten():
            x1, y1, x2, y2 = boxes[idx[j]].astype(int)
            text = f"{labels.get(names[c], names[c])} {scores[idx[j]]:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (80, 220, 80), 3)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 8, y1), (80, 220, 80), -1)
            cv2.putText(img, text, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


def main():
    import closet

    parser = argparse.ArgumentParser()
    parser.add_argument("photo")
    parser.add_argument("--save", metavar="JPG", help="write the photo with boxes drawn on it")
    args = parser.parse_args()

    cfg = config.load()
    opts = cfg["yolo"]
    model = Model(opts["model"])
    img = cv2.imread(args.photo)
    if img is None:
        sys.exit(f"can't read {args.photo}")

    boxes, scores, classes = model.run(img)
    best = item_scores(scores, classes, model.names)
    labels = {i["id"]: i["name"] for i in closet.load()}

    for name, score in sorted(best.items(), key=lambda kv: -kv[1]):
        verdict = "present" if score >= opts["present"] else "unsure" if score >= opts["unsure"] else ""
        if name not in model.trained:
            verdict = "too few examples, uses the fallback"
        print(f"  {name:<10} {labels.get(name, '?'):<32} {score:.2f}  {verdict}")
    for name in labels:
        if name not in best:
            print(f"  {name:<10} {labels[name]:<32}   -   not in the model yet")

    if args.save:
        cv2.imwrite(args.save, draw(img, boxes, scores, classes, model.names, labels, opts["unsure"]))
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
