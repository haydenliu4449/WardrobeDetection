# Training the YOLO model

This replaces the API call with a detector trained on photos of this closet. The model has one class per wardrobe item (`top-01`, `shoes-03`, ...), so it says directly which of her clothes are there.

Rough plan: collect photos on the Pi for a couple of weeks, label them on a PC, train on the PC, copy the model back to the Pi.

## 1. Collect photos (Pi)

```
sudo systemctl enable --now collect-photos.timer
```

This takes a photo every hour from 7am to 10pm into `data/training_photos/`. If the closet looks the same as the last kept photo, the new one is thrown away. Labeling the same closet over and over doesn't teach the model anything. The share of the photo that changed is logged each time (`journalctl -u collect-photos`). If it's keeping too many or too few, adjust `collect_min_change` in `config.toml`.

You can also grab one by hand at any point: `python3 capture.py --collect`.

Things that help:
- Let it run for 2-3 weeks of normal use, so clothes come and go and laundry days are in there.
- Every item should show up in a decent number of photos. An item that was never in the closet during collection can't be learned. It'll use the API fallback until it's in a later batch.
- A couple of hundred kept photos is a good first round.

Turn it off when you're done:

```
sudo systemctl disable --now collect-photos.timer
```

## 2. Copy everything to the PC

From the project folder on the PC:

```
scp -r pi@mirror.local:closet-mirror/data .
```

This brings over the photos and `data/wardrobe.json`, which the next step needs.

## 3. Label

```
python training/prepare_dataset.py classes
```

This writes `training/classes.txt` (one item id per line) and `training/label_studio.xml`, and prints which id is which item. `python closet.py list` shows the same list.

Label in [Label Studio](https://labelstud.io), which runs locally so the photos stay on your machines (`pip install label-studio`, then `label-studio start`):
1. Create a project and import `data/training_photos`.
2. Under Settings > Labeling Interface > Code, paste in `label_studio.xml`.
3. Draw a box around every item you can see and pick its id. If only a sleeve is visible on a hanger, box the sleeve. That's what the camera will see in the morning too.
4. Export in YOLO format (the version with images, if yours offers both).

CVAT (export as "Ultralytics YOLO Detection") and Roboflow (export as YOLOv8) also work, but Roboflow uploads the photos to their servers.

## 4. Build the dataset

```
python training/prepare_dataset.py split path/to/export
```

If the export only has the label files and no images, add `--photos data/training_photos` and it matches them up with the originals.

This copies the export into `training/dataset/` with a train/val split and writes `closet.yaml`. Validation holds out whole days rather than random photos. Photos from the same day are nearly identical, and splitting them randomly would make the score look better than it really is.

It also prints how many boxes each item got. Items marked "needs more labels" are worth photographing and labeling more of.

## 5. Train (PC)

Install PyTorch first, using the command for your setup from https://pytorch.org/get-started/locally/. If you just `pip install ultralytics`, you get the CPU-only build on Windows. Then:

```
pip install -r training/requirements.txt
python training/train.py
```

It starts from the pretrained `yolo26n.pt` (nano, the smallest) and fine-tunes on the closet photos. With an NVIDIA GPU this takes minutes; on a CPU, hours. If you don't have a GPU, Google Colab's free one works: upload the project folder and run the same two commands.

Color augmentation is turned down on purpose. A lot of items are only told apart by color (white vs. cream shirt, navy vs. black pants), and the default hue jitter would teach the model to ignore exactly that.

At the end it prints mAP50 on the held-out days and the weakest items, and writes:
- `models/closet.onnx`: the model
- `models/closet.json`: class names, plus which items had enough examples to count as trained (`--min-boxes`, default 10)

Training runs, plots and confusion matrices end up in `training/runs/`. They're useful for the presentation.

## 6. Use it (Pi)

```
scp -r models pi@mirror.local:closet-mirror/
```

On the Pi:

```
./setup.sh --yolo        # once, installs onnxruntime (needs the 64-bit OS)
python3 yolo_detect.py data/captures/test.jpg --save boxes.jpg
```

That prints a score for every item and saves the photo with boxes drawn on it. Use it to check the model and tune the thresholds. Then switch it on in `config.toml`:

```
[detect]
method = "yolo"

[yolo]
present = 0.5     # counts as there
unsure = 0.25     # maybe, still gets suggested
api_fallback = true
```

## When she buys new clothes

Add them with `closet.py` as usual. Until the model is retrained they're not in it, so `api_fallback` asks the API about just those items. Without an API key they're assumed to be there. Retrain whenever a few new things have piled up: collect, label the new photos, rerun steps 4-6.
