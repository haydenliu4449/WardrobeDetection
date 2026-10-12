# Closet Mirror

A Raspberry Pi Zero 2 W takes a photo of the closet every morning, works out which clothes are actually in there, and shows a few outfit ideas on a mirror based on the day's weather.

## How it works

```
morning.py (systemd timer, 5:45 every day)
  capture.py        webcam photo of the closet, LED strip on for a moment first
  detect.py         closet photo + catalog contact sheet -> vision model API -> which items are there
   or yolo_detect.py  closet photo -> YOLO model trained on this closet, runs on the Pi
  weather.py        today's forecast from Open-Meteo
  outfits.py        score every valid combination, keep the best three
  -> data/today.json

display.py (always running)
  watches data/today.json and redraws the mirror when it changes
```

A few decisions worth explaining:

**Catalog first, then matching.** Recognizing clothes from scratch in a closet photo doesn't work well. Things overlap, a hanger might only show a sleeve, and folded stacks all look the same. So every item gets photographed once and added to `data/wardrobe.json`. In the morning the model gets the closet photo plus a contact sheet of the catalog (one image with every item labeled by id), and only has to answer "which of these are here?". Anything not found is assumed to be in the laundry.

**Two ways to detect.** `[detect] method` in `config.toml` picks one. Both return the same list of item ids, so nothing else in the project cares which one ran.

- `api` (the default): the Zero has 512MB of RAM, so a general vision model can't run locally. `detect.py` calls the Anthropic API with `requests` instead of the SDK to keep dependencies down, and uses a forced tool call so the answer always comes back as structured JSON. IDs the model returns that aren't in the catalog get dropped. It works from day one with no training data.
- `yolo`: a small YOLO detector trained on photos of this closet, with one class per item. That's practical here because the camera never moves: same angle, same light, a few dozen known items. Training happens on a PC and the model is exported to ONNX, so the Pi only needs onnxruntime (no PyTorch), and inference takes a few seconds. No API cost, and the photo never leaves the Pi. Clothes the model wasn't trained on yet (new purchases) fall back to the API, or are assumed present if there's no key. See [training/README.md](training/README.md).

**Outfits are picked with rules, not the model.** `outfits.py` builds every top+bottom (or dress) + shoes combination, adds a coat if it's cold or likely to rain, and scores each one on:
- color (neutrals go with anything; tonal, analogous and complementary pairs score well; three bold colors doesn't)
- warmth vs. the forecast (each item has a 1-5 warmth rating)
- formality, weekday vs. weekend
- no two patterned pieces
- avoiding pieces suggested in the last few days

The top three that don't share pieces win. It's quick on the Zero, predictable, and easy to test. A small random tiebreak seeded by the date keeps suggestions changing day to day while staying the same if it reruns.

**Pillow draws, pygame displays.** All layout happens in `render.py` with Pillow. `display.py` just pushes the finished image to the screen with pygame (KMS/DRM, no desktop needed). That means the screen can be tested anywhere with `--snapshot`, and the Zero only resizes photos once per outfit instead of every frame.

**Things fail gracefully.** If the camera or API fails, or the photo looks wrong (fewer than 3 items found usually means the door was shut or the light was off), it falls back to the last good inventory. If the forecast can't be fetched it assumes a mild day. The display shows a message instead of crashing.

## Hardware

- Raspberry Pi Zero 2 W
- USB webcam + micro-USB OTG adapter. It goes in the port labeled USB, not PWR.
- Monitor + mini-HDMI to HDMI cable
- Two-way mirror (glass or acrylic) in front of the monitor. OLED works best because black pixels are fully off, but any monitor works.
- 5V 2.5A power supply (the webcam draws power through the Pi)
- Optional: LED strip switched by a MOSFET on a GPIO pin, so the closet is lit for the photo
- Optional: push button on a GPIO pin to flip between outfits

Wi-Fi on the Zero is 2.4 GHz only, so check the signal where it'll be mounted.

## Setup on the Pi

Flash Raspberry Pi OS Lite (pick the 64-bit version if you plan to use YOLO), enable SSH, then copy this folder over and run setup:

```
scp -r "Personal Project" pi@mirror.local:closet-mirror
ssh pi@mirror.local
cd closet-mirror && ./setup.sh          # or ./setup.sh --yolo
```

(Replace `pi@mirror.local` with your username and hostname.) It installs everything from apt (pip would have to compile OpenCV and pygame, which takes hours on the Zero), creates `config.toml` and `.env`, and installs the systemd units. `--yolo` also installs onnxruntime. After that:

1. Put the API key in `.env`.
2. Set latitude/longitude in `config.toml`.
3. Add clothes (below).
4. Run `python3 capture.py` and check `data/captures/test.jpg` to aim the camera.
5. Run `python3 morning.py` once by hand and check the output.
6. Reboot.

To keep the console from showing through the mirror, add this to the end of the line in `/boot/firmware/cmdline.txt`:

```
consoleblank=0 vt.global_cursor_default=0
```

If the monitor is mounted on its side, set `rotate` in `config.toml` (90 or 270).

Logs: `journalctl -u morning-outfits` and `journalctl -u mirror-display`.

## Adding clothes

```
python3 closet.py add photo.jpg       # prompts for name, category, color, warmth, formality
python3 closet.py add photo.jpg --name "black ankle boots" --category shoes --color black --warmth 3 --formality 2
python3 closet.py list
python3 closet.py remove top-03
```

Tips:
- Photograph each item laid flat or hanging, against something dark. On the mirror a black background disappears and the clothes look like they're floating. White backgrounds show up as bright boxes.
- `--tags waterproof` on a jacket makes it preferred on rainy days. `--tags avoid_rain` (suede, etc.) gets penalized when it's wet.
- Colors are matched against the names in `colors.py`. Unknown names are treated as neutrals.
- Items without a photo still work. The mirror shows a color swatch instead.

## Trying it on a laptop

```
pip install -r requirements.txt
python3 morning.py --demo         # sample wardrobe, no camera or API
python3 display.py --windowed     # arrow keys / space to flip outfits, q to quit
python3 display.py --snapshot screen.png
```

Other flags: `morning.py --photo some_photo.jpg` runs detection on an existing photo, and `--no-detect` skips detection entirely.

## Tests

```
python3 -m unittest discover -s tests
```

These cover:
- outfit building (completeness, weather, rain, variety, repeats) and color scoring
- parsing the API's reply, and the fallbacks when detection fails
- YOLO output decoding, box coordinates, thresholds, and the API fallback for untrained items
- the dataset split (whole days held out, bad labels dropped)
- the duplicate-photo check in collection mode

## Files

| file | what it does |
|---|---|
| `morning.py` | the daily run, ties everything together |
| `capture.py` | webcam photo (OpenCV, falls back to fswebcam), closet light, training photo collection |
| `detect.py` | contact sheet + API call + parsing |
| `yolo_detect.py` | runs the trained ONNX model, API fallback for untrained items |
| `training/prepare_dataset.py` | label list for the labeling tool, train/val split |
| `training/train.py` | fine-tunes YOLO on a PC and exports to ONNX |
| `weather.py` | Open-Meteo forecast |
| `outfits.py` | builds and scores combinations |
| `colors.py` | color names, neutrals, color matching |
| `render.py` | draws the mirror screen |
| `display.py` | fullscreen loop, button, screen hours |
| `closet.py` | add/list/remove wardrobe items |
| `config.py` | settings and paths |

## Privacy

Photos stay on the Pi. With the API method, the only thing that leaves it is the morning closet photo, sent for detection. With YOLO, nothing leaves it unless an item needs the fallback. Old captures are deleted after two weeks. The camera only points into the closet and normally only runs once a day.

Collection mode is the exception: it takes a photo every hour from 7am to 10pm while it's turned on. Make sure everyone who uses the closet knows when it's running. Turn it off once you have enough photos, and delete the photos you don't need after labeling.

## Ideas for later

- Learn from which outfit actually gets chosen (the button press could log it) and weight the scoring toward her taste
- Fill in item details automatically from the photo when adding clothes
- Remove photo backgrounds on a laptop before copying them to the Pi
- Motion sensor so the screen only wakes up when someone's in front of it
