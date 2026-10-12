"""Shows today's outfits on the mirror.

    python3 display.py                     fullscreen, what the systemd service runs
    python3 display.py --windowed          for trying it out on a laptop
    python3 display.py --snapshot out.png  render one frame to a file and exit

Arrow keys / space flip between outfits.
"""

import argparse
import json
import logging
import threading
import time
from datetime import date, datetime
from pathlib import Path

import closet
import config
import render

log = logging.getLogger(__name__)


def load_state():
    try:
        with open(config.STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def state_mtime():
    try:
        return config.STATE_FILE.stat().st_mtime
    except FileNotFoundError:
        return None


def in_hours(disp, now):
    start = datetime.strptime(disp["show_from"], "%H:%M").time()
    end = datetime.strptime(disp["show_until"], "%H:%M").time()
    t = now.time()
    if start <= end:
        return start <= t < end
    return t >= start or t < end  # window that crosses midnight


def make_frame(state, index, size, disp):
    if state is None:
        return render.render_message(size, "No outfits yet", "run morning.py to pick some")
    if state["date"] != date.today().isoformat():
        return render.render_message(size, "Still picking today's outfits")
    if not state.get("outfits"):
        return render.render_message(size, "Couldn't put together an outfit today",
                                     state.get("error", ""))
    wardrobe = Path(state.get("wardrobe", config.WARDROBE_FILE))
    items = {i["id"]: i for i in closet.load(wardrobe)}
    return render.render(state, index % len(state["outfits"]), size, disp, items)


class Button:
    def __init__(self, pin):
        self.pressed_flag = threading.Event()
        self.btn = None  # keep a reference or gpiozero releases the pin
        if not pin:
            return
        try:
            from gpiozero import Button as GpioButton
            self.btn = GpioButton(pin, bounce_time=0.05)
            self.btn.when_pressed = self.pressed_flag.set
        except Exception as e:
            log.warning("button disabled: %s", e)

    def pressed(self):
        if self.pressed_flag.is_set():
            self.pressed_flag.clear()
            return True
        return False


def to_surface(pygame, img):
    # python3-pygame on Bookworm is 2.1.2, which only has fromstring
    loader = getattr(pygame.image, "frombytes", None) or pygame.image.fromstring
    return loader(img.tobytes(), img.size, "RGB")


def run(cfg, windowed=False):
    import pygame

    disp = cfg["display"]
    rotation = disp["rotate"] % 360

    pygame.display.init()
    if windowed:
        scale = 0.45
        screen = pygame.display.set_mode((int(disp["width"] * scale), int(disp["height"] * scale)))
        pygame.display.set_caption("mirror")
        rotation = 0
    else:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.mouse.set_visible(False)

    out_w, out_h = screen.get_size()
    size = (out_h, out_w) if rotation in (90, 270) else (out_w, out_h)
    log.info("screen %dx%d, drawing at %dx%d", out_w, out_h, *size)

    button = Button(disp["button_pin"])
    clock = pygame.time.Clock()
    state, mtime = None, object()
    index, last_flip = 0, time.monotonic()
    frames, shown = {}, None

    while True:
        step = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return
                if event.key in (pygame.K_RIGHT, pygame.K_SPACE):
                    step = 1
                elif event.key == pygame.K_LEFT:
                    step = -1
        if button.pressed():
            step = 1

        m = state_mtime()
        if m != mtime:
            state, mtime = load_state(), m
            index, frames = 0, {}

        count = len(state.get("outfits", [])) if state else 0
        if not step and disp["cycle_seconds"] and time.monotonic() - last_flip > disp["cycle_seconds"]:
            step = 1
        if step and count:
            index = (index + step) % count
            last_flip = time.monotonic()

        visible = windowed or in_hours(disp, datetime.now())
        key = (mtime, index, visible, date.today())
        if key != shown:
            screen.fill((0, 0, 0))
            if visible:
                if key not in frames:
                    img = make_frame(state, index, size, disp)
                    if rotation:
                        img = img.rotate(rotation, expand=True)
                    frames[key] = to_surface(pygame, img)
                screen.blit(frames[key], (0, 0))
            pygame.display.flip()
            shown = key

        clock.tick(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--snapshot", metavar="PNG")
    parser.add_argument("--outfit", type=int, default=1, help="which outfit to snapshot")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = config.load()

    if args.snapshot:
        disp = cfg["display"]
        img = make_frame(load_state(), args.outfit - 1, (disp["width"], disp["height"]), disp)
        img.save(args.snapshot)
        print(f"saved {args.snapshot}")
        return

    run(cfg, windowed=args.windowed)


if __name__ == "__main__":
    main()
