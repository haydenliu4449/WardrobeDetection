"""Takes a photo of the closet.

Run it on its own to check the camera angle:
    python3 capture.py            -> data/captures/test.jpg
"""

import logging
import shutil
import subprocess
import time
from datetime import datetime

import config

log = logging.getLogger(__name__)


class Light:
    """LED strip on a GPIO pin (through a MOSFET or relay). Does nothing if pin is 0."""

    def __init__(self, pin, settle):
        self.dev = None
        self.settle = settle
        if pin:
            try:
                from gpiozero import OutputDevice
                self.dev = OutputDevice(pin)
            except Exception as e:  # not on a Pi, or gpiozero missing
                log.warning("closet light disabled: %s", e)

    def __enter__(self):
        if self.dev:
            self.dev.on()
            time.sleep(self.settle)
        return self

    def __exit__(self, *exc):
        if self.dev:
            self.dev.off()


def grab_opencv(cam_cfg, out_path):
    import cv2

    cap = cv2.VideoCapture(cam_cfg["device"], cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"can't open camera {cam_cfg['device']}")
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
        # cheap webcams take a second or so to settle exposure and white balance,
        # the first frames come out dark/blue
        for _ in range(cam_cfg["warmup_frames"]):
            cap.read()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera returned no frame")
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    finally:
        cap.release()


def grab_fswebcam(cam_cfg, out_path):
    device = cam_cfg["device"]
    if isinstance(device, int):
        device = f"/dev/video{device}"
    cmd = ["fswebcam", "-q", "-d", device,
           "-r", f"{cam_cfg['width']}x{cam_cfg['height']}",
           "-S", str(cam_cfg["warmup_frames"]), "--no-banner", "--jpeg", "90",
           str(out_path)]
    subprocess.run(cmd, check=True, timeout=60)


def take_photo(cfg, out_path=None):
    cam = cfg["camera"]
    if out_path is None:
        out_path = config.CAPTURES_DIR / f"{datetime.now():%Y-%m-%d_%H%M}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Light(cam["light_pin"], cam["light_settle"]):
        try:
            grab_opencv(cam, out_path)
        except ImportError:
            if not shutil.which("fswebcam"):
                raise RuntimeError("need either python3-opencv or fswebcam installed")
            grab_fswebcam(cam, out_path)

    log.info("saved %s", out_path)
    return out_path


def cleanup(keep=14):
    shots = sorted(config.CAPTURES_DIR.glob("20*.jpg"))
    for old in shots[:-keep]:
        old.unlink()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    take_photo(config.load(), config.CAPTURES_DIR / "test.jpg")
