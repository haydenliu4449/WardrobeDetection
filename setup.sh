#!/usr/bin/env bash
# One-time setup on the Pi (Raspberry Pi OS Lite, Bookworm or newer).
# Run as your normal user, it calls sudo where it needs to.
set -euo pipefail

if [ "$EUID" -eq 0 ]; then
    echo "run this without sudo"
    exit 1
fi

cd "$(dirname "$0")"
DIR="$(pwd)"

# everything from apt, pip would have to compile opencv/pygame on the Zero
sudo apt-get update
sudo apt-get install -y python3-opencv python3-pygame python3-pil python3-requests \
    python3-gpiozero fswebcam fonts-dejavu-core

mkdir -p data/photos data/captures
[ -f config.toml ] || cp config.example.toml config.toml
[ -f .env ] || cp .env.example .env
chmod 600 .env

# webcam, screen and GPIO without root
sudo usermod -aG video,render,input,gpio "$USER"

for unit in systemd/*; do
    sed -e "s|__USER__|$USER|g" -e "s|__DIR__|$DIR|g" "$unit" \
        | sudo tee "/etc/systemd/system/$(basename "$unit")" > /dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable morning-outfits.timer mirror-display.service

cat <<EOF

Done. Next:
  1. put the API key in .env
  2. set latitude/longitude in config.toml
  3. add clothes:     python3 closet.py add photo.jpg
  4. aim the camera:  python3 capture.py   (then check data/captures/test.jpg)
  5. try a run:       python3 morning.py
  6. reboot - the display starts on its own and the timer runs every morning
EOF
