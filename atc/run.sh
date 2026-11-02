#!/usr/bin/env bash
# Launch the air traffic control screensaver fullscreen. Extra args go to the
# hack (e.g. --sim --seed 42, or --lat 51.47 --lon -0.45 for live traffic).
# Built from atc.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/atc" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Atc"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Atc"] fullscreen enable'

fi
wait "$pid"
