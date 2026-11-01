#!/usr/bin/env bash
# Launch the rain-on-the-window screensaver fullscreen. Extra args go to the hack
# (e.g. --weather storm --lat 51.48 --lon 0.0).
# Built from rain.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/rain" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Rain"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Rain"] fullscreen enable'

fi
wait "$pid"
