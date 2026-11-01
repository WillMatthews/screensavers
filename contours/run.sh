#!/usr/bin/env bash
# Launch the topographic-map screensaver fullscreen. Extra args go to the hack
# (e.g. --seed 42 --light --interval 25 --speed 2).
# Built from contours.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/contours" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Contours"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Contours"] fullscreen enable'

fi
wait "$pid"
