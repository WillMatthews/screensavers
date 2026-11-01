#!/usr/bin/env bash
# Launch the stirred-letters screensaver fullscreen. Extra args go to the hack
# (e.g. --cell 28 --speed 1.5 --seed 42 --text story.txt).
# Built from stir.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/stir" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Stir"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Stir"] fullscreen enable'

fi
wait "$pid"
