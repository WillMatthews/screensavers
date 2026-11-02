#!/usr/bin/env bash
# Launch the sea-cliff library screensaver fullscreen. Extra args go to the hack
# (e.g. --seed 42 --hour 17.5 --age 2).
# Built from library.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/library" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Library"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Library"] fullscreen enable'

fi
wait "$pid"
