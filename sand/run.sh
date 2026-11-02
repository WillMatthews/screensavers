#!/usr/bin/env bash
# Launch the falling-sand world screensaver fullscreen. Extra args go to the
# hack (e.g. --seed 42 --hour 21 --cell 5).
# Built from sand.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/sand" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Sand"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Sand"] fullscreen enable'

fi
wait "$pid"
