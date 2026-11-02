#!/usr/bin/env bash
# Launch the relaxing Voronoi screensaver fullscreen. Extra args go to the hack
# (e.g. --palette glass --cells 900 --seed 42).
# Built from voronoi.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/voronoi" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Voronoi"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Voronoi"] fullscreen enable'

fi
wait "$pid"
