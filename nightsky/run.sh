#!/usr/bin/env bash
# Launch the night-sky screensaver fullscreen. Extra args go to the hack
# (e.g. --lines --timelapse 300 --lat 55.9 --lon -3.2 --time "2026-12-21 22:00").
# Built from nightsky.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/nightsky" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Nightsky"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Nightsky"] fullscreen enable'

fi
wait "$pid"
