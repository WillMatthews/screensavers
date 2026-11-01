#!/usr/bin/env bash
# Launch the marbling screensaver fullscreen. Extra args go to the hack
# (e.g. --seed 42 --recipe whirl).
# Built from marbling.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/marbling" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Marbling"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Marbling"] fullscreen enable'

fi
wait "$pid"
