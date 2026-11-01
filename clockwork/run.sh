#!/usr/bin/env bash
# Launch the clockwork screensaver fullscreen. Extra args go to the hack
# (e.g. --seed 42 --scale 0.8).
# Built from clockwork.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/clockwork" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Clockwork"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Clockwork"] fullscreen enable'

fi
wait "$pid"
