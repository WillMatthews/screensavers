#!/usr/bin/env bash
# Launch the quantum screensaver fullscreen. Extra args go to the hack
# (e.g. --seed 42 --palette paper --act slit).
# Built from quantum.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/quantum" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Quantum"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Quantum"] fullscreen enable'

fi
wait "$pid"
