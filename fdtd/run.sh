#!/usr/bin/env bash
# Launch the FDTD (Maxwell solver) screensaver fullscreen. Extra args go to the hack
# (e.g. --act array --palette paper --poynting --seed 42).
# Built from fdtd.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/fdtd" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Fdtd"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Fdtd"] fullscreen enable'

fi
wait "$pid"
