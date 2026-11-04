#!/usr/bin/env bash
# Launch the Klüver form-constant screensaver fullscreen. Extra args go to the
# hack (e.g. --palette jewel --form lattice --seed 42).
# Built from kluver.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/kluver" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Kluver"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Kluver"] fullscreen enable'

fi
wait "$pid"
