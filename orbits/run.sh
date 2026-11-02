#!/usr/bin/env bash
# Launch the mission-control trajectory display fullscreen. Extra args go to the hack
# (e.g. --seed 42 --palette paper --act 1).
# Built from orbits.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/orbits" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Orbits"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Orbits"] fullscreen enable'

fi
wait "$pid"
