#!/usr/bin/env bash
# Launch the origami screensaver fullscreen. Extra args go to the hack
# (e.g. --palette washi --scene miura --seed 42).
# Built from origami.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/origami" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Origami"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Origami"] fullscreen enable'

fi
wait "$pid"
