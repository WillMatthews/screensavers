#!/usr/bin/env bash
# Launch the cut-paper screensaver fullscreen. Extra args go to the hack
# (e.g. --palette dusk --scene tunnel --seed 42).
# Built from papercut.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/papercut" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Papercut"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Papercut"] fullscreen enable'

fi
wait "$pid"
