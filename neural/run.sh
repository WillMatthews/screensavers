#!/usr/bin/env bash
# Launch the neural-network screensaver fullscreen. Extra args go to the hack
# (e.g. --seed 42 --palette paper --act 2).
# Built from neural.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/neural" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Neural"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Neural"] fullscreen enable'

fi
wait "$pid"
