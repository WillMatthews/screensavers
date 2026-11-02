#!/usr/bin/env bash
# Launch the network-stack diagram screensaver fullscreen. Extra args go to the hack
# (e.g. --palette amber --fps 30).
# Built from netflow.c (make).
# Press q to quit.
dir="$(cd "$(dirname "$0")" && pwd)"
"$dir/netflow" --window "$@" &
pid=$!
if command -v swaymsg >/dev/null; then
  # wait for the window to map, or sway re-tiles it after we fullscreen it
  for _ in $(seq 50); do
    swaymsg -t get_tree | grep -q '"class": "Netflow"' && break
    sleep 0.1
  done
  sleep 0.2
  swaymsg -q '[class="Netflow"] fullscreen enable'

fi
wait "$pid"
