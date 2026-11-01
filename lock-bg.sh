#!/usr/bin/env bash
# Lock-screen background for swaylock-plugin --command-each (one instance per
# output). flightsim and stir draw their own layer surfaces; pipes and maze run on the
# root window of a private Xwayland, which windowtolayer turns into a layer
# surface behind the password ring.
# Usage: lock-bg.sh [pipes|maze|flightsim|stir] [seed] [epoch]   (default: random)
# With no args it reads "saver seed epoch" from $XDG_RUNTIME_DIR/lock-bg.cur,
# which lock.sh and lock-next.sh write; swaylock-plugin re-runs this same
# command whenever the saver dies, so that file is how a skip takes effect.
# flightsim spans all outputs as one scene: every instance needs the same seed
# and epoch, plus the sway layout and which output it's on.
# Only savers built on this host are run (savers.sh): asked for one that
# isn't, it runs a random built one instead.
dir="$(cd "$(dirname "$0")" && pwd)"
. "$dir/savers.sh"
cur=${XDG_RUNTIME_DIR:-/tmp}/lock-bg.cur
(($# == 0)) && [[ -r $cur ]] && read -r -a args < "$cur" && set -- "${args[@]}"
mapfile -t avail < <(built_savers "$dir")
if ((${#avail[@]} == 0)); then
  echo "lock-bg.sh: no screensaver is built here (make -C flightsim, make -C stir)" >&2
  exit 1
fi
pick=${1:-}
if [[ " ${avail[*]} " != *" $pick "* ]]; then
  [[ -n $pick ]] && echo "lock-bg.sh: '$pick' isn't built here, running another" >&2
  pick=$(shuf -n1 -e "${avail[@]}")
fi
case $pick in
  pipes)     cmd=("$dir/pipes/bin/pipes" --root --delay 5000) ;;
  maze)      cmd=("$dir/maze3d/bin/maze3d" --root) ;;
  stir)      cmd=("$dir/stir/stir" --layer)
             [[ -n ${2:-} ]] && cmd+=(--seed "$2") ;;
  flightsim)
    cmd=("$dir/flightsim/flightsim" --layer)
    [[ -n ${2:-} ]] && cmd+=(--seed "$2")
    [[ -n ${3:-} ]] && cmd+=(--epoch "$3")
    # "x,y,w,h;..." of the active outputs, and our index in it
    if [[ -n ${SWAYLOCK_PLUGIN_OUTPUT_NAME:-} ]] &&
       read -r layout idx < <(swaymsg -t get_outputs -r 2>/dev/null | jq -r --arg me "$SWAYLOCK_PLUGIN_OUTPUT_NAME" '
         [.[] | select(.active)] as $o
         | ([$o[].rect | "\(.x),\(.y),\(.width),\(.height)"] | join(";")) + " "
         + ([$o[].name] | index($me) // -1 | tostring)') && ((idx >= 0)); then
      cmd+=(--layout "$layout" --output "$idx")
    fi ;;
  *) echo "lock-bg.sh: unknown screensaver '$pick'" >&2; exit 1 ;;
esac
# flightsim and stir are native Wayland layer-shell clients; the others are X11
# hacks, run on the root window of a private Xwayland that windowtolayer turns
# into a layer surface (that path tears under load, hence flightsim's own)
if [[ $pick == flightsim || $pick == stir ]]; then exec "${cmd[@]}"; fi
exec "$dir/build/windowtolayer/target/release/windowtolayer" \
  "$dir/build/swaylock-plugin/example_xwayland_wrapper.py" "${cmd[@]}"
