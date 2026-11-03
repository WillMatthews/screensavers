#!/usr/bin/env bash
# Lock-screen background for swaylock-plugin --command-each (one instance per
# output). Most savers draw their own layer surfaces; pipes and maze run on the
# root window of a private Xwayland, which windowtolayer turns into a layer
# surface behind the password ring.
# Usage: lock-bg.sh [SAVER] [seed] [epoch]   (default: random; SAVERS in savers.sh)
# With no args it reads "saver seed epoch" from $XDG_RUNTIME_DIR/lock-bg.cur,
# which lock.sh and lock-next.sh write; swaylock-plugin re-runs this same
# command whenever the saver dies, so that file is how a skip takes effect.
# Savers in SPANNING (savers.sh) treat all outputs as one wall: every instance
# needs the same seed and epoch, plus the sway layout and which output it's on.
# Only savers built on this host are run (savers.sh): asked for one that
# isn't, it runs a random built one instead.
dir="$(cd "$(dirname "$0")" && pwd)"
. "$dir/savers.sh"
cur=${XDG_RUNTIME_DIR:-/tmp}/lock-bg.cur
(($# == 0)) && [[ -r $cur ]] && read -r -a args < "$cur" && set -- "${args[@]}"
mapfile -t avail < <(built_savers "$dir")
if ((${#avail[@]} == 0)); then
  echo "lock-bg.sh: no screensaver is built here (e.g. make -C stir)" >&2
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
  *)  # Native layer-shell savers normally share a seed across outputs.
      cmd=("$(saver_bin "$dir" "$pick")" --layer)
      seed=${2:-}
      if [[ ( $pick == soapfilm || $pick == aquarium ) && -n ${SWAYLOCK_PLUGIN_OUTPUT_NAME:-} ]]; then
        # Full scene per monitor, with independent randomness. Keep the seed
        # stable if the plugin restarts this output during the same lock.
        read -r seed _ < <(printf '%s:%s' "${seed:-$SRANDOM}" "$SWAYLOCK_PLUGIN_OUTPUT_NAME" | cksum)
        seed=$((seed | 1))
      fi
      [[ -n $seed ]] && cmd+=(--seed "$seed")
      # savers that span outputs (savers.sh SPANNING) also get the shared start
      # time, the "x,y,w,h;..." of the active outputs and our index in it
      if [[ " ${SPANNING[*]} " == *" $pick "* ]]; then
        [[ -n ${3:-} ]] && cmd+=(--epoch "$3")
        if [[ -n ${SWAYLOCK_PLUGIN_OUTPUT_NAME:-} ]] &&
           read -r layout idx < <(swaymsg -t get_outputs -r 2>/dev/null | jq -r --arg me "$SWAYLOCK_PLUGIN_OUTPUT_NAME" '
             [.[] | select(.active)] as $o
             | ([$o[].rect | "\(.x),\(.y),\(.width),\(.height)"] | join(";")) + " "
             + ([$o[].name] | index($me) // -1 | tostring)') && ((idx >= 0)); then
          cmd+=(--layout "$layout" --output "$idx")
        fi
      fi ;;
esac
# Only these savers accept geographic coordinates. With auto mode disabled or
# unavailable, omit the flags so each saver reads the saved manual location.
case $pick in
  nightsky|atc|clocktower)
    if read -r lat lon < <(python3 "$dir/location.py" resolve); then
      cmd+=(--lat "$lat" --lon "$lon")
    fi ;;
esac
# pipes and maze are X11 hacks, run on the root window of a private Xwayland
# that windowtolayer turns into a layer surface (that path tears under load,
# hence everything else being native Wayland layer-shell clients)
if [[ $pick != pipes && $pick != maze ]]; then exec "${cmd[@]}"; fi
exec "$dir/build/windowtolayer/target/release/windowtolayer" \
  "$dir/build/swaylock-plugin/example_xwayland_wrapper.py" "${cmd[@]}"
