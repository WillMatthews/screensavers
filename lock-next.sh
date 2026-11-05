#!/usr/bin/env bash
# Skip to the next lock screensaver (sway: bindsym --locked $mod+n). Writes the
# new pick for lock-bg.sh, then kills the running savers; swaylock-plugin
# restarts its command on every output and they pick up the new one.
# --reroll (bindsym --locked $mod+m) keeps the saver and rolls a new seed and
# start time instead: a new palette, scene or piece, and doesn't use a pick.
pgrep -x swaylock-plugin >/dev/null || exit 0
dir="$(cd "$(dirname "$0")" && pwd)"
cur=${XDG_RUNTIME_DIR:-/tmp}/lock-bg.cur
saver=
[[ ${1:-} == --reroll && -r $cur ]] && read -r saver _ < "$cur"
[[ -n $saver ]] || saver=$("$dir/pick-saver.sh")
echo "$saver $((SRANDOM | 1)) $(date +%s.%N)" > "$cur"
# anchored on the full path so this script's own command line can't match
pkill -f "^$dir/(pipes/bin/pipes|maze3d/bin/maze3d|(flightsim|stir|splitflap|gauges|marbling|contours|clockwork|nightsky|ink|soapfilm|orrery|lavalamp|atc|billiards|pendulums|voronoi|aquarium|netflow|kalman|clocktower|neural|bridge|currents|fdtd|quantum|sand|papercut|kluver|tiedye)/[a-z]+) --(root|layer)"
