#!/usr/bin/env bash
# Print the next lock screensaver from a shuffle bag: every saver once per
# cycle in random order, and a new cycle never opens with the one just shown
# (plain shuf streaked badly). Logs each pick to lock-saver.log.
# Only savers built on this host go in the bag (savers.sh).
dir="$(cd "$(dirname "$0")" && pwd)"
. "$dir/savers.sh"
mapfile -t avail < <(built_savers "$dir")
if ((${#avail[@]} == 0)); then
    echo "pick-saver.sh: no screensaver is built here" >&2
    exit 1
fi
STATE=${XDG_STATE_HOME:-$HOME/.local/state}
BAG=$STATE/lock-saver-bag
# Serialize the whole read/pick/write transaction: overlapping skip or lock
# requests must consume different entries, including when refilling the bag.
mkdir -p "$STATE" || exit 1
exec 9> "$BAG.lock" || exit 1
flock -x 9 || exit 1
read -r -a held 2>/dev/null < "$BAG" || held=()
bag=()
for s in "${held[@]}"; do [[ " ${avail[*]} " == *" $s "* ]] && bag+=("$s"); done
if ((${#bag[@]} == 0)); then
    last=$(tail -n1 "$STATE/lock-saver.log" 2>/dev/null | awk '{print $NF}')
    mapfile -t bag < <(shuf -e "${avail[@]}")
    [[ ${bag[0]} == "$last" ]] && bag=("${bag[@]:1}" "${bag[0]}")
fi
echo "${bag[@]:1}" > "$BAG" || exit 1
echo "$(date -Is) ${bag[0]}" >> "$STATE/lock-saver.log" || exit 1
echo "${bag[0]}"
