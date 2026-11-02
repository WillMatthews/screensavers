# Sourced by lock-bg.sh and pick-saver.sh: the lock screensavers, and which
# of them this host can actually run. Binaries are built per host and aren't
# in git, so a saver nobody has built here yet must never be picked: a pick
# that can't start leaves swaylock-plugin re-running it over a blank screen.

SAVERS=(pipes maze flightsim stir gauges contours clockwork nightsky soapfilm aquarium orrery atc voronoi pendulums)

# kept and buildable, but out of the rotation while they're being worked on
BENCHED=(marbling ink lavalamp billiards algorithms netflow splitflap kalman clocktower neural)

# savers that take --epoch/--layout/--output and treat all outputs as one wall
SPANNING=(flightsim stir nightsky marbling splitflap gauges clockwork ink soapfilm orrery lavalamp atc billiards pendulums algorithms voronoi aquarium netflow kalman clocktower neural)

# the executable for a saver
saver_bin() {
    case $2 in
        pipes)     echo "$1/pipes/bin/pipes" ;;
        maze)      echo "$1/maze3d/bin/maze3d" ;;
        *)         echo "$1/$2/$2" ;;  # the native ones: <name>/<name>
    esac
}

# print the savers runnable from wallpapers dir $1, one per line
built_savers() {
    local s
    for s in "${SAVERS[@]}"; do
        case $s in
            pipes|maze)     # X11 hacks: also need the Xwayland-to-layer plumbing
                [[ -x $1/build/windowtolayer/target/release/windowtolayer &&
                   -f $1/build/swaylock-plugin/example_xwayland_wrapper.py ]] || continue ;;
        esac
        [[ -x $(saver_bin "$1" "$s") ]] && echo "$s"
    done
}
