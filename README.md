# wallpapers

Lock-screen savers for sway. Each one is a small, self-contained C program
with OpenGL 3.3 shaders that draws straight onto the lock screen behind the
swaylock password ring. Most show something live about the machine (the
time, how long it's been locked, CPU load and temperature, memory, battery,
the hostname) folded into the scene rather than pasted over it. Every saver
can also span several monitors as one picture.

![nightsky across three monitors](docs/img/wall-nightsky.jpg)
<sub>`nightsky` on a three-monitor wall (portrait, laptop, landscape): one panorama, one horizon.</sub>

## How it fits together

```
swaylock-plugin --command-each lock-bg.sh      one instance per output
        │
        ├─ lock-bg.sh         picks the saver (or reads $XDG_RUNTIME_DIR/lock-bg.cur),
        │                     passes --seed / --epoch / --layout / --output, execs it
        ├─ savers.sh          which savers exist, which are benched, which span outputs
        ├─ pick-saver.sh      shuffle bag: every saver once per cycle, no repeats at the seam
        └─ lock-next.sh       skip to the next saver while locked (bind to $mod+n --locked)
```

`savers.sh` holds three lists:

| List | Meaning |
|---|---|
| `SAVERS` | the lock-screen rotation |
| `BENCHED` | built and runnable, but kept out of the rotation while being worked on |
| `SPANNING` | savers that take `--epoch --layout --output` and treat all outputs as one wall |

Binaries aren't committed; each host builds what it wants (`make -C <saver>`).
`lock-bg.sh` only ever picks savers that are built on the current host, so an
unbuilt one can never leave the lock screen blank.

### Multi-monitor

swaylock-plugin starts one saver process per output, and they can't talk to
each other. For savers in `SPANNING`, `lock-bg.sh` gives every instance the
same seed and start time (`--epoch`), the logical rectangle of every active
output (`--layout 'x,y,w,h;...'`, from `swaymsg -t get_outputs`) and which one
it is (`--output N`). Each saver then computes the same world from those
alone and draws its own slice of it: one continuous scene, or a different
role per screen. The output under the middle of the layout is the "front"
one and carries any clock or caption.

![splitflap: platform, departures and arrivals boards](docs/img/wall-splitflap.jpg)
<sub>`splitflap` gives each screen a different board of the same station, all flipping together.</sub>

![gauges: core bank, main cluster, auxiliaries](docs/img/wall-gauges.jpg)
<sub>`gauges` puts the main cluster in the middle and extra instruments either side.</sub>

## The savers

### In the rotation

| | |
|---|---|
| ![stir](docs/img/stir.jpg) | **stir**: away messages ("Will is away from this computer…") and live details on a grid of letters, twisted like Midjourney's home page, on an old monitor with phosphor trails and a clock that fades in. Palettes: brass, paper, phosphor, amber, ice. `--text FILE` uses your own messages. |
| ![gauges](docs/img/gauges.jpg) | **gauges**: the machine's vital signs on a brass instrument panel: CPU temperature and load, a clock, memory, battery, and odometers for uptime and time away. |
| ![contours](docs/img/contours.jpg) | **contours**: a drifting topographic map of terrain carved by its own drainage, with streams, villages, roads and invented place names. The sun moves and hills cast shadows. Palettes: brass, cream, slate, blueprint, survey. |
| ![clockwork](docs/img/clockwork.jpg) | **clockwork**: an opened-up clock movement ticking at the real time; on a wall the side screens show the escapement and the gear train. |
| ![nightsky](docs/img/nightsky.jpg) | **nightsky**: the real sky for your place and time over a mountain lake: stars, Milky Way, Moon phase, planets, meteors, twilight. `--lines` draws constellations. |
| ![flightsim](docs/img/flightsim.jpg) | **flightsim**: flying low over procedural mountains, forests and lakes with a head-up display; spans a wall as one camera. |
| ![orrery](docs/img/orrery.jpg) | **orrery**: a brass and enamel orrery in a dim study, planets where they really are tonight, its gear train turning in true ratios; now and then it winds forward months and back. |
| ![soapfilm](docs/img/soapfilm.jpg) | **soapfilm**: physically computed thin-film colour: a draining film in a wire frame that turns black and bursts, drifting bubbles that merge and pop, and a coarsening foam. |
| ![aquarium](docs/img/aquarium.jpg) | **aquarium**: a planted freshwater tank through the front glass: schooling neon tetras, angelfish, discus and corys with lofted bodies and rayed fins, a fire eel that burrows into the sand, cherry and Amano shrimp picking at the stones. Caustics, light shafts and real time-of-day lighting; a thermometer on the glass reads the CPU temperature. |
| ![atc](docs/img/atc.jpg) | **atc**: an air-traffic radar scope, a paper flight-strip board and an approach view. Real ADS-B from a local receiver or, if a location is set, adsb.lol; otherwise simulated traffic. Airships (the Goodyear blimps) get special attention. |
| ![pendulums](docs/img/pendulums.jpg) | **pendulums**: chaos made visible. A thousand double pendulums started a hair apart swing as one line, fray into a rainbow ribbon and bloom into chaos; also the flip-time chaos map and a Poincaré section building up. Live Lyapunov and energy-error readouts. |
| ![voronoi](docs/img/voronoi.jpg) | **voronoi**: Lloyd's relaxation: hundreds of Voronoi cells settle into a honeycomb, are stirred by ripples, swirls and wind, divide and fuse, and now and then bud into the time or the hostname. Stained glass, giraffe, crackle, blueprint, paper and honey. |

Also in the rotation on hosts that build them: **pipes** and **maze**, the
xscreensaver classics, vendored as binaries and run through a private
Xwayland whose root window `windowtolayer` turns into a layer surface (see
`lock-bg.sh`).

### Benched

These are built and work, but stay out of the rotation until they're promoted
in `savers.sh`.

| | |
|---|---|
| ![splitflap](docs/img/splitflap.jpg) | **splitflap**: a railway departures board that clatters at each minute. Portrait screens get a tall platform board; a wall gets departures, arrivals and platform boards. Palettes: brass, classic, blue, led. |
| ![lavalamp](docs/img/lavalamp.jpg) | **lavalamp**: a lava lamp warming up from cold, its wax swelling, necking, pinching off, rising and sinking back; on a wall, one lamp per screen along a shared shelf. |
| ![ink](docs/img/ink.jpg) | **ink**: drops of ink falling into a tank of water and rolling into vortex rings and curling wisps, on a GPU fluid sim. Palettes: indigo, sumi, botanical, neon, sunset. |
| ![marbling](docs/img/marbling.jpg) | **marbling**: paper marbling made live: inks dropped, combed and raked into stone, chevron, nonpareil and bouquet patterns, then the sheet is lifted. |
| ![billiards](docs/img/billiards.jpg) | **billiards**: mathematical billiards. Ideal balls in circles, ellipses, stadia, cardioids and mushrooms trace glowing orbits next to their phase portraits: integrable, mixed and chaotic tables side by side. |
| ![algorithms](docs/img/algorithms.jpg) | **algorithms**: sorting and maze algorithms at work: 22 sorts shown as bars, dots, a colour wheel, a picture sorting itself back together and sorting networks; 11 maze generators and 6 solvers. On a wall, sorts race each other screen by screen, or one maze spans them all. |
| ![netflow](docs/img/netflow.jpg) | **netflow**: the machine's own networking as a live stack diagram: packets flowing through interfaces, IP and TCP/UDP/ICMP to the sockets, per-interface scopes and connection-state dials. Aggregate counters from /proc only: no addresses, ports or captures. |

## Building

Debian/Ubuntu packages:

```sh
sudo apt install build-essential pkg-config libgl-dev libegl-dev libx11-dev \
    libwayland-dev wayland-protocols libfontconfig-dev libfreetype-dev
```

Then build whichever savers you want on this host:

```sh
make -C stir
make -C nightsky
for s in $(. ./savers.sh; echo "${SAVERS[@]}"); do [ -f $s/Makefile ] && make -C $s; done
```

`flightsim/protocols/` holds the wlr-layer-shell protocol XML that every
saver's Makefile turns into Wayland glue.

## Running one by hand

Every saver has the same interface:

```sh
./stir/run.sh                       # fullscreen window (fullscreens itself under sway)
./stir/stir --window                # plain X11 window
./stir/stir --layer                 # wlr-layer-shell surface (what the lock screen uses)
./stir/stir --shot out.ppm --size 1920x1200 --at 30   # render one frame offscreen
./stir/stir --bench 300 --size 1920x1080              # time 300 frames offscreen
```

Common options: `--seed N`, `--palette NAME` (where there are palettes),
`--epoch UNIXTIME --layout 'x,y,w,h;...' --output N` (spanning). Run any
saver with `--help` for its own options.

To try the multi-monitor behaviour on one screen, render each output's slice
offscreen and paste them together:

```sh
L='0,0,1440,2560;1440,560,1969,1231;3409,560,2560,1440'
./nightsky/nightsky --shot o0.ppm --size 1440x2560 --layout "$L" --output 0 --epoch 1790000000
./nightsky/nightsky --shot o1.ppm --size 1969x1231 --layout "$L" --output 1 --epoch 1790000000
./nightsky/nightsky --shot o2.ppm --size 2560x1440 --layout "$L" --output 2 --epoch 1790000000
```

## Location, network and privacy

- **Location:** `nightsky` and `atc` read an optional `~/.config/wallpapers/location`
  containing `LAT LON` (north and east positive), or take `--lat/--lon`.
  Without it, nightsky uses Greenwich and atc simulates traffic.
- **Network:** only `atc` ever uses the network, and only when a location is
  set. It polls `api.adsb.lol` every 15 s with the coordinates rounded to two
  decimals, plus a check for the Goodyear airships every 5 minutes. It shares
  one cached fetch between all outputs and backs off on errors. A local
  readsb/dump1090 receiver on `127.0.0.1` is used in preference and keeps
  everything on the machine. `--sim` forces offline.
- **On screen:** the hostname, user name, times and aggregate machine
  statistics. Nothing else about the machine or the person.

## Adding a saver

1. Create `NAME/` with `NAME.c`, a `Makefile`, `run.sh` and a `.gitignore`
   for the binary and the generated `proto-*` files. Copy the plumbing
   (window, layer surface, `--shot`, `--bench`, layout) from an existing
   saver such as `gauges/`.
2. Add `NAME` to `BENCHED` in `savers.sh` (and to `SPANNING` if it takes
   `--epoch/--layout/--output`), and to the `pkill` pattern in `lock-next.sh`.
3. When it's ready, move it from `BENCHED` to `SAVERS`.
