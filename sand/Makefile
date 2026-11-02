CFLAGS ?= -O2 -Wall -Wextra

# Wayland protocol glue (generated): wlr-layer-shell for --layer, and the
# xdg-shell it refers to for popups
XDG_SHELL = $(shell pkg-config --variable=pkgdatadir wayland-protocols)/stable/xdg-shell/xdg-shell.xml
LAYER_XML = ../flightsim/protocols/wlr-layer-shell-unstable-v1.xml
PROTO_H = proto-layer-shell.h
PROTO_C = proto-layer-shell.c proto-xdg-shell.c

sand: sand.c $(PROTO_H) $(PROTO_C)
	$(CC) $(CFLAGS) -o $@ sand.c $(PROTO_C) -lGL -lX11 -lEGL -lwayland-client -lwayland-egl -lm

proto-layer-shell.h: $(LAYER_XML)
	wayland-scanner client-header $< $@
proto-layer-shell.c: $(LAYER_XML)
	wayland-scanner private-code $< $@
proto-xdg-shell.c: $(XDG_SHELL)
	wayland-scanner private-code $< $@

clean:
	rm -f sand $(PROTO_H) $(PROTO_C)
