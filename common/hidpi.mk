# Protocol glue for common/hidpi.h, generated into the saver's own directory
# (like its other proto-* files). Include from a saver's Makefile and add
# $(HIDPI_H) to its prerequisites and $(HIDPI_C) to its sources. Include it
# after the saver's own rule, so that stays the default goal, and compile
# with -I. so hidpi.h finds the generated headers.
WL_PROTOCOLS ?= $(shell pkg-config --variable=pkgdatadir wayland-protocols)
FRACTIONAL_XML = $(WL_PROTOCOLS)/staging/fractional-scale/fractional-scale-v1.xml
VIEWPORTER_XML = $(WL_PROTOCOLS)/stable/viewporter/viewporter.xml
HIDPI_H ?= proto-fractional-scale.h proto-viewporter.h
HIDPI_C ?= proto-fractional-scale.c proto-viewporter.c

proto-fractional-scale.h: $(FRACTIONAL_XML)
	wayland-scanner client-header $< $@
proto-fractional-scale.c: $(FRACTIONAL_XML)
	wayland-scanner private-code $< $@
proto-viewporter.h: $(VIEWPORTER_XML)
	wayland-scanner client-header $< $@
proto-viewporter.c: $(VIEWPORTER_XML)
	wayland-scanner private-code $< $@
