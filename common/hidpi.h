/*
 * hidpi.h — render a layer surface at the output's real pixel density.
 *
 * Without this a saver is told its output's *logical* size (1920x1200 on a
 * 3840x2400 panel at scale 2) and draws that many pixels, which the
 * compositor then stretches: soft, half-resolution lines. With it the saver
 * asks for the output's preferred (possibly fractional) scale via
 * wp_fractional_scale_v1, draws a buffer of logical size x scale, and uses
 * wp_viewporter to tell the compositor that buffer covers the logical area.
 *
 * --render-scale F (0 < F <= 1) draws F times the native pixels instead, for
 * savers too heavy to run at full resolution on a given GPU; lock-bg.sh reads
 * it per saver from ~/.config/wallpapers/render-scale.
 *
 * Use, in a saver's Wayland code:
 *   wl_global():    if (hidpi_bind(&hd, r, name, intf, ver)) return;
 *   after wl_compositor_create_surface():  hidpi_attach(&hd, wl.surf);
 *   ls_configure(): hidpi_logical(&hd, w, h); then hidpi_size(&hd, &W, &H)
 *   once configured: wl_display_roundtrip(), so the scale has arrived
 *                    before the EGL window is created, then hidpi_size again
 *   each frame:     if (hidpi_take_change(&hd)) { hidpi_size(); resize }
 * Without the protocols (older compositors) it falls back to drawing at the
 * logical size, as before.
 *
 * The Makefile fragment common/hidpi.mk generates the protocol glue.
 */
#ifndef HIDPI_H
#define HIDPI_H

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <wayland-client.h>
#include "proto-fractional-scale.h"
#include "proto-viewporter.h"

struct hidpi {
    struct wp_viewporter *vporter;
    struct wp_fractional_scale_manager_v1 *fsm;
    struct wp_viewport *view;
    struct wp_fractional_scale_v1 *fs;
    uint32_t scale120;          /* the output's scale x 120; 120 = 1.0 */
    int lw, lh;                 /* logical size, from the layer-shell configure */
    double render;              /* --render-scale: fraction of native pixels to draw */
    int changed;                /* the buffer size should be recomputed */
};

static void hidpi_preferred(void *data, struct wp_fractional_scale_v1 *fs, uint32_t scale) {
    (void)fs;
    struct hidpi *h = data;
    if (scale && scale != h->scale120) { h->scale120 = scale; h->changed = 1; }
}
static const struct wp_fractional_scale_v1_listener hidpi_fs_l = {hidpi_preferred};

static inline void hidpi_init(struct hidpi *h, double render_scale) {
    memset(h, 0, sizeof *h);
    h->scale120 = 120;
    h->render = render_scale > 0 && render_scale <= 1 ? render_scale : 1;
}

/* call from the registry's global handler; returns 1 if it took the global */
static inline int hidpi_bind(struct hidpi *h, struct wl_registry *r, uint32_t name, const char *intf, uint32_t ver) {
    (void)ver;
    if (!strcmp(intf, wp_viewporter_interface.name)) {
        h->vporter = wl_registry_bind(r, name, &wp_viewporter_interface, 1);
        return 1;
    }
    if (!strcmp(intf, wp_fractional_scale_manager_v1_interface.name)) {
        h->fsm = wl_registry_bind(r, name, &wp_fractional_scale_manager_v1_interface, 1);
        return 1;
    }
    return 0;
}

/* once the surface exists, before its first commit */
static inline void hidpi_attach(struct hidpi *h, struct wl_surface *surf) {
    if (!h->vporter) return;                       /* can't map a bigger buffer onto the logical area */
    h->view = wp_viewporter_get_viewport(h->vporter, surf);
    if (h->fsm) {
        h->fs = wp_fractional_scale_manager_v1_get_fractional_scale(h->fsm, surf);
        wp_fractional_scale_v1_add_listener(h->fs, &hidpi_fs_l, h);
    }
}

/* the logical size the layer surface was configured to */
static inline void hidpi_logical(struct hidpi *h, int w, int h_) {
    if (w > 0 && h_ > 0 && (w != h->lw || h_ != h->lh)) { h->lw = w; h->lh = h_; h->changed = 1; }
}

/* the buffer size to draw; also (re)sets the viewport's destination */
static inline void hidpi_size(struct hidpi *h, int *bw, int *bh) {
    if (h->lw <= 0 || h->lh <= 0) return;
    double k = h->view ? h->scale120 / 120.0 * h->render : 1.0;
    int w = (int)lround(h->lw * k), hh = (int)lround(h->lh * k);
    *bw = w < 1 ? 1 : w;
    *bh = hh < 1 ? 1 : hh;
    if (h->view) wp_viewport_set_destination(h->view, h->lw, h->lh);
}

/* whether the size needs recomputing since last asked */
static inline int hidpi_take_change(struct hidpi *h) {
    int c = h->changed;
    h->changed = 0;
    return c;
}

/* the factor from logical to buffer pixels, e.g. to scale a spanning layout */
static inline double hidpi_factor(const struct hidpi *h, int bw) {
    return h->lw > 0 ? (double)bw / h->lw : 1.0;
}

#endif
