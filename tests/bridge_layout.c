/* Compile with the same libraries as bridge/Makefile; no display needed. */
#define main bridge_program_main
#include "../bridge/bridge.c"
#undef main
#include <assert.h>

static void check_panel(const Box *b) {
    assert(b->on && b->w > 0 && b->h > 0);
    int inside = 0;
    for (int i = 0; i < span_n; i++) {
        const float *r = span[i];
        inside |= b->x >= r[0] && b->y >= r[1] &&
                  b->x + b->w <= r[0] + r[2] + 0.01f &&
                  b->y + b->h <= r[1] + r[3] + 0.01f;
    }
    assert(inside);
}

int main(void) {
    parse_layout("0,0,1440,2560;1440,665,1969,1231;3409,560,2560,1440");
    double lengths[NBRIDGE] = {0}, starts[NBRIDGE] = {0};
    for (span_me = 0; span_me < span_n; span_me++) {
        W = (int)(span[span_me][2] / 2); H = (int)(span[span_me][3] / 2);
        layout();
        assert(stage[2] > 0 && stage[3] > 0);
        for (int i = 0; i < NBOX; i++) check_panel(&box[i]);
        assert(box[BX_READ].x < 1440 && box[BX_PLOT].x > 3409);
        for (int k = 0; k < NBRIDGE; k++) {
            Br b;
            br_build(&b, k, 42);
            bridge_fit(&b);
            assert(isfinite(wk) && wk > 0);
            for (int i = 0; i < b.ntrain; i++) {
                const Train *t = &b.tr[i];
                assert(t->v > 0 ? WX(t->s0) < 0 : WX(t->s0) > canvas_w);
            }
            if (!span_me) { lengths[k] = b.span; starts[k] = b.tr[0].s0; }
            else { assert(fabs(lengths[k] - b.span) < 1e-9); assert(fabs(starts[k] - b.tr[0].s0) < 1e-9); }
            br_free(&b);
        }
    }
    /* A carriage straddling either outer screen edge is drawn at full
     * opacity, leaving the GPU to clip just the offscreen part. */
    canvas_w = 100; wk = 1; wox = 0; woy = 20; U = sc = 1;
    stage[0] = 30; stage[2] = 40; P = &pals[0];
    Br b = {0};
    for (int side = 0; side < 2; side++) {
        Train t = {.s0 = side ? 90 : 10, .v = side ? -1 : 1, .ncar = 1};
        t.car0[0] = 0.6; t.car1[0] = 20.4; t.loco[0] = 1;
        nvb = 0;
        draw_train(&b, &t, 0);
        assert(nvb > 0 && vb[0].a > 0.9);
        int outside = 0, inside = 0;
        for (int i = 0; i < nvb; i++) {
            outside |= vb[i].x < 0 || vb[i].x > canvas_w;
            inside |= vb[i].x >= 0 && vb[i].x <= canvas_w;
        }
        assert(outside && inside);
    }
    puts("Bridge: monitor layouts, shared geometry, offscreen train starts and partial carriages pass");
    return 0;
}
