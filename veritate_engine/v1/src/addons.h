// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - inference-time addon contract for the C engine. mirrors the python contract
//   in veritate_mri/addons/ -- one vtable per addon (reset / observe / bias_logits)
//   plus a chain that composes them. spec at documentation/addons/c_engine_port.md.
// - the bias_logits hook fires after the lm_head matmul and before sampling, on a
//   float copy of the logits scaled by the model's output requant. addons mutate
//   the float buffer in place. order is the order addons were added to the chain.
// veritate_engine/src/addons.h
// ------------------------------------------------------------------------------------

#ifndef VERITATE_ADDONS_H
#define VERITATE_ADDONS_H

#include <stdint.h>
#include <stddef.h>

// ------------------------------------------------------------------------------------
// addon contract -- one vtable per addon. caller owns the addon_t lifecycle.

typedef struct addon_t addon_t;

typedef struct {
    void (*reset)      (addon_t* self);
    void (*observe)    (addon_t* self, int byte_int);
    void (*bias_logits)(addon_t* self, float* logits, int32_t vocab);
    void (*destroy)    (addon_t* self);
} addon_vtable_t;

struct addon_t {
    const char*           id;       // e.g. "slot_table". points at static storage.
    const addon_vtable_t* vtable;
    void*                 state;    // opaque per-addon
};

// chain composes a list of addons. broadcasts reset/observe; pipes bias_logits in
// order. zero-cost when empty (chain == NULL or count == 0).
typedef struct {
    addon_t** addons;
    int32_t   count;
    int32_t   cap;
} addon_chain_t;

void addon_chain_init   (addon_chain_t* c);
void addon_chain_free   (addon_chain_t* c);
void addon_chain_add    (addon_chain_t* c, addon_t* a);
void addon_chain_reset  (addon_chain_t* c);
void addon_chain_observe(addon_chain_t* c, int byte_int);
void addon_chain_observe_bytes(addon_chain_t* c, const uint8_t* bs, int32_t n);
void addon_chain_bias_logits(addon_chain_t* c, float* logits, int32_t vocab);

// ------------------------------------------------------------------------------------
// global chain. set once at startup from --addons <csv>; sample_token_ext reads it
// before sampling. NULL means "no addons configured", zero-cost path.

extern addon_chain_t* g_addons_chain;

void addons_set_global(addon_chain_t* c);
addon_chain_t* addons_get_global(void);

// build a chain from a comma-separated id list. unknown ids are skipped with a
// warning to stderr. returns a heap-allocated chain owned by the caller; free with
// addon_chain_free + free(c).
addon_chain_t* addons_build_chain(const char* csv);

// ------------------------------------------------------------------------------------
// per-addon factories. one per shipped addon. each returns an addon_t* whose
// vtable + state are owned by the caller. NULL on allocation failure.

addon_t* addon_slot_table_new(void);

#endif
