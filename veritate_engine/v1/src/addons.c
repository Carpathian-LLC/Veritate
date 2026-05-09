// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - addon chain implementation + global registry + csv parser.
// - factories live with their addon (e.g. addons/slot_table.c). the registry below
//   maps the addon id string to its factory. add new ids here in the same commit
//   as the addon's source file.
// veritate_engine/src/addons.c
// ------------------------------------------------------------------------------------
// Imports

#include "addons.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ------------------------------------------------------------------------------------
// Constants

#define ADDON_CHAIN_INITIAL_CAP 4
#define ADDON_ID_MAX_LEN        64

addon_chain_t* g_addons_chain = NULL;

typedef addon_t* (*addon_factory_fn)(void);

typedef struct {
    const char*       id;
    addon_factory_fn  factory;
} addon_factory_entry_t;

static const addon_factory_entry_t ADDON_FACTORIES[] = {
    { "slot_table", addon_slot_table_new },
    { NULL,         NULL                 },
};


// ------------------------------------------------------------------------------------
// Functions

void addon_chain_init(addon_chain_t* c) {
    c->addons = NULL;
    c->count  = 0;
    c->cap    = 0;
}

void addon_chain_free(addon_chain_t* c) {
    if (c == NULL) return;
    for (int32_t i = 0; i < c->count; i++) {
        addon_t* a = c->addons[i];
        if (a != NULL && a->vtable != NULL && a->vtable->destroy != NULL) {
            a->vtable->destroy(a);
        }
        free(a);
    }
    free(c->addons);
    c->addons = NULL;
    c->count  = 0;
    c->cap    = 0;
}

void addon_chain_add(addon_chain_t* c, addon_t* a) {
    if (a == NULL) return;
    if (c->count == c->cap) {
        int32_t new_cap = c->cap == 0 ? ADDON_CHAIN_INITIAL_CAP : c->cap * 2;
        addon_t** grown = (addon_t**)realloc(c->addons, (size_t)new_cap * sizeof(addon_t*));
        if (grown == NULL) return;
        c->addons = grown;
        c->cap    = new_cap;
    }
    c->addons[c->count++] = a;
}

void addon_chain_reset(addon_chain_t* c) {
    if (c == NULL) return;
    for (int32_t i = 0; i < c->count; i++) {
        addon_t* a = c->addons[i];
        if (a != NULL && a->vtable != NULL && a->vtable->reset != NULL) {
            a->vtable->reset(a);
        }
    }
}

void addon_chain_observe(addon_chain_t* c, int byte_int) {
    if (c == NULL) return;
    for (int32_t i = 0; i < c->count; i++) {
        addon_t* a = c->addons[i];
        if (a != NULL && a->vtable != NULL && a->vtable->observe != NULL) {
            a->vtable->observe(a, byte_int);
        }
    }
}

void addon_chain_observe_bytes(addon_chain_t* c, const uint8_t* bs, int32_t n) {
    if (c == NULL) return;
    for (int32_t i = 0; i < n; i++) {
        addon_chain_observe(c, (int)bs[i]);
    }
}

void addon_chain_bias_logits(addon_chain_t* c, float* logits, int32_t vocab) {
    if (c == NULL) return;
    for (int32_t i = 0; i < c->count; i++) {
        addon_t* a = c->addons[i];
        if (a != NULL && a->vtable != NULL && a->vtable->bias_logits != NULL) {
            a->vtable->bias_logits(a, logits, vocab);
        }
    }
}

void addons_set_global(addon_chain_t* c) {
    g_addons_chain = c;
}

addon_chain_t* addons_get_global(void) {
    return g_addons_chain;
}

static addon_factory_fn lookup_factory(const char* id) {
    for (const addon_factory_entry_t* e = ADDON_FACTORIES; e->id != NULL; e++) {
        if (strcmp(e->id, id) == 0) return e->factory;
    }
    return NULL;
}

addon_chain_t* addons_build_chain(const char* csv) {
    addon_chain_t* c = (addon_chain_t*)malloc(sizeof(addon_chain_t));
    if (c == NULL) return NULL;
    addon_chain_init(c);
    if (csv == NULL || csv[0] == '\0') return c;

    char buf[ADDON_ID_MAX_LEN];
    const char* p = csv;
    while (*p != '\0') {
        while (*p == ' ' || *p == ',') p++;
        if (*p == '\0') break;
        int32_t n = 0;
        while (*p != '\0' && *p != ',' && n < ADDON_ID_MAX_LEN - 1) {
            buf[n++] = *p++;
        }
        buf[n] = '\0';
        while (n > 0 && buf[n - 1] == ' ') buf[--n] = '\0';
        if (n == 0) continue;
        addon_factory_fn f = lookup_factory(buf);
        if (f == NULL) {
            fprintf(stderr, "addons: unknown id '%s', skipping\n", buf);
            continue;
        }
        addon_t* a = f();
        if (a == NULL) {
            fprintf(stderr, "addons: factory for '%s' returned NULL, skipping\n", buf);
            continue;
        }
        addon_chain_add(c, a);
    }
    return c;
}
