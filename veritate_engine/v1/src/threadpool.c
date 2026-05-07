// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - persistent thread-pool shim. windows: CreateThread + auto-reset events.
//   posix: pthread + condvar pairs that mimic auto-reset events.
// - lazy init on first veritate_pool_run. workers stay alive for process life.
// - generic per-call surface: caller hands a work fn and N arg pointers; pool
//   wakes the first N workers, blocks until each returns.
// veritate_engine/src/threadpool.c
// ------------------------------------------------------------------------------------
// Imports:

#include "portability.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
    #include <windows.h>
#else
    #include <pthread.h>
    #include <unistd.h>
    #if defined(__APPLE__)
        #include <sys/sysctl.h>
    #endif
#endif

// ------------------------------------------------------------------------------------
// Constants

// initial worker count cap. real count is min(detected_cpus, VERITATE_MAX_THREADS).
#define POOL_DEFAULT_CAP VERITATE_MAX_THREADS

// ------------------------------------------------------------------------------------
// Functions

typedef struct {
    veritate_work_fn fn;
    void*            arg;
    int32_t          worker_idx;
#if defined(_WIN32)
    HANDLE wake;
    HANDLE done;
    HANDLE thread;
#else
    pthread_mutex_t  wake_mu;
    pthread_cond_t   wake_cv;
    int              wake_flag;
    pthread_mutex_t  done_mu;
    pthread_cond_t   done_cv;
    int              done_flag;
    pthread_t        thread;
#endif
    int alive;
} pool_worker_t;

static pool_worker_t pool[POOL_DEFAULT_CAP];
static int32_t       pool_n = 0;
static int           pool_ready = 0;

#if defined(_WIN32)
static CRITICAL_SECTION pool_lock;
static int              pool_lock_init = 0;
#else
static pthread_mutex_t  pool_lock = PTHREAD_MUTEX_INITIALIZER;
#endif

// ------------------------------------------------------------------------------------

static int32_t detect_cpu_count(void) {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    int32_t n = (int32_t)si.dwNumberOfProcessors;
    return n > 0 ? n : 1;
#elif defined(__APPLE__)
    int32_t n = 0;
    size_t  sz = sizeof(n);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &sz, NULL, 0) == 0 && n > 0) return n;
    if (sysctlbyname("hw.activecpu",              &n, &sz, NULL, 0) == 0 && n > 0) return n;
    long fb = sysconf(_SC_NPROCESSORS_ONLN);
    return fb > 0 ? (int32_t)fb : 1;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int32_t)n : 1;
#endif
}

// ------------------------------------------------------------------------------------

#if defined(_WIN32)
static DWORD WINAPI worker_loop(LPVOID raw) {
    pool_worker_t* w = (pool_worker_t*)raw;
    while (1) {
        WaitForSingleObject(w->wake, INFINITE);
        if (!w->alive) return 0;
        w->fn(w->arg, w->worker_idx);
        SetEvent(w->done);
    }
}
#else
static void* worker_loop(void* raw) {
    pool_worker_t* w = (pool_worker_t*)raw;
    while (1) {
        pthread_mutex_lock(&w->wake_mu);
        while (!w->wake_flag) pthread_cond_wait(&w->wake_cv, &w->wake_mu);
        w->wake_flag = 0;
        pthread_mutex_unlock(&w->wake_mu);

        if (!w->alive) return NULL;
        w->fn(w->arg, w->worker_idx);

        pthread_mutex_lock(&w->done_mu);
        w->done_flag = 1;
        pthread_cond_signal(&w->done_cv);
        pthread_mutex_unlock(&w->done_mu);
    }
}
#endif

// ------------------------------------------------------------------------------------

static void pool_init_once(void) {
#if defined(_WIN32)
    if (!pool_lock_init) { InitializeCriticalSection(&pool_lock); pool_lock_init = 1; }
    EnterCriticalSection(&pool_lock);
#else
    pthread_mutex_lock(&pool_lock);
#endif
    if (!pool_ready) {
        int32_t n = detect_cpu_count();
        if (n > POOL_DEFAULT_CAP) n = POOL_DEFAULT_CAP;
        if (n < 1) n = 1;
        pool_n = n;
        for (int32_t t = 0; t < pool_n; t++) {
            pool[t].fn         = NULL;
            pool[t].arg        = NULL;
            pool[t].worker_idx = t;
            pool[t].alive      = 1;
#if defined(_WIN32)
            pool[t].wake   = CreateEvent(NULL, FALSE, FALSE, NULL);
            pool[t].done   = CreateEvent(NULL, FALSE, FALSE, NULL);
            pool[t].thread = CreateThread(NULL, 0, worker_loop, &pool[t], 0, NULL);
#else
            pthread_mutex_init(&pool[t].wake_mu, NULL);
            pthread_cond_init (&pool[t].wake_cv, NULL);
            pool[t].wake_flag = 0;
            pthread_mutex_init(&pool[t].done_mu, NULL);
            pthread_cond_init (&pool[t].done_cv, NULL);
            pool[t].done_flag = 0;
            pthread_create(&pool[t].thread, NULL, worker_loop, &pool[t]);
#endif
        }
        pool_ready = 1;
    }
#if defined(_WIN32)
    LeaveCriticalSection(&pool_lock);
#else
    pthread_mutex_unlock(&pool_lock);
#endif
}

// ------------------------------------------------------------------------------------

int32_t veritate_pool_size(void) {
    if (!pool_ready) pool_init_once();
    return pool_n;
}

// ------------------------------------------------------------------------------------

void veritate_pool_run(veritate_work_fn fn, void* const* args, int32_t n) {
    if (!pool_ready) pool_init_once();
    if (n < 1) return;
    if (n > pool_n) n = pool_n;

    for (int32_t t = 0; t < n; t++) {
        pool[t].fn  = fn;
        pool[t].arg = (void*)args[t];
#if defined(_WIN32)
        SetEvent(pool[t].wake);
#else
        pthread_mutex_lock(&pool[t].wake_mu);
        pool[t].wake_flag = 1;
        pthread_cond_signal(&pool[t].wake_cv);
        pthread_mutex_unlock(&pool[t].wake_mu);
#endif
    }

#if defined(_WIN32)
    HANDLE done_handles[POOL_DEFAULT_CAP];
    for (int32_t t = 0; t < n; t++) done_handles[t] = pool[t].done;
    WaitForMultipleObjects((DWORD)n, done_handles, TRUE, INFINITE);
#else
    for (int32_t t = 0; t < n; t++) {
        pthread_mutex_lock(&pool[t].done_mu);
        while (!pool[t].done_flag) pthread_cond_wait(&pool[t].done_cv, &pool[t].done_mu);
        pool[t].done_flag = 0;
        pthread_mutex_unlock(&pool[t].done_mu);
    }
#endif
}
