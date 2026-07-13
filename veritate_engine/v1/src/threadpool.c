// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - persistent thread-pool shim. spin-then-park dispatch: the hot path (wake +
//   completion) is lock-free atomics so a sub-us matvec split beats the pool
//   overhead; workers park on a condvar only after a bounded idle spin, so an
//   idle engine burns no cores (energy target). condvar/SRWLOCK is the park
//   fallback, never the wake latency.
// - lazy init on first veritate_pool_run/size. workers stay alive for process
//   life. generic per-call surface: caller hands a work fn and N arg pointers;
//   pool wakes the first N workers, blocks until each returns.
// - VERITATE_POOL_SPIN overrides the per-side spin budget (iterations).
// veritate_engine/src/threadpool.c
// ------------------------------------------------------------------------------------
// Imports:

#include "portability.h"

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_WIN32)
    #include <windows.h>
    typedef SRWLOCK            pool_mu_t;
    typedef CONDITION_VARIABLE pool_cv_t;
    #define POOL_MU_INIT(m)   InitializeSRWLock(m)
    #define POOL_MU_LOCK(m)   AcquireSRWLockExclusive(m)
    #define POOL_MU_UNLOCK(m) ReleaseSRWLockExclusive(m)
    #define POOL_CV_INIT(c)   InitializeConditionVariable(c)
    #define POOL_CV_WAIT(c,m) SleepConditionVariableSRW((c), (m), INFINITE, 0)
    #define POOL_CV_SIGNAL(c) WakeConditionVariable(c)
#else
    #include <pthread.h>
    #include <unistd.h>
    #if defined(__APPLE__)
        #include <sys/sysctl.h>
    #endif
    typedef pthread_mutex_t pool_mu_t;
    typedef pthread_cond_t  pool_cv_t;
    #define POOL_MU_INIT(m)   pthread_mutex_init((m), NULL)
    #define POOL_MU_LOCK(m)   pthread_mutex_lock(m)
    #define POOL_MU_UNLOCK(m) pthread_mutex_unlock(m)
    #define POOL_CV_INIT(c)   pthread_cond_init((c), NULL)
    #define POOL_CV_WAIT(c,m) pthread_cond_wait((c), (m))
    #define POOL_CV_SIGNAL(c) pthread_cond_signal(c)
#endif

#if defined(__x86_64__) || defined(_M_X64)
    #define POOL_RELAX() __asm__ __volatile__("pause" ::: "memory")
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define POOL_RELAX() __asm__ __volatile__("yield" ::: "memory")
#else
    #define POOL_RELAX() ((void)0)
#endif

// ------------------------------------------------------------------------------------
// Constants

#define POOL_DEFAULT_CAP  VERITATE_MAX_THREADS

// idle spin budget per side before parking. ~covers the inter-dispatch serial
// gaps of a decode burst (tens of us) so workers stay hot mid-request, then park
// within a fraction of a ms once the stream goes quiet.
#define POOL_SPIN_ITERS   200000

// ------------------------------------------------------------------------------------
// Functions

typedef struct {
    veritate_work_fn fn;
    void*            arg;
    int32_t          worker_idx;
    _Atomic int32_t  wake_flag;
    _Atomic int32_t  parked;
    pool_mu_t        wake_mu;
    pool_cv_t        wake_cv;
    int              alive;
#if defined(_WIN32)
    HANDLE thread;
#else
    pthread_t thread;
#endif
} pool_worker_t;

static pool_worker_t   pool[POOL_DEFAULT_CAP];
static int32_t         pool_n = 0;
static int             pool_ready = 0;
static long            pool_spin = POOL_SPIN_ITERS;

static _Atomic int32_t pool_done_count = 0;
static _Atomic int32_t pool_main_waiting = 0;
static pool_mu_t       pool_done_mu;
static pool_cv_t       pool_done_cv;

#if defined(_WIN32)
static CRITICAL_SECTION pool_init_lock;
static int              pool_init_lock_ready = 0;
#else
static pthread_mutex_t  pool_init_lock = PTHREAD_MUTEX_INITIALIZER;
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
#else
static void* worker_loop(void* raw) {
#endif
    pool_worker_t* w = (pool_worker_t*)raw;
    while (1) {
        int got = 0;
        for (long i = 0; i < pool_spin; i++) {
            if (atomic_load_explicit(&w->wake_flag, memory_order_acquire)) { got = 1; break; }
            POOL_RELAX();
        }
        if (!got) {
            POOL_MU_LOCK(&w->wake_mu);
            atomic_store_explicit(&w->parked, 1, memory_order_seq_cst);
            while (!atomic_load_explicit(&w->wake_flag, memory_order_seq_cst))
                POOL_CV_WAIT(&w->wake_cv, &w->wake_mu);
            atomic_store_explicit(&w->parked, 0, memory_order_relaxed);
            POOL_MU_UNLOCK(&w->wake_mu);
        }
        atomic_store_explicit(&w->wake_flag, 0, memory_order_relaxed);
        if (!w->alive) break;
        w->fn(w->arg, w->worker_idx);
        atomic_fetch_add_explicit(&pool_done_count, 1, memory_order_seq_cst);
        if (atomic_load_explicit(&pool_main_waiting, memory_order_seq_cst)) {
            POOL_MU_LOCK(&pool_done_mu);
            POOL_CV_SIGNAL(&pool_done_cv);
            POOL_MU_UNLOCK(&pool_done_mu);
        }
    }
#if defined(_WIN32)
    return 0;
#else
    return NULL;
#endif
}

// ------------------------------------------------------------------------------------

static void pool_init_once(void) {
#if defined(_WIN32)
    if (!pool_init_lock_ready) { InitializeCriticalSection(&pool_init_lock); pool_init_lock_ready = 1; }
    EnterCriticalSection(&pool_init_lock);
#else
    pthread_mutex_lock(&pool_init_lock);
#endif
    if (!pool_ready) {
        const char* sp = getenv("VERITATE_POOL_SPIN");
        if (sp && *sp) { long v = atol(sp); if (v >= 0) pool_spin = v; }
        int32_t n = detect_cpu_count();
        if (n > POOL_DEFAULT_CAP) n = POOL_DEFAULT_CAP;
        if (n < 1) n = 1;
        pool_n = n;
        POOL_MU_INIT(&pool_done_mu);
        POOL_CV_INIT(&pool_done_cv);
        for (int32_t t = 0; t < pool_n; t++) {
            pool[t].fn         = NULL;
            pool[t].arg        = NULL;
            pool[t].worker_idx = t;
            pool[t].alive      = 1;
            atomic_store_explicit(&pool[t].wake_flag, 0, memory_order_relaxed);
            atomic_store_explicit(&pool[t].parked,    0, memory_order_relaxed);
            POOL_MU_INIT(&pool[t].wake_mu);
            POOL_CV_INIT(&pool[t].wake_cv);
#if defined(_WIN32)
            pool[t].thread = CreateThread(NULL, 0, worker_loop, &pool[t], 0, NULL);
#else
            pthread_create(&pool[t].thread, NULL, worker_loop, &pool[t]);
#endif
        }
        pool_ready = 1;
    }
#if defined(_WIN32)
    LeaveCriticalSection(&pool_init_lock);
#else
    pthread_mutex_unlock(&pool_init_lock);
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

    atomic_store_explicit(&pool_done_count, 0, memory_order_seq_cst);
    for (int32_t t = 0; t < n; t++) {
        pool[t].fn  = fn;
        pool[t].arg = (void*)args[t];
        atomic_store_explicit(&pool[t].wake_flag, 1, memory_order_seq_cst);
        if (atomic_load_explicit(&pool[t].parked, memory_order_seq_cst)) {
            POOL_MU_LOCK(&pool[t].wake_mu);
            POOL_CV_SIGNAL(&pool[t].wake_cv);
            POOL_MU_UNLOCK(&pool[t].wake_mu);
        }
    }

    int done = 0;
    for (long i = 0; i < pool_spin; i++) {
        if (atomic_load_explicit(&pool_done_count, memory_order_acquire) >= n) { done = 1; break; }
        POOL_RELAX();
    }
    if (!done) {
        POOL_MU_LOCK(&pool_done_mu);
        atomic_store_explicit(&pool_main_waiting, 1, memory_order_seq_cst);
        while (atomic_load_explicit(&pool_done_count, memory_order_seq_cst) < n)
            POOL_CV_WAIT(&pool_done_cv, &pool_done_mu);
        atomic_store_explicit(&pool_main_waiting, 0, memory_order_relaxed);
        POOL_MU_UNLOCK(&pool_done_mu);
    }
}
