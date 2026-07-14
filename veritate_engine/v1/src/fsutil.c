// ------------------------------------------------------------------------------------
// Developed by Carpathian, LLC.
// ------------------------------------------------------------------------------------
// Legal Notice: Distribution Not Authorized.
// ------------------------------------------------------------------------------------
// Notes:
// - file-system shim for the persistent state cache. windows uses
//   FindFirstFile / MoveFileEx(REPLACE_EXISTING) / SetFileTime; posix uses
//   dirent / stat / rename / utimensat. dual-OS split mirrors threadpool.c.
// - state_cache.c consumes this surface and never includes an OS header (rule 34).
// veritate_engine/src/fsutil.c
// ------------------------------------------------------------------------------------
// Imports:

#include "portability.h"

#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
    #include <windows.h>
    #include <sys/stat.h>
#else
    #include <dirent.h>
    #include <errno.h>
    #include <fcntl.h>
    #include <sys/stat.h>
    #include <sys/types.h>
#endif

// ------------------------------------------------------------------------------------
// Constants

// 100ns ticks between the win32 (1601) and unix (1970) epochs.
#define FS_WIN_EPOCH_TICKS  116444736000000000ULL
#define FS_NS_PER_SEC       1000000000ULL
#define FS_DIR_MODE         0777

// ------------------------------------------------------------------------------------
// Functions

#if defined(_WIN32)

static uint64_t fs_filetime_ns(FILETIME ft) {
    ULARGE_INTEGER u;
    u.LowPart = ft.dwLowDateTime;
    u.HighPart = ft.dwHighDateTime;
    if (u.QuadPart < FS_WIN_EPOCH_TICKS) return 0;
    return (u.QuadPart - FS_WIN_EPOCH_TICKS) * 100ULL;
}

int veritate_stat(const char* path, uint64_t* size, uint64_t* mtime_ns) {
    WIN32_FILE_ATTRIBUTE_DATA fad;
    if (!GetFileAttributesExA(path, GetFileExInfoStandard, &fad)) return -1;
    ULARGE_INTEGER sz;
    sz.LowPart = fad.nFileSizeLow;
    sz.HighPart = fad.nFileSizeHigh;
    if (size) *size = sz.QuadPart;
    if (mtime_ns) *mtime_ns = fs_filetime_ns(fad.ftLastWriteTime);
    return 0;
}

void veritate_dir_list(const char* dir, veritate_dir_fn cb, void* ctx) {
    char pattern[1024];
    snprintf(pattern, sizeof(pattern), "%s\\*", dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        ULARGE_INTEGER sz;
        sz.LowPart = fd.nFileSizeLow;
        sz.HighPart = fd.nFileSizeHigh;
        cb(fd.cFileName, sz.QuadPart, fs_filetime_ns(fd.ftLastWriteTime), ctx);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
}

int veritate_remove(const char* path) {
    return DeleteFileA(path) ? 0 : -1;
}

int veritate_rename(const char* from, const char* to) {
    return MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? 0 : -1;
}

int veritate_touch(const char* path) {
    HANDLE h = CreateFileA(path, FILE_WRITE_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE,
                           NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return -1;
    FILETIME now;
    GetSystemTimeAsFileTime(&now);
    int ok = SetFileTime(h, NULL, NULL, &now) ? 0 : -1;
    CloseHandle(h);
    return ok;
}

int veritate_mkdir(const char* path) {
    if (CreateDirectoryA(path, NULL)) return 0;
    return GetLastError() == ERROR_ALREADY_EXISTS ? 0 : -1;
}

#else

static uint64_t fs_timespec_ns(const struct stat* st) {
#if defined(__APPLE__)
    return (uint64_t)st->st_mtimespec.tv_sec * FS_NS_PER_SEC + (uint64_t)st->st_mtimespec.tv_nsec;
#else
    return (uint64_t)st->st_mtim.tv_sec * FS_NS_PER_SEC + (uint64_t)st->st_mtim.tv_nsec;
#endif
}

int veritate_stat(const char* path, uint64_t* size, uint64_t* mtime_ns) {
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    if (size) *size = (uint64_t)st.st_size;
    if (mtime_ns) *mtime_ns = fs_timespec_ns(&st);
    return 0;
}

void veritate_dir_list(const char* dir, veritate_dir_fn cb, void* ctx) {
    DIR* d = opendir(dir);
    if (!d) return;
    struct dirent* e;
    char full[1024];
    while ((e = readdir(d)) != NULL) {
        snprintf(full, sizeof(full), "%s/%s", dir, e->d_name);
        struct stat st;
        if (stat(full, &st) != 0 || !S_ISREG(st.st_mode)) continue;
        cb(e->d_name, (uint64_t)st.st_size, fs_timespec_ns(&st), ctx);
    }
    closedir(d);
}

int veritate_remove(const char* path) {
    return remove(path);
}

int veritate_rename(const char* from, const char* to) {
    return rename(from, to);
}

int veritate_touch(const char* path) {
    return utimensat(AT_FDCWD, path, NULL, 0);
}

int veritate_mkdir(const char* path) {
    if (mkdir(path, FS_DIR_MODE) == 0) return 0;
    return errno == EEXIST ? 0 : -1;
}

#endif
