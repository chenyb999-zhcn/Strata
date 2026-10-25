// src/platform/memory.cpp - see include/strata/platform/memory.hpp.
#include "strata/platform/memory.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi1_4.h>
#include <cstring>
#else
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>
#endif

namespace strata::platform {

#if defined(_WIN32)
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    HANDLE self = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) {
        r.note = "GetProcessWorkingSetSizeEx failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    // Locked pages count against the minimum working set, so it must grow by the region plus headroom for the
    // rest of the process. Soft limits: the maximum is not enforced, only the minimum is raised.
    const SIZE_T margin = (SIZE_T) 512 << 20;
    const SIZE_T new_min = min_ws + (SIZE_T) bytes + margin;
    const SIZE_T new_max = max_ws > new_min + margin ? max_ws : new_min + margin;
    if (!SetProcessWorkingSetSizeEx(self, new_min, new_max,
                                    QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        r.note = "SetProcessWorkingSetSizeEx(" + std::to_string((unsigned long long) (new_min >> 20)) +
                 " MiB) failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    const uint64_t chunk = 1ull << 30;
    uint8_t* base = (uint8_t*) p;
    for (uint64_t off = 0; off < bytes; off += chunk) {
        const uint64_t n = bytes - off < chunk ? bytes - off : chunk;
        if (!VirtualLock(base + off, (SIZE_T) n)) {
            r.note = "VirtualLock stopped at " + std::to_string((unsigned long long) (off >> 20)) + " of " +
                     std::to_string((unsigned long long) (bytes >> 20)) + " MiB (error " +
                     std::to_string(GetLastError()) + ")";
            r.ok = off > 0;
            return r;
        }
        r.locked_bytes = off + n;
    }
    r.ok = true;
    r.note = "locked " + std::to_string((unsigned long long) (bytes >> 20)) + " MiB via working-set minimum + VirtualLock";
    return r;
}

LockResult lock_resident_ranges(const std::vector<std::pair<void*, uint64_t>>& regions) {
    LockResult r;
    uint64_t total = 0;
    for (const auto& g : regions) total += g.second;
    if (total == 0) { r.note = "nothing to lock"; return r; }
    HANDLE self = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) {
        r.note = "GetProcessWorkingSetSizeEx failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    // same growth as lock_resident, but for the sum of the regions and with ONE margin
    const SIZE_T margin = (SIZE_T) 512 << 20;
    const SIZE_T new_min = min_ws + (SIZE_T) total + margin;
    const SIZE_T new_max = max_ws > new_min + margin ? max_ws : new_min + margin;
    if (!SetProcessWorkingSetSizeEx(self, new_min, new_max,
                                    QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        r.note = "SetProcessWorkingSetSizeEx(" + std::to_string((unsigned long long) (new_min >> 20)) +
                 " MiB) failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    const uint64_t chunk = 1ull << 30;
    for (const auto& g : regions) {
        uint8_t* base = (uint8_t*) g.first;
        for (uint64_t off = 0; off < g.second; off += chunk) {
            const uint64_t n = g.second - off < chunk ? g.second - off : chunk;
            if (!VirtualLock(base + off, (SIZE_T) n)) {
                r.note = "VirtualLock stopped after " + std::to_string((unsigned long long) (r.locked_bytes >> 20)) +
                         " of " + std::to_string((unsigned long long) (total >> 20)) + " MiB (error " +
                         std::to_string(GetLastError()) + ")";
                r.ok = r.locked_bytes > 0;
                return r;
            }
            r.locked_bytes += n;
        }
    }
    r.ok = true;
    r.note = "locked " + std::to_string((unsigned long long) (total >> 20)) + " MiB in " +
             std::to_string(regions.size()) + " regions via working-set minimum + VirtualLock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0) return;
    const uint64_t chunk = 1ull << 30;
    for (uint64_t off = 0; off < bytes; off += chunk)
        VirtualUnlock((uint8_t*) p + off, (SIZE_T) (bytes - off < chunk ? bytes - off : chunk));
}

bool gpu_shared_memory_budget(const void* luid, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    // dxgi.dll is loaded when asked, not linked: a start that never needs this keeps the imports it had
    HMODULE dxgi = LoadLibraryA("dxgi.dll");
    if (dxgi == nullptr) { why = "dxgi.dll not found"; return false; }
    using CreateFactory = HRESULT(WINAPI*)(REFIID, void**);
    const auto create = (CreateFactory) (void*) GetProcAddress(dxgi, "CreateDXGIFactory1");
    IDXGIFactory1* factory = nullptr;
    if (create == nullptr || FAILED(create(__uuidof(IDXGIFactory1), (void**) &factory)) || factory == nullptr) {
        why = "CreateDXGIFactory1 failed";
        FreeLibrary(dxgi);
        return false;
    }
    bool ok = false;
    why = "no DXGI adapter has the CUDA device's LUID";
    for (UINT i = 0; !ok; ++i) {
        IDXGIAdapter1* a = nullptr;
        if (factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND || a == nullptr) break;
        DXGI_ADAPTER_DESC1 d{};
        if (SUCCEEDED(a->GetDesc1(&d)) && std::memcmp(&d.AdapterLuid, luid, sizeof d.AdapterLuid) == 0) {
            IDXGIAdapter3* a3 = nullptr;
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (SUCCEEDED(a->QueryInterface(__uuidof(IDXGIAdapter3), (void**) &a3)) && a3 != nullptr &&
                SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info))) {
                budget = info.Budget;
                usage = info.CurrentUsage;
                ok = budget > 0;
                why = ok ? "" : "the adapter reports no shared-memory budget";
            } else {
                why = "QueryVideoMemoryInfo failed";
            }
            if (a3 != nullptr) a3->Release();
            a->Release();
            break;
        }
        a->Release();
    }
    factory->Release();
    FreeLibrary(dxgi);
    return ok;
}

uint64_t total_physical_memory() {
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullTotalPhys : 0;
}

bool read_ahead_enabled() { return false; }
void advise_willneed(const void*, uint64_t) {}
void advise_willneed(int, uint64_t, uint64_t) {}
void populate_read(const MappedRange*, unsigned, unsigned) {}
void populate_read(const void*, uint64_t, unsigned) {}
bool read_range_direct(const std::string&, uint64_t, uint64_t, void*, unsigned, unsigned) { return false; }
double resident_fraction(const void*, uint64_t) { return 1.0; }   // no mincore: keep the caller's default path
#else
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    if (mlock(p, bytes) != 0) { r.note = "mlock failed (raise ulimit -l)"; return r; }
    r.ok = true;
    r.locked_bytes = bytes;
    r.note = "mlock";
    return r;
}

LockResult lock_resident_ranges(const std::vector<std::pair<void*, uint64_t>>& regions) {
    LockResult r;
    for (const auto& g : regions) {
        if (g.first == nullptr || g.second == 0) continue;
        if (mlock(g.first, g.second) != 0) { r.note = "mlock failed (raise ulimit -l)"; r.ok = r.locked_bytes > 0; return r; }
        r.locked_bytes += g.second;
    }
    r.ok = r.locked_bytes > 0;
    r.note = r.ok ? "mlock of " + std::to_string(regions.size()) + " regions" : "nothing to lock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p != nullptr && bytes != 0) munlock(p, bytes);
}

bool gpu_shared_memory_budget(const void*, uint64_t& budget, uint64_t& usage, std::string& why) {
    budget = usage = 0;
    why = "DXGI is Windows-only";
    return false;
}

uint64_t total_physical_memory() {
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? (uint64_t) pages * (uint64_t) page : 0;
}

namespace {
// A read-ahead advise is asynchronous but each call only queues one window, and the consumer
// waits for it: 128 KiB steps made a 3.6 GiB startup load queue ~29k windows one after another
// (60 MB/s on an NVMe that does 1.4 GB/s with a deep queue).  Ask for 4 MiB at a time instead:
// the window cap still applies per call, but the calls no longer serialize the load.
constexpr uint64_t kAdviseStep = 4ull << 20;
}

bool read_ahead_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_READ_AHEAD");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
}

void advise_willneed(const void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0 || !read_ahead_enabled()) return;
    const long ps = sysconf(_SC_PAGE_SIZE);
    const uintptr_t pg = ps > 0 ? (uintptr_t) ps : 4096, end = (uintptr_t) p + bytes;
    for (uintptr_t a = (uintptr_t) p & ~(pg - 1); a < end; a += kAdviseStep)
        (void) madvise((void*) a, (size_t) std::min<uintptr_t>(kAdviseStep, end - a), MADV_WILLNEED);
}

void advise_willneed(int fd, uint64_t offset, uint64_t bytes) {
    if (fd < 0 || !read_ahead_enabled()) return;
    for (uint64_t at = 0; at < bytes; at += kAdviseStep)
#if defined(__APPLE__)   // macOS: F_RDADVISE is the read-ahead hint
    {
        radvisory ra{(off_t) (offset + at), (int) std::min(kAdviseStep, bytes - at)};
        (void) fcntl(fd, F_RDADVISE, &ra);
    }
#else
        (void) posix_fadvise(fd, (off_t) (offset + at), (off_t) std::min(kAdviseStep, bytes - at), POSIX_FADV_WILLNEED);
#endif
}

#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22   // Linux 5.14; a build host without the header definition still gets the advise
#endif

void populate_read(const void* p, uint64_t bytes, unsigned threads) {
    const MappedRange one{p, bytes};
    populate_read(&one, 1, threads);
}

void populate_read(const MappedRange* ranges, unsigned count, unsigned threads) {
    if (ranges == nullptr || count == 0) return;
    const long ps = sysconf(_SC_PAGE_SIZE);
    const uintptr_t page = ps > 0 ? (uintptr_t) ps : 4096;
    // Probe one page first: a kernel without the advice fails at once, and a hint is all that is left.
    uintptr_t first = 0;
    for (unsigned i = 0; i < count && first == 0; ++i)
        if (ranges[i].p != nullptr && ranges[i].bytes != 0) first = ((uintptr_t) ranges[i].p) & ~(page - 1);
    if (first == 0) return;
    if (madvise((void*) first, (size_t) page, MADV_POPULATE_READ) != 0) {
        for (unsigned i = 0; i < count; ++i) advise_willneed(ranges[i].p, ranges[i].bytes);
        return;
    }
    constexpr uintptr_t kPiece = 256ull << 20;
    std::vector<std::pair<uintptr_t, uintptr_t>> work;
    for (unsigned i = 0; i < count; ++i) {
        if (ranges[i].p == nullptr || ranges[i].bytes == 0) continue;
        const uintptr_t a = ((uintptr_t) ranges[i].p) & ~(page - 1);
        const uintptr_t b = (uintptr_t) ranges[i].p + (uintptr_t) ranges[i].bytes;
        for (uintptr_t at = a; at < b; at += kPiece) work.emplace_back(at, std::min(b, at + kPiece));
    }
    if (work.empty()) return;
    std::atomic<unsigned> next{0};
    auto run = [&] {
        for (unsigned i; (i = next.fetch_add(1)) < (unsigned) work.size();) {
            const uintptr_t a = work[i].first, b = work[i].second;
            (void) madvise((void*) a, (size_t) (b - a), MADV_WILLNEED);        // queue this piece's read-ahead
            (void) madvise((void*) a, (size_t) (b - a), MADV_POPULATE_READ);    // and fault it in
        }
    };
    unsigned n = threads < 1 ? 1u : threads;
    if (n > (unsigned) work.size()) n = (unsigned) work.size();
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < n; ++t) pool.emplace_back(run);
    run();
    for (auto& t : pool) t.join();
}
#endif

ProcIo proc_io_sample() {
    ProcIo r;
#if defined(__linux__)
    if (std::FILE* f = std::fopen("/proc/self/io", "r")) {
        char key[64];
        unsigned long long v;
        while (std::fscanf(f, "%63[^:]: %llu\n", key, &v) == 2)
            if (std::strcmp(key, "read_bytes") == 0) { r.read_bytes = v; r.valid = true; }
        std::fclose(f);
    }
    if (std::FILE* f = std::fopen("/proc/self/stat", "r")) {
        char buf[1024];
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        std::fclose(f);
        if (const char* p = std::strrchr(buf, ')')) {   // fields after the command: state ppid ... minflt cminflt majflt
            unsigned long long minflt = 0, cminflt = 0, majflt = 0;
            if (std::sscanf(p + 2, "%*c %*d %*d %*d %*d %*d %*u %llu %llu %llu", &minflt, &cminflt, &majflt) == 3)
                r.major_faults = majflt;
        }
    }
#endif
    return r;
}

}  // namespace strata::platform

#if !defined(_WIN32)
namespace strata::platform {
namespace {
// One O_DIRECT fd per model file, kept for the life of the process: the dense loader asks for one tensor at a
// time (495 of them on the shard of #4064), and an open/close pair per tensor cost more than the read itself.
std::mutex g_direct_mu;
std::map<std::string, int> g_direct_fds;
int direct_fd(const std::string& path) {
    std::lock_guard<std::mutex> lk(g_direct_mu);
    auto it = g_direct_fds.find(path);
    if (it != g_direct_fds.end()) return it->second;
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd >= 0) g_direct_fds.emplace(path, fd);
    return fd;
}
}  // namespace

bool read_range_direct(const std::string& path, uint64_t off, uint64_t len, void* dst, unsigned threads,
                       unsigned req_bytes) {
    if (len == 0) return true;
    if (dst == nullptr || (off % 512) != 0 || (len % 512) != 0 || (((uintptr_t) dst) % 512) != 0) return false;
    const int fd = direct_fd(path);
    if (fd < 0) return false;
    const uint64_t req = req_bytes < 4096 ? 4096 : req_bytes;
    std::atomic<uint64_t> next{0};
    std::atomic<bool> ok{true};
    auto run = [&] {
        for (uint64_t at; (at = next.fetch_add(req)) < len;) {
            const uint64_t n = len - at < req ? len - at : req;
            uint8_t* p = (uint8_t*) dst + at;
            uint64_t done = 0;
            while (done < n) {
                const ssize_t got = ::pread(fd, p + done, (size_t) (n - done), (off_t) (off + at + done));
                if (got <= 0) { ok.store(false); return; }
                done += (uint64_t) got;
            }
        }
    };
    const uint64_t requests = (len + req - 1) / req;
    unsigned n = threads < 1 ? 1u : threads;
    if ((uint64_t) n > requests) n = (unsigned) requests;
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < n; ++t) pool.emplace_back(run);
    run();
    for (auto& th : pool) th.join();
    return ok.load();
}
double resident_fraction(const void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0) return 0.0;
    const long ps = sysconf(_SC_PAGE_SIZE);
    const uint64_t page = ps > 0 ? (uint64_t) ps : 4096;
    const uint64_t pages = (bytes + page - 1) / page;
    std::vector<unsigned char> vec((size_t) ((pages + 7) / 8), 0);
    if (mincore((void*) p, (size_t) bytes, vec.data()) != 0) return 0.0;
    const uint64_t step_pages = (1ull << 20) / page;      // one sample per MiB
    uint64_t seen = 0, hit = 0;
    for (uint64_t i = 0; i < pages; i += step_pages, ++seen)
        if (vec[(size_t) (i / 8)] & (1u << (i % 8))) ++hit;
    return seen == 0 ? 0.0 : (double) hit / (double) seen;
}
}   // namespace strata::platform
#endif
