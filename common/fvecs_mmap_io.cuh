#pragma once
// mmap-backed .fvecs loader for the JHQ GPU demos.
//
// read_fvecs() (fvecs_io.cuh) loads the whole base set into a heap
// std::vector<float> -- fine for small datasets, but on a memory-constrained
// box (this project hit a container with a 2GB cgroup memory.max) it OOMs
// well before the file itself is anywhere near that size, since the RAM
// requirement is the FULL n*d*4 bytes, resident, for the whole run.
//
// load_fvecs_mmap() instead:
//   1. Streams the source .fvecs file (format: [int32 dim][float32 x dim]
//      per record) into a header-free packed float32 sibling file
//      (<path>.raw_f32), using a small bounded host buffer -- never holds
//      more than a few thousand rows in RAM at once. Cached: a second call
//      against the same .fvecs skips reconversion if a correctly-sized
//      .raw_f32 already exists.
//   2. mmap()s that clean file PROT_READ. Because it's header-free, the
//      mapping IS a valid n*d contiguous float array -- the exact shape
//      JHQGpuIndex::add()/train() expect -- so no second host-side copy is
//      needed. The backing pages come from the OS page cache (demand-paged
//      from disk as add()'s internal cudaMemcpy reads through them, and
//      reclaimable under memory pressure) instead of a committed heap
//      allocation sized to the whole dataset.
//
// This does NOT fix GPU-side memory: JHQGpuIndex::rotate_on_gpu() still
// cudaMallocs two full n*d float buffers and add() still requires the whole
// dataset in one call (its IVF list construction needs global cluster-
// assignment counts). For datasets whose n*d*4*2 bytes exceed the GPU's
// VRAM, this loader alone is not enough -- add() itself would need a
// genuinely batched/two-pass rewrite, which is separate, larger work.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>
#include <algorithm>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct MmapFloatMatrix {
    const float* data = nullptr;
    int n = 0, d = 0;
    void* map_base = nullptr;
    size_t map_bytes = 0;
    int fd = -1;

    MmapFloatMatrix() = default;
    MmapFloatMatrix(const MmapFloatMatrix&) = delete;
    MmapFloatMatrix& operator=(const MmapFloatMatrix&) = delete;
    MmapFloatMatrix(MmapFloatMatrix&& o) noexcept { *this = std::move(o); }
    MmapFloatMatrix& operator=(MmapFloatMatrix&& o) noexcept {
        if (this != &o) {
            reset();
            data = o.data; n = o.n; d = o.d;
            map_base = o.map_base; map_bytes = o.map_bytes; fd = o.fd;
            o.data = nullptr; o.map_base = nullptr; o.map_bytes = 0; o.fd = -1;
        }
        return *this;
    }
    ~MmapFloatMatrix() { reset(); }

    void reset() {
        if (map_base) { munmap(map_base, map_bytes); map_base = nullptr; }
        if (fd >= 0) { close(fd); fd = -1; }
    }
};

inline MmapFloatMatrix load_fvecs_mmap(const char* fvecs_path,
                                        size_t convert_chunk_rows = 20000) {
    FILE* f = fopen(fvecs_path, "rb");
    if (!f) throw std::runtime_error(std::string("cannot open ") + fvecs_path);

    int32_t d = 0;
    if (fread(&d, sizeof(int32_t), 1, f) != 1 || d <= 0)
        throw std::runtime_error(std::string("empty/corrupt fvecs: ") + fvecs_path);
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    long rec_bytes = (long)sizeof(int32_t) + (long)d * (long)sizeof(float);
    long n = fsz / rec_bytes;

    std::string raw_path = std::string(fvecs_path) + ".raw_f32";
    long want_bytes = n * (long)d * (long)sizeof(float);

    struct stat st;
    bool need_convert = !(stat(raw_path.c_str(), &st) == 0 &&
                           (long)st.st_size == want_bytes);

    if (need_convert) {
        printf("  [mmap loader] converting %s -> %s (%ld x %d, streamed, "
               "%zu rows/chunk, no full-array buffering)\n",
               fvecs_path, raw_path.c_str(), n, (int)d, convert_chunk_rows);
        FILE* out = fopen(raw_path.c_str(), "wb");
        if (!out) throw std::runtime_error("cannot create " + raw_path);
        std::vector<float> buf((size_t)convert_chunk_rows * d);
        long done = 0;
        while (done < n) {
            long take = std::min((long)convert_chunk_rows, n - done);
            for (long i = 0; i < take; i++) {
                int32_t dd;
                if (fread(&dd, sizeof(int32_t), 1, f) != 1 || dd != d)
                    throw std::runtime_error("corrupt/mismatched fvecs record in " +
                                              std::string(fvecs_path));
                if (fread(buf.data() + (size_t)i * d, sizeof(float), (size_t)d, f)
                        != (size_t)d)
                    throw std::runtime_error("truncated fvecs record in " +
                                              std::string(fvecs_path));
            }
            fwrite(buf.data(), sizeof(float), (size_t)take * d, out);
            done += take;
        }
        fclose(out);
    } else {
        printf("  [mmap loader] reusing cached %s\n", raw_path.c_str());
    }
    fclose(f);

    MmapFloatMatrix m;
    m.n = (int)n; m.d = d;
    m.fd = open(raw_path.c_str(), O_RDONLY);
    if (m.fd < 0) throw std::runtime_error("cannot open " + raw_path);
    m.map_bytes = (size_t)want_bytes;
    void* p = mmap(nullptr, m.map_bytes, PROT_READ, MAP_SHARED, m.fd, 0);
    if (p == MAP_FAILED) throw std::runtime_error("mmap failed for " + raw_path);
#ifdef MADV_SEQUENTIAL
    // add()'s internal cudaMemcpy reads this pointer sequentially once --
    // hint the kernel so it read-aheads and evicts-behind instead of
    // caching the whole mapping resident, which is the whole point.
    madvise(p, m.map_bytes, MADV_SEQUENTIAL);
#endif
    m.map_base = p;
    m.data = (const float*)p;
    return m;
}
