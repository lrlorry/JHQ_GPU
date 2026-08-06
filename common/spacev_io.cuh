#pragma once
// common/spacev_io.cuh
//
// I/O utilities for the big-ann-benchmarks-style SPACEV-100M dataset:
//   - I8BinReader: streaming reader for base.i8bin / query.i8bin
//       (8-byte header: int32 npts, int32 dim; then npts*dim raw int8, no padding)
//   - load_id_map: loads ids.NNNM.i32bin (local -> global id mapping)
//   - global_to_local: binary search over the (sorted) id map
//   - build_restricted_gt: restricts the official (full-corpus) ground truth
//       to the ids actually present in our subset, translating global ids to
//       local indices and tracking per-query survivor counts.
//
// This header intentionally does NOT depend on fvecs_io.cuh — SPACEV uses a
// different binary layout (no per-vector int32 dim prefix, int8 payload).
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>

// Streaming reader for the big-ann-benchmarks-style i8bin format:
// 8-byte header (int32 npts, int32 dim), then npts*dim raw int8, no padding.
struct I8BinReader {
    FILE* f = nullptr;
    int npts = 0, dim = 0;
    long long data_offset = 8;

    // Opens `path`, reads the 8-byte header. Returns false on failure.
    bool open(const char* path) {
        f = fopen(path, "rb");
        if (!f) return false;
        int32_t npts_i32 = 0, dim_i32 = 0;
        if (fread(&npts_i32, sizeof(int32_t), 1, f) != 1 ||
            fread(&dim_i32,  sizeof(int32_t), 1, f) != 1) {
            fclose(f); f = nullptr;
            return false;
        }
        npts = (int)npts_i32;
        dim  = (int)dim_i32;
        data_offset = 8;
        return true;
    }

    // Read `count` vectors starting at vector index `start` into `out`
    // (resized to actual_count*dim int8_t). Returns actual vectors read (may
    // be less than count near EOF, or 0 if start is out of range).
    int read_batch(long long start, int count, std::vector<int8_t>& out) {
        if (!f || dim <= 0) { out.clear(); return 0; }
        if (start < 0 || start >= (long long)npts || count <= 0) { out.clear(); return 0; }
        long long avail = (long long)npts - start;
        long long want  = std::min((long long)count, avail);
        out.resize((size_t)(want * dim));
        long long byte_off = data_offset + start * (long long)dim;
        // fseek's offset is a `long`; on the target 64-bit Linux build server
        // `long` is 8 bytes, so this covers the full ~10GB SPACEV-100M base file.
        if (fseek(f, (long)byte_off, SEEK_SET) != 0) { out.clear(); return 0; }
        size_t nread = fread(out.data(), 1, (size_t)(want * dim), f);
        long long actual_vecs = (long long)(nread / (size_t)dim);
        if (actual_vecs < want) out.resize((size_t)(actual_vecs * dim));
        return (int)actual_vecs;
    }

    void close() { if (f) { fclose(f); f = nullptr; } }
    ~I8BinReader() { close(); }
};

// Loads ids.NNNM.i32bin fully: local_to_global[i] = global id for local
// index i. Header is 8 bytes (two int32: npts, then an unused/ignorable
// second field), followed by npts little-endian int32 global ids.
// Verifies (linear scan) strictly increasing order; prints a warning (does
// not throw/abort) if violated anywhere.
inline std::vector<int32_t> load_id_map(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "load_id_map: cannot open %s\n", path);
        exit(1);
    }
    int32_t npts_i32 = 0, unused_i32 = 0;
    if (fread(&npts_i32, sizeof(int32_t), 1, f) != 1 ||
        fread(&unused_i32, sizeof(int32_t), 1, f) != 1) {
        fprintf(stderr, "load_id_map: failed to read header from %s\n", path);
        fclose(f);
        exit(1);
    }
    (void)unused_i32;
    long long npts = (long long)npts_i32;
    std::vector<int32_t> ids((size_t)npts);
    size_t nread = fread(ids.data(), sizeof(int32_t), (size_t)npts, f);
    fclose(f);
    if (nread != (size_t)npts) {
        fprintf(stderr,
                "load_id_map: WARNING short read on %s (expected %lld ids, got %zu) — "
                "truncating to what was read\n",
                path, npts, nread);
        ids.resize(nread);
    }

    bool increasing = true;
    long long first_violation = -1;
    for (size_t i = 1; i < ids.size(); i++) {
        if (ids[i] <= ids[i - 1]) {
            increasing = false;
            first_violation = (long long)i;
            break;
        }
    }
    if (!increasing) {
        fprintf(stderr,
                "load_id_map: WARNING local_to_global loaded from %s is NOT strictly "
                "increasing (first violation at local index %lld) — continuing anyway; "
                "global_to_local binary search assumes sortedness and may be unreliable "
                "near non-monotonic regions\n",
                path, first_violation);
    }
    return ids;
}

// Binary-search global_id in a sorted local_to_global map. Returns local
// index if present, -1 otherwise. O(log n).
inline int global_to_local(const std::vector<int32_t>& local_to_global, int32_t global_id) {
    auto it = std::lower_bound(local_to_global.begin(), local_to_global.end(), global_id);
    if (it != local_to_global.end() && *it == global_id)
        return (int)(it - local_to_global.begin());
    return -1;
}

// Reads groundtruth.NNK.i32bin (header npts,K then npts*K int32 global ids,
// original per-query order preserved) and restricts each query's candidate
// list against local_to_global, producing per-query local-index top-k lists
// plus survivor-count diagnostics.
struct RestrictedGT {
    int nq = 0, k = 0;
    std::vector<int32_t> ids;           // [nq * k], -1 for unfilled slots
    std::vector<int32_t> num_survivors; // [nq], raw survivor count (before truncation to k)
};

// gt_path: path to groundtruth.NNK.i32bin.
// local_to_global: sorted id map from load_id_map(), used for membership testing.
// k: restricted top-k size to keep per query (<= the file's own K).
// max_queries: number of queries (from the front of the file) to process;
//              -1 means "all npts queries in the file".
inline RestrictedGT build_restricted_gt(const char* gt_path,
                                         const std::vector<int32_t>& local_to_global,
                                         int k, int max_queries) {
    FILE* f = fopen(gt_path, "rb");
    if (!f) {
        fprintf(stderr, "build_restricted_gt: cannot open %s\n", gt_path);
        exit(1);
    }
    int32_t npts_i32 = 0, K_i32 = 0;
    if (fread(&npts_i32, sizeof(int32_t), 1, f) != 1 ||
        fread(&K_i32, sizeof(int32_t), 1, f) != 1) {
        fprintf(stderr, "build_restricted_gt: failed to read header from %s\n", gt_path);
        fclose(f);
        exit(1);
    }
    int npts_file = (int)npts_i32;
    int Kfile     = (int)K_i32;
    int nq = (max_queries < 0) ? npts_file : std::min(max_queries, npts_file);

    RestrictedGT out;
    out.nq = nq;
    out.k  = k;
    out.ids.assign((size_t)nq * (size_t)k, -1);
    out.num_survivors.assign((size_t)nq, 0);

    std::vector<int32_t> row((size_t)Kfile);
    for (int qi = 0; qi < nq; qi++) {
        size_t nread = fread(row.data(), sizeof(int32_t), (size_t)Kfile, f);
        if (nread != (size_t)Kfile) {
            fprintf(stderr,
                    "build_restricted_gt: WARNING short read for query %d in %s "
                    "(expected %d, got %zu) — remaining slots treated as absent\n",
                    qi, gt_path, Kfile, nread);
            std::fill(row.begin() + nread, row.end(), (int32_t)-1);
        }
        int32_t* dst = out.ids.data() + (size_t)qi * (size_t)k;
        int survivors = 0;
        for (int c = 0; c < Kfile; c++) {
            int32_t gid = row[c];
            if (gid < 0) continue; // defensive: skip short-read padding
            int local = global_to_local(local_to_global, gid);
            if (local < 0) continue; // not present in our subset
            if (survivors < k) dst[survivors] = local;
            survivors++;
        }
        out.num_survivors[qi] = survivors;
    }
    fclose(f);
    return out;
}
