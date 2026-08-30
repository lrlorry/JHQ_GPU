#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>

// Read an .fvecs file into a flat float vector.
// Returns number of vectors; sets dim.
inline int read_fvecs(const char* path, std::vector<float>& out, int& dim) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }

    int d;
    fread(&d, sizeof(int), 1, f);
    rewind(f);

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);

    int n = (int)(sz / (sizeof(int) + (long)d * sizeof(float)));
    dim = d;
    out.resize((size_t)n * d);

    for (int i = 0; i < n; i++) {
        int dd;
        fread(&dd, sizeof(int), 1, f);
        fread(out.data() + (size_t)i * d, sizeof(float), d, f);
    }
    fclose(f);
    return n;
}

// Read an .ivecs file (ground-truth ids).
inline int read_ivecs(const char* path, std::vector<int>& out, int& dim) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }

    int d;
    fread(&d, sizeof(int), 1, f);
    rewind(f);

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);

    int n = (int)(sz / ((long)(sizeof(int) + (long)d * sizeof(int))));
    dim = d;
    out.resize((size_t)n * d);

    for (int i = 0; i < n; i++) {
        int dd;
        fread(&dd, sizeof(int), 1, f);
        fread(out.data() + (size_t)i * d, sizeof(int), d, f);
    }
    fclose(f);
    return n;
}

// Write a flat int vector as .ivecs -- same layout read_ivecs expects, so a
// dumped neighbour list can be fed back through the normal readers.
// Used by demo_jhq_v15_eval_fix to persist returned ids alongside each run:
// with the ids on disk, changing the evaluation metric never costs a re-search
// again (see common/recall.cuh for why that mattered).
inline void write_ivecs(const char* path, const int* data, int n, int dim) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); exit(1); }
    for (int i = 0; i < n; i++) {
        fwrite(&dim, sizeof(int), 1, f);
        fwrite(data + (size_t)i * dim, sizeof(int), dim, f);
    }
    fclose(f);
}
