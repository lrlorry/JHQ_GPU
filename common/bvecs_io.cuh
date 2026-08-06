#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Streaming reader for BIGANN/SIFT .bvecs files. Each record is:
//   int32 dimension, followed by dimension uint8 coordinates.
// Coordinates are exposed as signed bytes after subtracting 128. Applying
// the same shift to base and query vectors preserves squared L2 exactly.
struct BVecsReader {
    FILE* f = nullptr;
    long long npts = 0;
    int dim = 0;
    long long record_bytes = 0;

    bool open(const char* path) {
        close();
        f = std::fopen(path, "rb");
        if (!f) return false;

        int32_t d = 0;
        if (std::fread(&d, sizeof(d), 1, f) != 1 || d <= 0) {
            close();
            return false;
        }
        dim = static_cast<int>(d);
        record_bytes = static_cast<long long>(sizeof(int32_t)) + dim;

        if (std::fseek(f, 0, SEEK_END) != 0) {
            close();
            return false;
        }
        const long end = std::ftell(f);
        if (end < 0 || static_cast<long long>(end) % record_bytes != 0) {
            close();
            return false;
        }
        npts = static_cast<long long>(end) / record_bytes;
        std::rewind(f);
        return true;
    }

    int read_batch(long long start, int count, std::vector<int8_t>& out) {
        out.clear();
        if (!f || start < 0 || start >= npts || count <= 0) return 0;

        const int actual = static_cast<int>(
            std::min<long long>(count, npts - start));
        const size_t bytes = static_cast<size_t>(actual) *
                             static_cast<size_t>(record_bytes);
        std::vector<uint8_t> records(bytes);

        const long long offset = start * record_bytes;
        if (std::fseek(f, static_cast<long>(offset), SEEK_SET) != 0) return 0;
        const size_t got = std::fread(records.data(), 1, bytes, f);
        const int got_records = static_cast<int>(got / record_bytes);
        out.resize(static_cast<size_t>(got_records) * dim);

        for (int i = 0; i < got_records; ++i) {
            const uint8_t* rec = records.data() +
                                 static_cast<size_t>(i) * record_bytes;
            int32_t record_dim = 0;
            std::memcpy(&record_dim, rec, sizeof(record_dim));
            if (record_dim != dim) {
                out.resize(static_cast<size_t>(i) * dim);
                return i;
            }
            const uint8_t* src = rec + sizeof(int32_t);
            int8_t* dst = out.data() + static_cast<size_t>(i) * dim;
            for (int j = 0; j < dim; ++j)
                dst[j] = static_cast<int8_t>(static_cast<int>(src[j]) - 128);
        }
        return got_records;
    }

    void close() {
        if (f) std::fclose(f);
        f = nullptr;
        npts = 0;
        dim = 0;
        record_bytes = 0;
    }

    ~BVecsReader() { close(); }
};

