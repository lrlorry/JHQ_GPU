// Export the trained coarse quantiser, so the CPU reference searches the same
// partitioning this GPU index does.
//
// JHQ_official reads its IVF centroids and assignments from
//   <name>_centroid_<nlist>.fvecs   and   <name>_cluster_id_<nlist>.ivecs
// rather than training its own. That is what makes a fair CPU-versus-GPU
// comparison possible: hand both implementations the same routing and the only
// thing left between them is the quantisation and the search.
//
// Without this the comparison measures training instead. JHQ_repro, for one,
// trains its IVF on at most 65,536 points with 20 iterations and its residual
// codebook with 25, where this port uses over a million points and 2000 --
// results/v34_lloyd_iterations/ measures that 25 is 7e-3 of recall short of
// the plateau. A CPU number taken against that state would be a statement
// about how long each side trained.
//
// The centroids are in the rotated space: the index trains them on Pi*x. The
// CPU side must apply the same Pi, which the cached JL state carries.
#ifndef JHQ_INDEX_HEADER
#define JHQ_INDEX_HEADER "jhq_v58_export/jhq_gpu_index.cuh"
#endif
#include JHQ_INDEX_HEADER
#include "common/fvecs_io.cuh"
#include "common/fvecs_mmap_io.cuh"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace jhq_gpu;

// common/fvecs_io.cuh writes ivecs but not fvecs, and adding one there would
// touch a header nine demos include. Same format: a 4-byte dimension in front
// of every row.
static bool write_fvecs_local(const char* path, const float* v, int n, int dim) {
    std::FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    for (int i = 0; i < n; ++i) {
        if (std::fwrite(&dim, sizeof(int), 1, f) != 1 ||
            std::fwrite(v + (size_t)i * dim, sizeof(float), dim, f) != (size_t)dim) {
            std::fclose(f); return false;
        }
    }
    return std::fclose(f) == 0;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
            "Usage: %s <base.fvecs> <out_dir> <name> <M> <nlist> [B] [Br] [n_train]\n"
            "Writes <out_dir>/<name>_centroid_<nlist>.fvecs\n"
            "   and <out_dir>/<name>_cluster_id_<nlist>.ivecs\n", argv[0]);
        return 1;
    }
    const char* base_path = argv[1];
    const std::string out_dir = argv[2], name = argv[3];
    const int M = std::atoi(argv[4]), nlist = std::atoi(argv[5]);
    const int B  = (argc > 6) ? std::atoi(argv[6]) : 8;
    const int Br = (argc > 7) ? std::atoi(argv[7]) : 8;

    MmapFloatMatrix base = load_fvecs_mmap(base_path);
    const int nb = base.n, d = base.d;
    int n_train = (argc > 8) ? std::atoi(argv[8]) : std::min(nb, 100000);
    std::printf("base=%dx%d  M=%d  nlist=%d  n_train=%d\n", nb, d, M, nlist, n_train);

    // d is a constructor argument, not a Params field -- demo_jhq_v36 does
    // JHQGpuIndex idx(d, p).
    JHQGpuIndex::Params p;
    p.M = M; p.B = B; p.Br = Br;
    p.nlist = nlist; p.nprobe = 8; p.alpha = 100.0f;
    JHQGpuIndex idx(d, p);
    // train() honours JHQ_INDEX_CACHE, so pointing this at the same cache the
    // measured runs used exports the centroids those runs actually searched,
    // rather than a fresh set that would differ in its last bits.
    idx.train(base.data, n_train);
    idx.add(base.data, nb);

    const std::vector<float>& c = idx.centroids();
    if ((int)c.size() != nlist * d) {
        std::fprintf(stderr, "centroids are %zu floats, expected %d\n",
                     c.size(), nlist * d);
        return 1;
    }
    const std::string cpath = out_dir + "/" + name + "_centroid_"
                            + std::to_string(nlist) + ".fvecs";
    if (!write_fvecs_local(cpath.c_str(), c.data(), nlist, d)) {
        std::fprintf(stderr, "could not write %s\n", cpath.c_str());
        return 1;
    }
    std::printf("wrote %s  (%d x %d)\n", cpath.c_str(), nlist, d);

    std::vector<int> assign;
    idx.vector_lists(assign);
    if ((int)assign.size() != nb) {
        std::fprintf(stderr, "assignments are %zu, expected %d\n", assign.size(), nb);
        return 1;
    }
    const std::string apath = out_dir + "/" + name + "_cluster_id_"
                            + std::to_string(nlist) + ".ivecs";
    write_ivecs(apath.c_str(), assign.data(), nb, 1);
    std::printf("wrote %s  (%d x 1)\n", apath.c_str(), nb);

    // A partitioning with an empty list, or one holding most of the data, is
    // worth knowing about before the CPU side spends an hour on it.
    std::vector<int> hist(nlist, 0);
    for (int a : assign) if (a >= 0 && a < nlist) hist[a]++;
    int empty = 0, mx = 0;
    for (int h : hist) { if (h == 0) ++empty; if (h > mx) mx = h; }
    std::printf("lists: %d empty, largest holds %d of %d (%.2f%%)\n",
                empty, mx, nb, 100.0 * mx / nb);
    return 0;
}
