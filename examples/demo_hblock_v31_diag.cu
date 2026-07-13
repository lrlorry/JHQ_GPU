#include "hblock_v30/jhq_gpu_index.cuh"
#include "common/fvecs_io.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <algorithm>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <base.fvecs> <query.fvecs> <gt.ivecs>\n"
            "         [K1=16] [K2=16] [K3=16] [ck1=2] [ck2=2]\n"
            "         [Kr=16] [batch=1024] [d_proj=64] [per_block_r=16] [km_iters=30]\n"
            "         [graph_degree=32] [graph_depth=64] [entry_per_cell=4]\n"
            "         [n_c2_nbrs=4] [n_c1_nbrs=2] [beam_size=64] [klocal=10] [mini_km_iters=5]\n",
            argv[0]);
        return 1;
    }

    auto base  = load_fvecs(argv[1]);
    auto query = load_fvecs(argv[2]);
    auto gt    = load_ivecs(argv[3]);

    int nb = base.rows, d = base.cols;
    int nq = query.rows;
    int d_gt = gt.cols;
    int k = 10;

    int K1   = (argc > 4)  ? atoi(argv[4])  : 16;
    int K2   = (argc > 5)  ? atoi(argv[5])  : 16;
    int K3   = (argc > 6)  ? atoi(argv[6])  : 16;
    int ck1  = (argc > 7)  ? atoi(argv[7])  : 2;
    int ck2  = (argc > 8)  ? atoi(argv[8])  : 2;
    int Kr   = (argc > 9)  ? atoi(argv[9])  : 16;
    int batch= (argc > 10) ? atoi(argv[10]) : 1024;
    int dprj = (argc > 11) ? atoi(argv[11]) : 64;
    int pbr  = (argc > 12) ? atoi(argv[12]) : 16;
    int kmi  = (argc > 13) ? atoi(argv[13]) : 30;
    int deg  = (argc > 14) ? atoi(argv[14]) : 32;
    int dep  = (argc > 15) ? atoi(argv[15]) : 64;
    int epc  = (argc > 16) ? atoi(argv[16]) : 4;
    int c2n  = (argc > 17) ? atoi(argv[17]) : 4;
    int c1n  = (argc > 18) ? atoi(argv[18]) : 2;
    int beam = (argc > 19) ? atoi(argv[19]) : 64;
    int kloc = (argc > 20) ? atoi(argv[20]) : 10;
    int mki  = (argc > 21) ? atoi(argv[21]) : 5;

    printf("nb=%d  nq=%d  d=%d  d_gt=%d\n", nb, nq, d, d_gt);

    hblock_v30::HBlockIndex::Params p;
    p.K1=K1; p.K2=K2; p.K3=K3; p.Kr=Kr; p.Br=4;
    p.ck1=ck1; p.ck2=ck2; p.ck3=K3;
    p.d_proj=dprj; p.per_block_r=pbr; p.klocal=kloc;
    p.km_iters=kmi; p.batch_size=batch;
    p.graph_degree=deg; p.graph_depth=dep; p.entry_per_cell=epc;
    p.n_c2_nbrs=c2n; p.n_c1_nbrs=c1n;
    p.beam_size=beam; p.mini_km_iters=mki;

    hblock_v30::HBlockIndex idx(d, p);

    int n_train = std::min(nb, 200000);
    printf("Training on %d vectors...\n", n_train);
    idx.train(base.data(), n_train);

    printf("Adding %d vectors...\n", nb);
    idx.add(base.data(), nb);

    printf("\ngraph_depth=%d  beam_size=%d  entry_per_cell=%d\n\n", dep, beam, epc);

    idx.diagnose_missed_gt(query.data(), nq, k, gt.data(), d_gt);

    return 0;
}
