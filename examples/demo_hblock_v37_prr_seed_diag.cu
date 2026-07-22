// demo_hblock_v37_prr_seed_diag.cu
//
// Exact-seed threshold diagnostic driver for hblock_v37_prr. Implements the
// experiment described in HBLOCK_V37_PRR_EXACT_SEED_DIAGNOSTIC_PROMPT.md:
// decide whether a small per-block exact-seed set gives a safe upper threshold
// (tau_seed2) tight enough to make a two-wave certified PRR search worthwhile,
// versus the current tau_U (upper-bound-derived) threshold which is far too loose.
//
// Builds ONE index (CERTIFIED_PRR + VECTOR_LEVEL) and sweeps every diagnostic
// parameter on that same index — no rebuild between controls or ef values.
#include "hblock_v37_prr/jhq_gpu_index.cuh"
#include "common/fvecs_io.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;
using hblock_v37_prr::HBlockIndex;
using hblock_v37_prr::SeedDiagRow;

static double recall_at_k(const int* ids, const int* gt, int nq, int k, int gt_k)
{
    int hits = 0;
    for (int i = 0; i < nq; i++)
        for (int j = 0; j < k; j++) {
            int found = ids[i * k + j];
            for (int g = 0; g < gt_k; g++)
                if (gt[i * gt_k + g] == found) { hits++; break; }
        }
    return (double)hits / (double)(nq * k);
}

// Run a control (FIXED_PER_BLOCK / CORRECTED_FIXED / CERTIFIED_PRR) at a given ef
// on the already-built index and print/report recall — no rebuild.
static void run_control(HBlockIndex& idx, const char* name,
                         const std::vector<float>& query, int nq, int k,
                         const std::vector<int>& gt, int gt_k, int ef, FILE* txt)
{
    std::vector<float> h_dists((long long)nq * k);
    std::vector<int>   h_ids((long long)nq * k);
    auto t0 = Clock::now();
    idx.search(query.data(), nq, k, h_dists.data(), h_ids.data(), ef);
    double ms = Ms(Clock::now() - t0).count();
    double rec = recall_at_k(h_ids.data(), gt.data(), nq, k, gt_k);
    double qps = nq / (ms / 1000.0);
    printf("  [control] %-16s ef=%-4d recall@10=%.4f  qps=%.0f  (%.2f ms)\n",
           name, ef, rec, qps, ms);
    if (txt)
        fprintf(txt, "  [control] %-16s ef=%-4d recall@10=%.4f  qps=%.0f  (%.2f ms)\n",
                name, ef, rec, qps, ms);
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <base.fvecs> <query.fvecs> <gt.ivecs> [out_prefix]\n",
            argv[0]);
        return 1;
    }

    std::vector<float> base, query;
    std::vector<int>   gt;
    int d_base, d_query, d_gt;
    int nb_total = read_fvecs(argv[1], base,  d_base);
    int nq_total = read_fvecs(argv[2], query, d_query);
             read_ivecs(argv[3], gt,    d_gt);
    const int k = 10;

    std::string out_prefix = (argc > 4) ? argv[4]
        : "results/hblock_v37_prr_seed_diag";
    std::string csv_path = out_prefix + ".csv";
    std::string txt_path = out_prefix + ".txt";

    // Same defaults as demo_hblock_v37_prr.
    const int K1 = 16, K2 = 16, K3 = 16;
    const int deg = 32, epc = 4, dprj = 64, pbr = 16, batch = 1024;
    const int max_ef = 256;

    printf("base=%d x %d  query=%d x %d  gt=%d x %d\n",
           nb_total, d_base, nq_total, d_query, (int)(gt.size()/d_gt), d_gt);
    printf("max_ef=%d  K1=%d K2=%d K3=%d  graph_degree=%d  entry_per_cell=%d\n"
           "  d_proj=%d  per_block_r=%d  batch=%d\n"
           "  search_mode=CERTIFIED_PRR  eps_mode=VECTOR_LEVEL\n",
           max_ef, K1, K2, K3, deg, epc, dprj, pbr, batch);

    HBlockIndex::Params p;
    p.K1 = K1; p.K2 = K2; p.K3 = K3;
    p.max_ef = max_ef;
    p.d_proj = dprj; p.per_block_r = pbr;
    p.graph_degree = deg; p.entry_per_cell = epc;
    p.batch_size = batch;
    p.search_mode = HBlockIndex::Params::CERTIFIED_PRR;
    p.eps_mode    = HBlockIndex::Params::VECTOR_LEVEL;

    HBlockIndex idx(d_base, p);

    int n_train = std::min(nb_total, 200000);
    printf("\nTraining on %d vectors...\n", n_train);
    auto t0 = Clock::now();
    idx.train(base.data(), n_train);
    printf("  train: %.1f ms\n", Ms(Clock::now() - t0).count());

    printf("Adding %d vectors...\n", nb_total);
    t0 = Clock::now();
    idx.add(base.data(), nb_total);
    printf("  add: %.1f ms\n", Ms(Clock::now() - t0).count());

    FILE* txt = fopen(txt_path.c_str(), "w");
    if (!txt) { fprintf(stderr, "cannot open %s\n", txt_path.c_str()); return 1; }
    fprintf(txt, "=== HBlock v37_prr: exact-seed threshold diagnostic ===\n");
    fprintf(txt, "base=%d x %d  query=%d x %d  k=%d\n", nb_total, d_base, nq_total, d_query, k);
    fprintf(txt, "K1=%d K2=%d K3=%d graph_degree=%d entry_per_cell=%d d_proj=%d "
                 "per_block_r=%d batch=%d max_ef=%d\n\n",
            K1, K2, K3, deg, epc, dprj, pbr, batch, max_ef);

    // ── Required controls (same built index, no rebuild) ──────────────────────
    printf("\n=== Controls (same index, no rebuild) ===\n");
    fprintf(txt, "=== Controls (same index, no rebuild) ===\n");

    idx.set_search_mode(HBlockIndex::Params::FIXED_PER_BLOCK);
    run_control(idx, "FIXED_PER_BLOCK", query, nq_total, k, gt, d_gt, 128, txt);
    run_control(idx, "FIXED_PER_BLOCK", query, nq_total, k, gt, d_gt, 256, txt);

    idx.set_search_mode(HBlockIndex::Params::CORRECTED_FIXED);
    run_control(idx, "CORRECTED_FIXED", query, nq_total, k, gt, d_gt, 128, txt);
    run_control(idx, "CORRECTED_FIXED", query, nq_total, k, gt, d_gt, 256, txt);

    idx.set_search_mode(HBlockIndex::Params::CERTIFIED_PRR);
    // These two calls also print the existing production tau_U survivor-count
    // diagnostic ("[prr diag] surv/block avg=...") — that IS the "current tau_U
    // survivor count" control the spec asks for.
    run_control(idx, "CERTIFIED_PRR", query, nq_total, k, gt, d_gt, 128, txt);
    run_control(idx, "CERTIFIED_PRR", query, nq_total, k, gt, d_gt, 256, txt);
    fprintf(txt, "\n");

    // ── Seed diagnostic sweep ───────────────────────────────────────────────
    const int efs[] = {32, 64, 128, 256};
    const int validate_nq = 100;

    std::vector<SeedDiagRow> all_rows;      // flattened across ef
    std::vector<double>      all_agreement; // parallel to all_rows

    FILE* csv = fopen(csv_path.c_str(), "w");
    if (!csv) { fprintf(stderr, "cannot open %s\n", csv_path.c_str()); return 1; }
    fprintf(csv,
        "ef,seed_per_block,nq,"
        "visited_blocks_per_query,valid_candidates_per_query,"
        "seeds_per_query,tau_u_mean,tau_seed_mean,tau_ratio_mean,"
        "survivors_per_block,p50_survivors_block,p90_survivors_block,p99_survivors_block,"
        "blocks_over_16_pct,nonseed_survivors_per_query,total_exact_per_query,"
        "baseline_exact_per_query,exact_ratio_mean,exact_ratio_p50,exact_ratio_p90,"
        "queries_better_than_baseline_pct,queries_with_insufficient_seeds,"
        "candidate_set_top10_agreement\n");

    printf("\n=== Seed diagnostic sweep (validate_nq=%d) ===\n", validate_nq);
    fprintf(txt, "=== Seed diagnostic sweep (validate_nq=%d) ===\n", validate_nq);
    fprintf(txt,
        "%-6s %-4s %-9s %-9s %-9s %-10s %-10s %-9s %-8s %-8s %-9s %-8s\n",
        "ef", "spb", "tau_U", "tau_seed", "ratio", "surv/blk", "p90surv",
        "exact/q", "base/q", "eratio", "agree%", "insuff");

    for (int ef : efs) {
        std::vector<SeedDiagRow> rows;
        double agreement[4] = {0, 0, 0, 0};
        idx.seed_diagnostic(query.data(), nq_total, k, ef, rows,
                            validate_nq, base.data(), agreement);

        for (size_t i = 0; i < rows.size(); i++) {
            const SeedDiagRow& r = rows[i];
            double agree_pct = 100.0 * agreement[i];

            fprintf(csv,
                "%d,%d,%d,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,"
                "%.4f,%.4f,%.4f,%.4f,%.6f,%.6f,%.6f,%.4f,%d,%.4f\n",
                r.ef, r.spb, nq_total,
                r.visited_blocks_q, r.valid_cands_q,
                r.seeds_q, r.tau_u_mean, r.tau_seed_mean, r.tau_ratio_mean,
                r.surv_block_avg, r.surv_block_p50, r.surv_block_p90, r.surv_block_p99,
                r.blocks_over16_pct, r.nonseed_surv_q, r.total_exact_q,
                r.baseline_exact_q, r.exact_ratio_mean, r.exact_ratio_p50, r.exact_ratio_p90,
                r.better_than_baseline_pct, r.insufficient_seed_queries,
                agree_pct);

            printf("%-6d %-4d %-9.1f %-9.1f %-9.3f %-10.1f %-10.1f %-9.1f %-8.1f %-8.3f %-9.1f %-8d\n",
                   r.ef, r.spb, r.tau_u_mean, r.tau_seed_mean, r.tau_ratio_mean,
                   r.surv_block_avg, r.surv_block_p90, r.total_exact_q,
                   r.baseline_exact_q, r.exact_ratio_mean, agree_pct,
                   r.insufficient_seed_queries);
            fprintf(txt, "%-6d %-4d %-9.1f %-9.1f %-9.3f %-10.1f %-10.1f %-9.1f %-8.1f %-8.3f %-9.1f %-8d\n",
                   r.ef, r.spb, r.tau_u_mean, r.tau_seed_mean, r.tau_ratio_mean,
                   r.surv_block_avg, r.surv_block_p90, r.total_exact_q,
                   r.baseline_exact_q, r.exact_ratio_mean, agree_pct,
                   r.insufficient_seed_queries);

            all_rows.push_back(r);
            all_agreement.push_back(agree_pct);
        }
    }
    fclose(csv);

    // ── Go / no-go evaluation ───────────────────────────────────────────────
    // Strong go @ ef in {128,256}: agreement==100% AND
    //   mean total_exact_required <= 0.5 * baseline_exact  (per-query means)
    //   AND p90 total_exact_required <= baseline_exact      (via p90 of the
    //   per-query exact_ratio distribution, i.e. exact_ratio_p90 <= 1.0)
    // Weak go: agreement==100% AND mean total_exact_required < baseline_exact.
    // No-go: mean total_exact_required >= baseline_exact for every spb, or any
    // unexplained candidate-set correctness failure (agreement < 100%).
    bool any_strong = false, any_weak = false, any_correctness_failure = false;
    std::string strong_desc, weak_desc;

    for (size_t i = 0; i < all_rows.size(); i++) {
        const SeedDiagRow& r = all_rows[i];
        double agree = all_agreement[i];
        if (agree < 100.0 - 1e-9) any_correctness_failure = true;

        if (r.ef == 128 || r.ef == 256) {
            bool strong = (agree >= 100.0 - 1e-9) &&
                          (r.total_exact_q <= 0.5 * r.baseline_exact_q) &&
                          (r.exact_ratio_p90 <= 1.0 + 1e-9);
            bool weak = (agree >= 100.0 - 1e-9) &&
                        (r.total_exact_q < r.baseline_exact_q);
            if (strong) {
                any_strong = true;
                char buf[256];
                snprintf(buf, sizeof(buf), "ef=%d spb=%d (exact/q=%.1f vs baseline/q=%.1f, ratio=%.3f, p90_ratio=%.3f)",
                         r.ef, r.spb, r.total_exact_q, r.baseline_exact_q,
                         r.exact_ratio_mean, r.exact_ratio_p90);
                if (!strong_desc.empty()) strong_desc += "; ";
                strong_desc += buf;
            }
            if (weak) {
                any_weak = true;
                char buf[256];
                snprintf(buf, sizeof(buf), "ef=%d spb=%d (exact/q=%.1f vs baseline/q=%.1f, ratio=%.3f)",
                         r.ef, r.spb, r.total_exact_q, r.baseline_exact_q, r.exact_ratio_mean);
                if (!weak_desc.empty()) weak_desc += "; ";
                weak_desc += buf;
            }
        }
    }

    const char* decision;
    if (any_correctness_failure)
        decision = "NO-GO (candidate-set correctness failure — see agreement column below 100%)";
    else if (any_strong)
        decision = "STRONG GO";
    else if (any_weak)
        decision = "WEAK GO";
    else
        decision = "NO-GO (best seed policy still has mean total_exact_required >= baseline_exact)";

    printf("\n=== Go/no-go decision: %s ===\n", decision);
    if (any_strong) printf("  strong-go rows: %s\n", strong_desc.c_str());
    if (any_weak)   printf("  weak-go rows:   %s\n", weak_desc.c_str());

    fprintf(txt, "\n=== Go/no-go decision ===\n");
    fprintf(txt, "Decision: %s\n", decision);
    if (any_strong) fprintf(txt, "  strong-go rows: %s\n", strong_desc.c_str());
    if (any_weak)   fprintf(txt, "  weak-go rows:   %s\n", weak_desc.c_str());
    fprintf(txt,
        "\nInterpretation notes:\n"
        "  1. tau_U (k-th smallest U2 over all visited candidates) is the current\n"
        "     production threshold; tau_seed_mean/tau_u_mean above is its ratio to\n"
        "     the exact-seed threshold tau_seed2 tested here.\n"
        "  2. Whether exact seeds give a sufficiently tighter safe upper threshold is\n"
        "     answered by total_exact_per_query vs baseline_exact_per_query above.\n"
        "  3. A tight tau_seed2 does not by itself prove the lower bound L2 is tight;\n"
        "     nonseed_survivors_per_query measures how many non-seed candidates still\n"
        "     cannot be eliminated by L2 <= tau_seed2.\n"
        "  4. Passing this survivor-count gate justifies implementing and benchmarking\n"
        "     a two-wave GPU search — it is not itself a QPS measurement.\n");
    fclose(txt);

    printf("\nWrote %s and %s\n", csv_path.c_str(), txt_path.c_str());
    return 0;
}
