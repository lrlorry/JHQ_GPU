#pragma once
// Standard Recall@k, plus the pre-v15 score it replaces.
//
//     Recall@k = |{returned top-k} ∩ {true top-k}| / k,  averaged over queries
//
// This matches the definition used by the JHQ CPU paper, by FAISS, and by the
// GPU baselines this repo has to compare against (CAGRA, SONG): the returned
// set intersected with the *exact k-NN set*, over k.
//
// Every demo_*.cu before v15 got this wrong in two independent ways.
//
//  1. Comparison depth. The old loop was `for (g = 0; g < gt_k; g++)`, with
//     gt_k = the .ivecs row width -- 20 for the datasets built by
//     scripts/download_jhq_datasets.py (GT_K = 20) and 100 for the ones built
//     by scripts/preprocess.py (--k default 100). So it measured
//     |top-k ∩ true top-gt_k| / k. That is always >= Recall@k, it saturates at
//     1.0000 long before the search is exact, and because the two generators
//     disagree it is not even the same quantity across datasets.
//     Note the row *stride* is still gt_k -- only the depth becomes k.
//
//  2. Slot reuse. A returned id was counted once per matching ground-truth
//     entry. A search returning the same id k times scored 1.0. Set semantics
//     means each ground-truth slot is consumed at most once.
//
// `legacy` reproduces the old number bit-for-bit so a re-run can be lined up
// against the pre-v15 CSVs in results/ instead of silently replacing them.

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

struct RecallResult {
    double    recall      = 0.0;  // standard Recall@k -- the one that goes in the paper
    double    legacy      = 0.0;  // pre-v15 |top-k ∩ true top-gt_k| / k, slots reusable
    int       k           = 0;
    int       gt_width    = 0;    // .ivecs row stride
    int       eval_depth  = 0;    // ground-truth entries actually compared (== k)
    long long dup_queries = 0;    // queries whose returned list held a repeated id
};

// labels : [nq, k]     returned neighbour ids
// gt     : [nq, gt_k]  exact neighbours, ascending by distance
inline RecallResult evaluate_recall(const int* labels, const int* gt,
                                    int nq, int k, int gt_k) {
    if (k <= 0)  throw std::invalid_argument("evaluate_recall: k must be positive");
    if (nq <= 0) throw std::invalid_argument("evaluate_recall: nq must be positive");
    if (gt_k < k)
        throw std::runtime_error(
            "evaluate_recall: ground-truth width " + std::to_string(gt_k) +
            " < k=" + std::to_string(k) + ". Recall@" + std::to_string(k) +
            " is undefined with a shallower ground truth -- regenerate it at "
            "least k wide (scripts/download_jhq_datasets.py: GT_K; "
            "scripts/preprocess.py: --k).");

    long long hits = 0, legacy_hits = 0, dup_q = 0;
    std::vector<char> gt_used(k);
    std::vector<int>  seen;
    seen.reserve(k);

    for (int i = 0; i < nq; ++i) {
        const int* pred = labels + (long long)i * k;
        const int* row  = gt     + (long long)i * gt_k;   // stride stays gt_k

        std::fill(gt_used.begin(), gt_used.end(), (char)0);
        seen.clear();
        bool dup = false;

        for (int j = 0; j < k; ++j) {
            const int found = pred[j];

            if (std::find(seen.begin(), seen.end(), found) != seen.end()) dup = true;
            else                                                          seen.push_back(found);

            // Standard: first k ground-truth entries only, each consumed once.
            for (int g = 0; g < k; ++g)
                if (!gt_used[g] && row[g] == found) { gt_used[g] = 1; ++hits; break; }

            // Legacy: whole row, slots reusable -- reproduces the pre-v15 score.
            for (int g = 0; g < gt_k; ++g)
                if (row[g] == found) { ++legacy_hits; break; }
        }
        if (dup) ++dup_q;
    }

    const double denom = (double)nq * (double)k;
    RecallResult r;
    r.recall      = (double)hits        / denom;
    r.legacy      = (double)legacy_hits / denom;
    r.k           = k;
    r.gt_width    = gt_k;
    r.eval_depth  = k;
    r.dup_queries = dup_q;
    return r;
}
