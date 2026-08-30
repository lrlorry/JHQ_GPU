// Unit tests for the Recall@k evaluator. Each case is one of the ways the
// pre-v15 demo evaluator could report a wrong number.
//
//   build/tests/test_recall   ->  exits non-zero on the first failure

#include "common/recall.cuh"

#include <cstdio>
#include <cmath>
#include <vector>

static int failures = 0;

static void check(const char* name, double got, double want) {
    const bool ok = std::fabs(got - want) < 1e-9;
    printf("%-52s got=%.4f want=%.4f  %s\n", name, got, want, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

int main() {
    // One query, k=10, ground truth 100 wide: ids 0..99 in rank order.
    const int k = 10, gt_k = 100;
    std::vector<int> gt(gt_k);
    for (int i = 0; i < gt_k; ++i) gt[i] = i;

    {   // Exact hit: the true top-10.
        std::vector<int> pred = {0,1,2,3,4,5,6,7,8,9};
        auto r = evaluate_recall(pred.data(), gt.data(), 1, k, gt_k);
        check("true top-10 returned", r.recall, 1.0);
        check("  legacy agrees here", r.legacy, 1.0);
    }
    {   // The case that motivated v15: ranks 11-20. Old evaluator said 1.0.
        std::vector<int> pred = {10,11,12,13,14,15,16,17,18,19};
        auto r = evaluate_recall(pred.data(), gt.data(), 1, k, gt_k);
        check("ranks 11-20 returned", r.recall, 0.0);
        check("  legacy reproduces the old 1.0", r.legacy, 1.0);
    }
    {   // Half and half.
        std::vector<int> pred = {0,1,2,3,4, 50,51,52,53,54};
        auto r = evaluate_recall(pred.data(), gt.data(), 1, k, gt_k);
        check("5 in top-10 + 5 in top-100", r.recall, 0.5);
        check("  legacy inflates to 1.0", r.legacy, 1.0);
    }
    {   // Nothing in the ground-truth row at all.
        std::vector<int> pred = {900,901,902,903,904,905,906,907,908,909};
        auto r = evaluate_recall(pred.data(), gt.data(), 1, k, gt_k);
        check("no overlap", r.recall, 0.0);
        check("  legacy also 0", r.legacy, 0.0);
    }
    {   // Duplicate ids must not be scored more than once.
        std::vector<int> pred = {0,0,0,0,0,0,0,0,0,0};
        auto r = evaluate_recall(pred.data(), gt.data(), 1, k, gt_k);
        check("same true-top-1 id repeated 10x", r.recall, 0.1);
        check("  legacy scores it 10 times", r.legacy, 1.0);
        check("  duplicate query flagged", (double)r.dup_queries, 1.0);
    }
    {   // gt_k == k: the two metrics must coincide.
        std::vector<int> gt10(gt.begin(), gt.begin() + k);
        std::vector<int> pred = {0,1,2,3,4, 90,91,92,93,94};
        auto r = evaluate_recall(pred.data(), gt10.data(), 1, k, k);
        check("gt_k == k: standard", r.recall, 0.5);
        check("gt_k == k: legacy identical", r.legacy, 0.5);
    }
    {   // Two queries with different stride offsets -- catches gt[i*k+g] indexing.
        std::vector<int> gt2(2 * gt_k);
        for (int i = 0; i < gt_k; ++i) { gt2[i] = i; gt2[gt_k + i] = 1000 + i; }
        std::vector<int> pred = {0,1,2,3,4,5,6,7,8,9,
                                 1000,1001,1002,1003,1004,1005,1006,1007,1008,1009};
        auto r = evaluate_recall(pred.data(), gt2.data(), 2, k, gt_k);
        check("2 queries, correct row stride", r.recall, 1.0);
    }
    {   // Ground truth shallower than k must be refused, not silently clamped.
        std::vector<int> gt5(5, 0);
        std::vector<int> pred(10, 0);
        bool threw = false;
        try { evaluate_recall(pred.data(), gt5.data(), 1, 10, 5); }
        catch (const std::exception&) { threw = true; }
        check("gt_k < k throws", threw ? 1.0 : 0.0, 1.0);
    }

    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "all passed",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
