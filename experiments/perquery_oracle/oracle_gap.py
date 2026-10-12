#!/usr/bin/env python3
"""Per-query adaptive refinement: how large is the oracle gap?

Question: within one workload, do queries need different refinement budgets,
and if every query got exactly the budget it needs, how much would that save
over the best *single* (workload-level) alpha?  Oracles only -- they use the
reference output or ground truth per query, which no online method has. A gap
that is small here closes the direction; a large one only justifies the next
step (a predictor evaluated on observable features, then a GPU kernel).

Two modes.

  python3 experiments/perquery_oracle/oracle_gap.py
      The four configurations dumped in paper_adc2026/data/alpha_perquery/
      (vogue-768, openai3-3072 at nlist=4096, np 128/512). Timing: fixed-alpha
      QPS from budget_arms_2ds.log (primary) and the dumps' own JSON QPS
      (sensitivity); stage shares from k_and_stages.log; and the ceiling for
      all 12 paper configurations from budget_arms.log.

  python3 experiments/perquery_oracle/oracle_gap.py --pq-dir DIR \\
          --config arxiv-768:128 --config arxiv-768:512 \\
          --arms-log paper_adc2026/data/budget_arms.log --tag arxiv
      New dumps (run_arxiv_dump.sh). The alpha grid is whatever files exist;
      the reference is analysed both at the largest alpha present and at 200,
      because arxiv is not saturated at 200. Timing: the dumps' own JSON QPS
      (primary; covers every alpha) and the arms log (sensitivity; alpha<=200).
      The arms log also serves as an identity check: if this index is the one
      the paper measured, held-out (odd-query) recall must equal the logged
      FIXED arms at every alpha, to the 4 decimals logged.

Cost model (an estimate, not GPU time): a query refined at alpha costs
T(alpha) = 1/QPS(alpha) of a uniform-alpha batch, made non-decreasing in alpha
by isotonic regression, and a mixed batch costs the mean of its queries' T.
That is the ideal case for per-query budgets: no tail from the slowest
blocks, no scheduling or bookkeeping cost, a free and perfect predictor, and
the scan's top-ck selection shrinking with each query's ck. Every gain below
is an upper bound.

Quality constraints, matched between workload-level and per-query policies on
the same queries (n=1000, k=10, so 10 slots = 1e-3 of all slots):
  LF0   output identical to the reference                    (label-free)
  LF10  <= 10 slots disagree with the reference              (label-free;
        implies recall loss <= 1e-3 against the reference on these queries)
  GT10  <= 10 true neighbours lost against the reference     (labels; per-query
        losses clamped at 0, so a lucky small-alpha gain buys no budget)
Per-query assignments under LF10/GT10 are exact multiple-choice knapsacks.

Decision rule for the arxiv run, fixed before its data exists (the thresholds
proposed in the review that asked for the run; they are a convention, not a
theorem): ideal GT10 gain with the largest reference and the primary timing
  >= 1.5x  GO        study a low-cost difficulty predictor, then GPU batches
  1.2-1.5x MARGINAL
  <  1.2x  STOP      close per-query adaptive refinement
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "seq_calibration"))
from common import DATA, K, ivecs, parse_arms_log  # noqa: E402

BUDGET = 10
OUT = os.path.join(HERE, "out")
DEFAULT_PQ = os.path.join(DATA, "alpha_perquery")
DEFAULT_CONFIGS = [("vogue-768", 128), ("vogue-768", 512),
                   ("openai3-3072", 128), ("openai3-3072", 512)]


# ------------------------------------------------------------------ loading --
def overlap(a, b):
    return np.array([len(set(x) & set(y)) for x, y in zip(a, b)], dtype=np.int64)


def load_dumps(pq_dir, ds, np_):
    alphas, ids, qps = [], {}, {}
    for p in glob.glob(os.path.join(pq_dir, "%s_np%d_a*.ivecs" % (ds, np_))):
        a = int(re.search(r"_a(\d+)\.ivecs$", p).group(1))
        alphas.append(a)
        ids[a] = ivecs(p)[:, :K]
        js = p[:-len(".ivecs")] + ".json"
        if os.path.exists(js):
            qps[a] = float(json.load(open(js))["perf"]["qps"])
    alphas.sort()
    gt = ivecs(os.path.join(pq_dir, "gt_%s.ivecs" % ds))[:, :K]
    n = min(len(gt), *(len(v) for v in ids.values()))
    ids = {a: v[:n] for a, v in ids.items()}
    hits = {a: overlap(v, gt[:n]) for a, v in ids.items()}
    return alphas, ids, hits, qps, n


def isotonic(T, grid):
    """Pool-adjacent-violators: the closest non-decreasing curve in alpha. More
    refinement candidates cannot cost less; a measured T(8) < T(4) is noise,
    and an oracle that exploits it reports a gain that is not there."""
    blocks = [[T[a], 1, [a]] for a in grid]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v = blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
            m = blocks[i][1] + blocks[i + 1][1]
            blocks[i] = [v / m, m, blocks[i][2] + blocks[i + 1][2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return {a: b[0] for b in blocks for a in b[2]}


def arms_entry(arms_log, ds, np_):
    hits = [v for (d, p, _), v in parse_arms_log(arms_log).items() if d == ds and p == np_]
    return hits[0] if hits else None


# --------------------------------------------------------------- oracles ----
def per_query_min(cost, grid):
    """Smallest alpha from which cost[a][q] stays 0 for every larger alpha."""
    n = len(cost[grid[-1]])
    best = np.full(n, grid[-1])
    ok = np.ones(n, dtype=bool)
    for a in reversed(grid):
        ok &= cost[a] == 0
        best[ok] = a
    return best


def mckp(cost, T, grid, budget):
    """Exact: one alpha per query, minimising sum T subject to sum cost <= budget."""
    n = len(cost[grid[-1]])
    dp = np.zeros(budget + 1)
    choice = np.zeros((n, budget + 1), dtype=np.int64)
    for q in range(n):
        nd = np.full(budget + 1, np.inf)
        for a in grid:
            c = int(cost[a][q])
            if c > budget:
                continue
            cand = dp[:budget + 1 - c] + T[a]
            idx = np.arange(c, budget + 1)
            better = cand < nd[idx]
            nd[idx[better]] = cand[better]
            choice[q, idx[better]] = a
        dp = nd
    picks = np.zeros(n, dtype=np.int64)
    cap = budget
    for q in range(n - 1, -1, -1):
        picks[q] = choice[q, cap]
        cap -= int(cost[picks[q]][q])
    return picks


def nesting(ids, grid):
    """Necessary condition for nested candidate sets with fixed scores: the
    reference items an alpha recovers never disappear at a larger alpha."""
    ref = ids[grid[-1]]
    viol = pairs = cnt = 0
    for i, a1 in enumerate(grid[:-1]):
        for a2 in grid[i + 1:]:
            for q in range(len(ref)):
                r = set(ref[q])
                s1, s2 = set(ids[a1][q]) & r, set(ids[a2][q]) & r
                pairs += 1
                viol += not s1 <= s2
                cnt += len(s1) > len(s2)
    return pairs, viol, cnt


def category(g):
    return "GO" if g >= 1.5 else ("MARGINAL" if g >= 1.2 else "STOP")


def analyze(lines, title, ids, hits, n, grid, Traw):
    """Everything for one (reference = grid[-1], timing source) scenario."""
    ref = grid[-1]
    T = isotonic(Traw, grid)
    dis = {a: K - overlap(ids[a], ids[ref]) for a in grid}
    gtl = {a: np.maximum(0, hits[ref] - hits[a]) for a in grid}
    a_lf, a_gt = per_query_min(dis, grid), per_query_min(gtl, grid)
    lines += ["", "-- %s" % title,
              "A. per-query minimal alpha (stays acceptable for every larger alpha):",
              "   %-36s" % "cumulative share of queries <= a" + "".join("%7d" % a for a in grid)]
    for nm, v in (("identical to reference output (LF)", a_lf), ("no lost true neighbour (GT)", a_gt)):
        lines.append("   %-36s" % nm + "".join("%6.1f%%" % (100 * np.mean(v <= a)) for a in grid))

    def single(costd):
        return min(a for a in grid if costd[a].sum() <= BUDGET)

    def single0(costd):
        return min(a for a in grid if costd[a].sum() == 0)

    pol = {"fixed alpha=%d (reference)" % ref: np.full(n, ref)}
    if 100 in grid and ref != 100:
        pol["fixed alpha=100"] = np.full(n, 100)
    pol["W-LF0  workload alpha=%d" % single0(dis)] = np.full(n, single0(dis))
    pol["PQ-LF0 per-query oracle"] = a_lf
    pol["W-LF10 workload alpha=%d" % single(dis)] = np.full(n, single(dis))
    pol["PQ-LF10 per-query oracle"] = mckp(dis, T, grid, BUDGET)
    pol["W-GT10 workload alpha=%d" % single(gtl)] = np.full(n, single(gtl))
    pol["PQ-GT10 per-query oracle"] = mckp(gtl, T, grid, BUDGET)
    pol["floor: every query alpha=%d" % grid[0]] = np.full(n, grid[0])
    lines += ["B. isotonic T(alpha) us/query: " + "  ".join("a%d=%.2f" % (a, T[a]) for a in grid),
              "   %-30s %9s %10s %9s %9s %9s %9s" % ("policy", "mean a", "cand/q", "T us/q",
                                                     "recall", "dis slot", "lost GT")]
    rows = {}
    for nm, al in pol.items():
        rows[nm] = {"mean_alpha": float(np.mean(al)), "cand": float(np.mean(al) * K),
                    "T_us": float(np.mean([T[a] for a in al])),
                    "recall": float(np.mean([hits[a][q] for q, a in enumerate(al)]) / K),
                    "dis_slots": int(sum(dis[a][q] for q, a in enumerate(al))),
                    "lost_gt": int(sum(gtl[a][q] for q, a in enumerate(al))),
                    "alpha_hist": {int(a): int((al == a).sum()) for a in grid}}
        r = rows[nm]
        lines.append("   %-30s %9.1f %10.0f %9.3f %9.4f %9d %9d" % (
            nm, r["mean_alpha"], r["cand"], r["T_us"], r["recall"], r["dis_slots"], r["lost_gt"]))
    wn = lambda tag: [k for k in rows if k.startswith(tag + " ")][0]
    gains = {}
    lines.append("   matched-quality gap (workload T / per-query T; refinement candidates):")
    for tag in ("LF0", "LF10", "GT10"):
        w, p = rows[wn("W-" + tag)], rows[wn("PQ-" + tag)]
        gains[tag] = w["T_us"] / p["T_us"]
        lines.append("     %-5s  time x%.3f   refinement candidates %+.0f%%" % (
            tag, gains[tag], 100 * (p["cand"] / w["cand"] - 1)))
    floor = rows["floor: every query alpha=%d" % grid[0]]["T_us"]
    gains["ceiling"] = rows[wn("W-GT10")]["T_us"] / floor
    lines.append("     ceiling (W-GT10 -> every query alpha=%d): time x%.3f" % (grid[0], gains["ceiling"]))
    return {"rows": rows, "gains": gains, "T": T,
            "cdf_lf": {a: float(np.mean(a_lf <= a)) for a in grid},
            "cdf_gt": {a: float(np.mean(a_gt <= a)) for a in grid}}


# -------------------------------------------------------- context tables ----
def stage_deltas():
    """alpha=4 -> alpha=100 increase in each stage's median time (item 8)."""
    d = defaultdict(list)
    for ln in open(os.path.join(DATA, "k_and_stages.log"), errors="ignore"):
        m = re.match(r"\s*STAGE (\S+) np=(\d+) a=([\d.]+) k=10 (\S+)\s+([\d.]+)", ln)
        if m:
            d[(m.group(1), int(m.group(2)), float(m.group(3)), m.group(4))].append(float(m.group(5)))
    out = {}
    stages = ("gemm_rotate", "gemm_centroid", "select_probes", "build_byte_lut",
              "scan_ivf", "residual_refine", "topk_final")
    for ds, np_ in {(k[0], k[1]) for k in d}:
        med = lambda a, s: statistics.median(d[(ds, np_, a, s)])
        tot = lambda a: sum(med(a, s) for s in stages)
        o = {s: med(100.0, s) - med(4.0, s) for s in ("scan_ivf", "residual_refine", "topk_final")}
        o.update(total=tot(100.0) - tot(4.0), total_a4=tot(4.0), total_a100=tot(100.0))
        out[(ds, np_)] = o
    return out


def ceilings_all_configs():
    """Paper's 12 configurations: T(oracle alpha)/T(2), quality ignored."""
    grid = [2, 4, 8, 16, 32, 64, 100, 200]
    rows = []
    for (ds, np_, nl), v in sorted(parse_arms_log(os.path.join(DATA, "budget_arms.log")).items(),
                                    key=lambda kv: (kv[0][1], kv[0][0])):
        T = isotonic({a: 1e6 / v["fixed"][a][1] for a in grid}, grid)
        orc = int(float(v["oracle"]["alpha"]))
        rows.append((ds, np_, nl, orc, T[orc], T[2], T[orc] / T[2],
                     100.0 * (1 - T[2] / T[orc]), v["fixed"][2][0], v["fixed"][orc][0]))
    return rows


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pq-dir", default=None)
    ap.add_argument("--config", action="append", default=None, help="ds:np")
    ap.add_argument("--arms-log", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    default = args.pq_dir is None
    pq_dir = DEFAULT_PQ if default else args.pq_dir
    configs = DEFAULT_CONFIGS if not args.config else \
        [(c.split(":")[0], int(c.split(":")[1])) for c in args.config]
    arms_log = args.arms_log or os.path.join(DATA, "budget_arms_2ds.log")
    tag = args.tag or ("default" if default else "custom")
    stages = stage_deltas() if default else {}
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for ds, np_ in configs:
        alphas, ids, hits, qps_self, n = load_dumps(pq_dir, ds, np_)
        arms = arms_entry(arms_log, ds, np_)
        lines = ["", "=" * 100, "%s/np%d   (n=%d queries, k=%d, alphas dumped: %s)" % (
            ds, np_, n, K, ",".join(map(str, alphas)))]
        out = {"n": n, "alphas": alphas}

        # identity: is this the index the arms log measured?  (odd queries held out)
        if arms is not None:
            odd = np.arange(1, n, 2)
            diffs = {a: hits[a][odd].mean() / K - arms["fixed"][a][0]
                     for a in alphas if a in arms["fixed"]}
            mx = max(abs(v) for v in diffs.values())
            lines.append("identity vs %s: max |held-out recall diff| = %.4f over %d alphas -> %s" % (
                os.path.basename(arms_log), mx, len(diffs),
                "SAME index and batch-independent results" if mx < 5e-5 else
                "different index (retrained), batch-dependent results, or another GPU's arithmetic"))
            out["identity_max_diff"] = mx

        # scenarios: (title, grid, timing)
        scen = []
        g_all = alphas
        g200 = [a for a in alphas if a <= 200]
        arms_T = {a: 1e6 / arms["fixed"][a][1] for a in g200 if arms and a in arms["fixed"]}
        self_T = {a: 1e6 / qps_self[a] for a in alphas if a in qps_self}
        if default:
            scen.append(("reference a=%d, timing %s (primary)" % (g200[-1], os.path.basename(arms_log)), g200, arms_T))
            scen.append(("reference a=%d, timing dump JSON QPS (sensitivity)" % g200[-1], g200, self_T))
        else:
            scen.append(("reference a=%d, timing dump JSON QPS (primary)" % g_all[-1], g_all, self_T))
            if g_all[-1] != g200[-1]:
                scen.append(("reference a=%d, timing dump JSON QPS (reference-budget effect)" % g200[-1], g200, self_T))
            if len(arms_T) == len(g200):
                scen.append(("reference a=%d, timing %s (sensitivity)" % (g200[-1], os.path.basename(arms_log)), g200, arms_T))
        out["scenarios"] = {}
        for title, grid, Tm in scen:
            if any(a not in Tm for a in grid):
                lines.append("   (skipped '%s': no QPS for some alpha)" % title)
                continue
            out["scenarios"][title] = analyze(lines, title, ids, hits, n, grid, Tm)

        first = next(iter(out["scenarios"].values()), None)
        if first and not default:
            g = first["gains"]
            lines.append("")
            lines.append("DECISION (pre-registered, primary scenario): GT10 x%.3f -> %s   "
                         "(LF10 x%.3f -> %s; ceiling x%.3f)" % (
                             g["GT10"], category(g["GT10"]), g["LF10"], category(g["LF10"]), g["ceiling"]))
            # Added before any arxiv data existed, when the 5090 became unavailable:
            # the primary timing is whatever card ran the dumps, while the paper's
            # context is the 5090, whose fixed-alpha QPS the arms log holds (a<=200).
            s5090 = [v for t, v in out["scenarios"].items() if "budget_arms.log" in t]
            if s5090:
                g5 = s5090[0]["gains"]
                lines.append("DECISION (5090 timing from budget_arms.log, reference a=200): GT10 x%.3f -> %s   "
                             "(LF10 x%.3f -> %s; ceiling x%.3f)" % (
                                 g5["GT10"], category(g5["GT10"]), g5["LF10"], category(g5["LF10"]),
                                 g5["ceiling"]))
        pairs, viol, cviol = nesting(ids, alphas)
        lines.append("C. nesting check over %d (query, a1<a2) pairs: recovered reference items lost at a "
                     "larger alpha in %d (%.3f%%); count decreases in %d" % (pairs, viol, 100.0 * viol / pairs, cviol))
        out["nesting"] = {"pairs": pairs, "set_violations": viol, "count_violations": cviol}
        if stages.get((ds, np_)):
            s = stages[(ds, np_)]
            lines.append("D. stage times (k_and_stages.log item 8, batch of 1000): total %.2f ms at a=4, "
                         "%.2f ms at a=100; of the increase %.2f ms, scan %.0f%%, refine %.0f%%, top-k %.0f%%" % (
                             s["total_a4"], s["total_a100"], s["total"], 100 * s["scan_ivf"] / s["total"],
                             100 * s["residual_refine"] / s["total"], 100 * s["topk_final"] / s["total"]))
        print("\n".join(lines))
        res["%s/np%d" % (ds, np_)] = out

    if default:
        print("\n" + "=" * 100)
        print("E. all 12 paper configurations (budget_arms.log, 500 held-out queries): ceiling on what any")
        print("   per-query scheme could gain over the workload oracle = T(oracle a)/T(a=2), quality ignored")
        print("   %-14s %4s %6s %6s %9s %9s %9s %12s %18s" % ("dataset", "np", "nlist", "orc a", "T(orc)us",
                                                               "T(2)us", "ceiling", "a-dep share", "recall a=2 / orc"))
        res["_ceilings"] = []
        for r in ceilings_all_configs():
            print("   %-14s %4d %6d %6d %9.2f %9.2f %8.2fx %11.0f%% %10.4f / %.4f" % r)
            res["_ceilings"].append(dict(zip(("ds", "np", "nlist", "oracle_alpha", "T_orc", "T_2", "ceiling",
                                              "alpha_dep_share_pct", "recall_a2", "recall_orc"), r)))
    path = os.path.join(OUT, "oracle_gap.json" if tag == "default" else "oracle_gap_%s.json" % tag)
    json.dump(res, open(path, "w"), indent=1, default=float)
    print("\nwrote %s" % os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
