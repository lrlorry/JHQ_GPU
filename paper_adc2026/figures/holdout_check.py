#!/usr/bin/env python3
"""Does the frontier's same-pool calibration change what it selects?

Section 6.1's dashed curve calibrates on a 32-query sample strided out of the
1,000 queries it then scores, so 32 of 1,000 reported queries were seen by the
selector.  fhold.log re-runs that protocol twice on one binary: JHQ_AS_HOLDOUT=1
calibrates on even-indexed queries and reports on odd, JHQ_AS_HOLDOUT=0
reproduces the pooled behaviour.

The quantity the leak can act on is the selected alpha: that is the whole
decision calibration makes, and under holdout the selector has seen none of
the queries the number is then reported on.  If the two arms pick the same
alpha, seeing 32 of the 1,000 reported queries did not change the outcome.

The other two columns do not isolate the leak and are printed as context.
Recall is measured on different query sets -- 500 odd-indexed against all
1,000 -- so a difference there mixes the leak with whichever half is easier.
QPS is not comparable at all, the batches being 500 against 1,000; gain is,
since it is a ratio taken inside one batch.
"""
import collections, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style

LOG = style.datafile("fhold.log")
ROW = re.compile(
    r"^\s*HOLD(\d)\s+(\S+)\s+M=\d+\s+nlist=\d+\s+np=(\d+)\s+AF_RESULT\s+"
    r"picked=(\d+)\s+probes=\d+\s+recall_picked=([\d.]+)\s+qps_picked=(\d+)\s+"
    r"recall_max=([\d.]+)\s+qps_max=(\d+)\s+gain=([\d.]+)")


def load(path=LOG):
    out = {}
    for ln in open(path, errors="ignore"):
        m = ROW.match(ln)
        if not m:
            continue
        ho, ds, np_ = int(m.group(1)), m.group(2), int(m.group(3))
        out[(ds, np_, ho)] = dict(
            alpha=int(m.group(4)), recall=float(m.group(5)),
            qps=int(m.group(6)), recall_max=float(m.group(7)),
            qps_max=int(m.group(8)), gain=float(m.group(9)))
    return out


def main():
    r = load()
    cells = sorted({(ds, np_) for ds, np_, _ in r})
    same = diffs = 0
    dr, dg = [], []
    print(f"{'dataset':<14}{'np':>5}  {'alpha H/P':>11}  "
          f"{'recall H':>9}{'recall P':>9}{'Δrecall':>9}  {'gain H':>7}{'gain P':>7}")
    for ds, np_ in cells:
        h, p = r.get((ds, np_, 1)), r.get((ds, np_, 0))
        if not (h and p):
            continue
        same += h["alpha"] == p["alpha"]
        diffs += 1
        dr.append(h["recall"] - p["recall"])
        dg.append(h["gain"] - p["gain"])
        print(f"{ds:<14}{np_:>5}  {h['alpha']:>5}/{p['alpha']:<5}  "
              f"{h['recall']:>9.4f}{p['recall']:>9.4f}{dr[-1]:>+9.4f}  "
              f"{h['gain']:>7.3f}{p['gain']:>7.3f}")
    if not diffs:
        print("no paired cells yet")
        return
    print(f"\n配对单元 {diffs}  选中 alpha 相同 {same}/{diffs}")
    print(f"held-out recall 差:  中位 {sorted(dr)[len(dr)//2]:+.4f}  "
          f"最大 |Δ| {max(abs(x) for x in dr):.4f}")
    print(f"gain 差:             中位 {sorted(dg)[len(dg)//2]:+.3f}  "
          f"最大 |Δ| {max(abs(x) for x in dg):.3f}")
    worse = sum(1 for x in dr if x < -1e-4)
    better = sum(1 for x in dr if x > 1e-4)
    print(f"held-out recall 更低 {worse} 个,更高 {better} 个 "
          f"(含子集效应,不是泄漏的度量)")
    dis = [(ds, np_) for ds, np_ in cells
           if (ds, np_, 1) in r and (ds, np_, 0) in r
           and r[(ds, np_, 1)]["alpha"] != r[(ds, np_, 0)]["alpha"]]
    if dis:
        print("选中 alpha 不同的单元: " + ", ".join(
            f"{d}/{n} ({r[(d,n,1)]['alpha']} vs {r[(d,n,0)]['alpha']})"
            for d, n in dis))


if __name__ == "__main__":
    main()
