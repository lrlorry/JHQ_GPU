#!/usr/bin/env python3
"""Does the frontier's same-pool calibration change what it selects?

Section 6.1's dashed curve calibrates on a 32-query sample strided out of the
1,000 queries it then scores, so 32 of 1,000 reported queries were seen by the
selector.  fhold.log re-runs that protocol twice on one binary: JHQ_AS_HOLDOUT=1
calibrates on even-indexed queries and reports on odd, JHQ_AS_HOLDOUT=0
reproduces the pooled behaviour.

QPS is not comparable between the arms -- the reported batch is 500 queries
under holdout and 1,000 pooled -- so this compares the three quantities that
are independent of batch size: the selected alpha, the recall it achieves, and
gain, which is a ratio measured inside one batch.
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
    # The pooled arm scores the queries its selector saw; if the leak flattered
    # the frontier, the held-out arm's recall is systematically the lower one.
    worse = sum(1 for x in dr if x < -1e-4)
    better = sum(1 for x in dr if x > 1e-4)
    print(f"held-out 更低 {worse} 个,更高 {better} 个,持平 {diffs-worse-better} 个")


if __name__ == "__main__":
    main()
