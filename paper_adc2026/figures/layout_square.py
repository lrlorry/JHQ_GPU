#!/usr/bin/env python3
"""The table x layout square on six datasets: what packed loads are worth.

Table 2's packed-loads row and the abstract's copy of it said "+30 to +48%".
That range was a literal in fig_ablation.py and no log reproduced it.  The only
measurement we held was lut_groups.log's phase 3 -- two datasets, G in {1,2},
three probe depths -- which gives +0.3% to +40.7%.  data/layout6.log is the same
square on all six datasets and four probe depths, one source tree, one commit:

           full table      factorised
  byte     v59_b_g1        v59_b_g2
  word     v59_g1          v59_g2

Two things the wider run changes.

**The gain is a function of probe depth, so a flat range hides the mechanism.**
It is +1% to +6% at nprobe=8 and +9% to +66% at nprobe=512.  Quoting "+30 to
+48%" without naming a probe depth describes no measurement: it is neither the
range over the cells, nor the range at any one depth.  The reason it moves is
the same one the factorisation section already gives -- a load is paid once per
candidate per subspace, so its share of the scan grows with the candidates
scanned together.

**The identity holds everywhere.** All four corners of the square compute the
same primary distance, so their recall must agree; across the 24 cells it agrees
to within 2e-4, which is the tie-break floor.  A layout that changed the answer
would show up here before any timing was read.

The square also re-measures G=2 against G=1 on six datasets rather than two.
That is a different population from Figure 3(a)'s granularity sweep and is
reported separately -- counting them together is exactly the mistake that put a
"seven of eight" in the paper that no line in Figure 3(a) could show.
"""
import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style

DS = ["vogue-768", "arxiv-768", "openai3-1536", "openai3-3072", "bge-m3",
      "stella"]
NP = [8, 32, 128, 512]


def load():
    R = {}
    for ln in open(style.datafile("layout6.log"), errors="ignore"):
        m = re.match(r"\s*(\S+)\s+M=(\d+)\s+layout=(\w+)\s+G=(\d+)\s+np=(\d+)\s+"
                     r"recall=([\d.]+)\s+qps=(\d+)", ln)
        if m:
            R[(m.group(1), m.group(3), int(m.group(4)), int(m.group(5)))] = (
                float(m.group(6)), int(m.group(7)))
    return R


def main():
    R = load()
    if not R:
        print("  layout6.log has no cells"); return

    # The identity first: no timing is worth reading if the four corners
    # disagree about what they computed.
    worst, where = 0.0, None
    for ds in DS:
        for np_ in NP:
            rs = [R[(ds, l, g, np_)][0] for l in ("byte", "word")
                  for g in (1, 2) if (ds, l, g, np_) in R]
            if len(rs) > 1 and max(rs) - min(rs) > worst:
                worst, where = max(rs) - min(rs), (ds, np_)
    print("  recall across the four corners agrees to %.1e (worst: %s), "
          "against a 2e-4 tie-break floor" % (worst, where))

    lay = collections.defaultdict(list)
    fac = []
    for ds in DS:
        for np_ in NP:
            for g in (1, 2):
                b, w = R.get((ds, "byte", g, np_)), R.get((ds, "word", g, np_))
                if b and w:
                    lay[np_].append(100 * (w[1] / b[1] - 1))
            for l in ("byte", "word"):
                a, c = R.get((ds, l, 1, np_)), R.get((ds, l, 2, np_))
                if a and c:
                    fac.append((ds, np_, l, 100 * (c[1] / a[1] - 1)))

    allg = [g for v in lay.values() for g in v]
    print("  packed 32-bit loads over %d cells (6 datasets, %d probe depths, "
          "G in {1,2}): %+.1f%% to %+.1f%%" % (len(allg), len(lay),
                                               min(allg), max(allg)))
    print("  by probe depth, which is what the flat range hid:")
    for np_ in sorted(lay):
        print("     nprobe=%-4d %+.1f%% to %+.1f%%"
              % (np_, min(lay[np_]), max(lay[np_])))

    f = [x[3] for x in fac]
    neg = [x for x in fac if x[3] <= 0]
    print("  factorisation G=2 against G=1 in the same square, %d cells: "
          "%+.1f%% to %+.1f%%, losing in %d"
          % (len(f), min(f), max(f), len(neg)))
    for ds, np_, l, g in neg:
        print("     loses: %s nprobe=%d %s layout %+.1f%%" % (ds, np_, l, g))


if __name__ == "__main__":
    main()
