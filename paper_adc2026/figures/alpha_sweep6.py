#!/usr/bin/env python3
"""The rule against alpha=100, six datasets at S=32: what the sweep is worth.

Section 6.4 said "over a broader 32-configuration sweep against alpha=100 --
six datasets at S=32 -- gain is 1.044--2.653x for at most 0.0048 recall, median
payback 1.75 batches".  No log in data/ produced 1.044 or 2.653; the strings
appear in paper_fronts.log and batch_matched_CONTAMINATED.log, neither of which
is this experiment, which is how a substring audit passed them.

data/alpha6cfg.log is that sweep, actually run: six datasets, nprobe in
{8,32,128,512}, S=32, eps=1e-3, at the nlist and n_train the frontier reports.
Three things it settles.

**24 configurations, not 32.**  Six datasets times four probe depths is 24.
Whatever set of 32 the sentence described, it was not six-by-four.

**The rule can lose.**  One configuration comes out below 1.0 -- arxiv-768 at
nprobe=128, where the rule picks alpha=100 anyway and pays the calibration for
nothing.  A published range starting at 1.044 has no losing case in it, and the
losing case is the informative one: arxiv is the dataset whose sweep was still
falling at alpha_max, so it is exactly where a label-free rule has the least to
work with.

**Payback is per batch of 1,000 queries, and the median is 11.5, not 1.75.**
The spread is 5 to 169 batches and it is not noise: calibration costs a fixed
number of grid probes while the saving is proportional to the gain, so the
configurations that gain least take longest to repay.  Quoting one median hides
that a 1.01x configuration never really repays.  We report the range.
"""
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style


def load():
    rows, tag = [], None
    for ln in open(style.datafile("alpha6cfg.log"), errors="ignore"):
        m = re.match(r"###\s+(\S+)\s+M=\d+\s+nlist=\d+\s+np=(\d+)", ln)
        if m:
            tag = (m.group(1), int(m.group(2)))
            continue
        m = re.search(r"AS_RESULT picked=([\d.]+) recall_picked=([\d.]+) "
                      r"qps_picked=(\d+) recall_max=([\d.]+) qps_max=(\d+) "
                      r"gain=([\d.]+) cal_ms=([\d.]+)", ln)
        if m and tag:
            rows.append(dict(ds=tag[0], np=tag[1], alpha=float(m.group(1)),
                             rp=float(m.group(2)), qp=int(m.group(3)),
                             rm=float(m.group(4)), qm=int(m.group(5)),
                             gain=float(m.group(6)), cal=float(m.group(7))))
    return rows


def main():
    rows = load()
    if not rows:
        print("  alpha6cfg.log has no results"); return
    g = [r["gain"] for r in rows]
    d = [r["rm"] - r["rp"] for r in rows]
    lose = [r for r in rows if r["gain"] < 1.0]
    print("  %d configurations (%d datasets x %d probe depths), S=32:"
          % (len(rows), len({r["ds"] for r in rows}), len({r["np"] for r in rows})))
    print("     gain %.3fx to %.3fx, giving up at most %.4f recall"
          % (min(g), max(g), max(d)))
    print("     the rule loses in %d: %s"
          % (len(lose), ", ".join("%s nprobe=%d %.3fx" % (r["ds"], r["np"], r["gain"])
                                  for r in lose) or "none"))
    # One batch is the 1,000 queries the driver searches.  Calibration is a
    # fixed number of grid probes; the saving scales with the gain, so payback
    # is a range and not a constant.
    pb = []
    for r in rows:
        if r["gain"] <= 1.0:
            continue
        per = 1000.0 * (1.0 / r["qm"] - 1.0 / r["qp"]) * 1000.0
        if per > 0:
            pb.append(r["cal"] / per)
    if pb:
        print("     payback %.1f to %.1f batches of 1,000 queries, median %.1f"
              % (min(pb), max(pb), statistics.median(pb)))


if __name__ == "__main__":
    main()
