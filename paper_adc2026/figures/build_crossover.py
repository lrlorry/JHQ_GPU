#!/usr/bin/env python3
"""Build cost against search benefit: when does CAGRA-int8's faster search pay?

Section 6.1 used to say JHQ builds in 1.1-14.3 s "against 9-75 s for CAGRA
int8". That took one dataset's minimum against another dataset's maximum, which
is not a comparison. Per dataset the cold builds are 1.1-13.4 s against
4.9-73.3 s, and the ratio spans 3.8x to 13.4x rather than the 5-8x claimed.

A ratio is also the wrong shape for the claim. What a reader wants to know is
when the trade flips, so this reports the crossover: the query volume at which
CAGRA-int8's higher throughput has repaid its slower build. Below it JHQ costs
less end to end.

Two choices keep the number honest rather than flattering:

  * CAGRA's *fastest* observed build is used, so the crossover is the earliest
    it could be -- the one least favourable to JHQ. Its median build would push
    every figure higher.
  * JHQ's build is the cold-cache run at nprobe=8. Its coarse training is
    cached in later runs, so warm builds are several times cheaper; reporting
    those would be comparing a warm build against a cold one.

Both sides are the whole index build, not the search-side work.
"""
import collections
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
ALIAS = {"stella": "stella-trec24"}


def cagra_int8_builds():
    b = collections.defaultdict(list)
    for f in glob.glob(os.path.join(REPO, "results", "**", "*.csv"), recursive=True):
        lines = [l for l in open(f) if not l.startswith("#")]
        if not lines:
            continue
        try:
            rows = list(csv.DictReader(lines))
        except Exception:
            continue
        for r in rows:
            if (r.get("method") or "").lower() != "cuvs-cagra-int8":
                continue
            # A row the harness quarantined is quarantined whole.  stella's
            # fastest int8 build, 73.297 s, is the CONTAMINATED row's
            # train_ms -- the same row whose 618,859 QPS is excluded from the
            # frontier.  Its contamination reason is a throughput spread, so
            # it does not directly impeach the build timer; but the reason it
            # was flagged is that the card was busy, and a build timed on a
            # busy card is no more trustworthy than a search timed on one.
            st = (r.get("status") or "ok").strip()
            if st and st != "ok":
                continue
            t = r.get("train_ms") or r.get("build_ms")
            if t:
                b[r.get("dataset")].append(float(t) / 1000.0)
    return b


def jhq_cold_builds():
    """train + add from the cold-cache nprobe=8 run."""
    out = {}
    for ln in open(style.datafile("paper_fronts.log")):
        m = re.search(r"FIX\s+(\S+)\s+M=\d+\s+nlist=\d+\s+np=8\s+.*"
                      r"train=([\d.]+)\s+add=([\d.]+)", ln)
        if m:
            out[m.group(1)] = (float(m.group(2)) + float(m.group(3))) / 1000.0
    return out


def main():
    cb, jb = cagra_int8_builds(), jhq_cold_builds()
    bl, fr = style.load_baselines(), style.load_fronts()
    ratios, crossovers = [], []
    print("%-14s %8s %8s   %s" % ("dataset", "JHQ (s)", "int8 (s)", "crossover"))
    for ds in style.DATASETS:
        key = ALIAS.get(ds, ds)
        if key not in cb or ds not in jb or ds not in fr:
            continue
        c = min(cb[key])
        ratios.append(c / jb[ds])
        cells = []
        for R in (0.90, 0.95):
            qj = style.interp(fr[ds]["rule"], R)
            c8 = bl.get(ds, {}).get("cagra8")
            qc = style.interp(style.pareto(c8), R) if c8 else None
            if not (qj and qc) or qc <= qj:
                cells.append("R=%.2f: int8 not faster" % R)
                continue
            n = (c - jb[ds]) / (1.0 / qj - 1.0 / qc)
            crossovers.append(n)
            cells.append("R=%.2f: %.1fM queries" % (R, n / 1e6))
        print("%-14s %8.2f %8.2f   %s" % (style.PRETTY[ds], jb[ds], c, "; ".join(cells)))
    print("\nbuild ratio %.1fx to %.1fx; crossover %.1fM to %.1fM queries"
          % (min(ratios), max(ratios), min(crossovers) / 1e6, max(crossovers) / 1e6))


if __name__ == "__main__":
    main()
