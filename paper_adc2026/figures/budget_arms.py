#!/usr/bin/env python3
"""Does the budget rule earn its place, or would one small fixed alpha do?

The paper reported the rule's gain against alpha=100 -- the default JHQ leaves
behind -- which invites the obvious objection: pin alpha=8 and skip the
calibration. budget_arms.log answers it. Every arm is measured on one index and
one timing harness, with calibration on even-indexed queries and every recall
below on the odd-indexed half.

What the numbers say, in the order that matters:

  1. No single fixed alpha works. The ground-truth oracle picks 64 on vogue-768
     and 4 on openai3-3072 -- a factor of sixteen apart. Pinning 4 costs vogue
     0.08 recall; pinning 64 costs openai3-3072 a quarter of its throughput for
     no recall at all.
  2. The rule finds the oracle's alpha at S>=64 in all four cells.
  3. S=32 is not safe, and the paper's default has to change: 25% to 69% of
     random samples at S=32 lose more than 1e-3 of held-out recall. At S=128
     that falls to 0-8%.
  4. Removing the sampling does not help. The full-pool criterion is *more*
     conservative than a sample -- it picks 100 on vogue where the oracle is 64
     -- because epsilon is an absolute slot count, so its relative strictness
     tightens as the pool grows. That is a property of the criterion the paper
     states but does not draw out.
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style


def load():
    cells = collections.OrderedDict()
    cur = None
    for ln in open(style.datafile("budget_arms.log"), errors="ignore"):
        m = re.match(r"### (\S+) M=(\d+) nlist=\d+ np=(\d+)", ln)
        if m:
            cur = (m.group(1), int(m.group(3)))
            cells[cur] = {"M": int(m.group(2)), "fixed": [], "rule": {}}
            continue
        if cur is None:
            continue
        m = re.match(r"ARM FIXED alpha=([\d.]+)\s+recall=([\d.]+) qps=(\d+)", ln)
        if m:
            cells[cur]["fixed"].append((float(m.group(1)), float(m.group(2)),
                                        int(m.group(3))))
        m = re.match(r"ARM ORACLE alpha=([\d.]+) tau=([\d.]+) ceiling=([\d.]+) "
                     r"recall=([\d.]+) qps=(\d+)", ln)
        if m:
            cells[cur]["oracle"] = (float(m.group(1)), float(m.group(3)),
                                    float(m.group(4)), int(m.group(5)))
        m = re.match(r"ARM FULL alpha=([\d.]+) recall=([\d.]+) qps=(\d+) "
                     r"cal_ms=([\d.]+)", ln)
        if m:
            cells[cur]["full"] = (float(m.group(1)), float(m.group(2)),
                                  int(m.group(3)), float(m.group(4)))
        m = re.match(r"ARM RULE S=(\d+)\s+reps=\d+ cal_ms=([\d.]+) "
                     r"mean_loss=([\d.]+) p95_loss=([\d.]+) "
                     r"frac_over_1e-3=([\d.]+) median_alpha=(\d+) "
                     r"mean_qps=(\d+)", ln)
        if m:
            cells[cur]["rule"][int(m.group(1))] = dict(
                cal_ms=float(m.group(2)), mean=float(m.group(3)),
                p95=float(m.group(4)), over=float(m.group(5)),
                alpha=int(m.group(6)), qps=int(m.group(7)))
    return cells


def main():
    C = load()
    print("1. Would one fixed alpha do?  The oracle's choice, per cell:\n")
    oracles = {}
    for (ds, np_), c in C.items():
        a, ceil, rec, q = c["oracle"]
        oracles[(ds, np_)] = a
        print("   %-14s np=%-4d  oracle alpha=%-4.0f recall=%.4f qps=%6d "
              "(ceiling %.4f)" % (style.PRETTY[ds], np_, a, rec, q, ceil))
    print("\n   The cost of pinning one alpha everywhere:")
    for cand in sorted({a for a in oracles.values()}):
        worst_r, worst_q, where_r, where_q = 0.0, 0.0, "", ""
        for (ds, np_), c in C.items():
            fx = {a: (r, q) for a, r, q in c["fixed"]}
            _, ceil, orec, oq = c["oracle"]
            if cand not in fx:
                continue
            dr = ceil - fx[cand][0]
            dq = 1.0 - fx[cand][1] / oq
            if dr > worst_r:
                worst_r, where_r = dr, "%s np=%d" % (style.PRETTY[ds], np_)
            if dq > worst_q:
                worst_q, where_q = dq, "%s np=%d" % (style.PRETTY[ds], np_)
        print("     alpha=%-4.0f worst recall given up %.4f (%s); "
              "worst throughput given up %.0f%% (%s)"
              % (cand, worst_r, where_r, 100 * worst_q, where_q))

    print("\n2. Does the rule find it?  Median alpha over 64 random samples:\n")
    print("   %-14s %-5s %-8s %s" % ("dataset", "np", "oracle", "S=32 / 64 / 128"))
    for (ds, np_), c in C.items():
        print("   %-14s %-5d %-8.0f %s" % (
            style.PRETTY[ds], np_, c["oracle"][0],
            " / ".join(str(c["rule"][s]["alpha"]) for s in sorted(c["rule"]))))

    print("\n3. What the sample risks, on held-out queries:\n")
    print("   %-14s %-5s %-5s %-9s %-9s %s"
          % ("dataset", "np", "S", "mean loss", "p95 loss", "draws losing >1e-3"))
    allover = collections.defaultdict(list)
    for (ds, np_), c in C.items():
        for s in sorted(c["rule"]):
            r = c["rule"][s]
            allover[s].append(r["over"])
            print("   %-14s %-5d %-5d %-9.4f %-9.4f %.0f%%"
                  % (style.PRETTY[ds], np_, s, r["mean"], r["p95"],
                     100 * r["over"]))
    print()
    for s in sorted(allover):
        print("   S=%-4d draws losing more than 1e-3: %.0f%% to %.0f%% across cells"
              % (s, 100 * min(allover[s]), 100 * max(allover[s])))

    print("\n4. Is the sampling the weak part?  Full pool against S=128:\n")
    for (ds, np_), c in C.items():
        fa, _, fq, fcal = c["full"]
        r = c["rule"][max(c["rule"])]
        print("   %-14s np=%-4d oracle=%-4.0f full-pool picks %-4.0f (%.1f ms), "
              "S=%d picks %-4d (%.1f ms)"
              % (style.PRETTY[ds], np_, c["oracle"][0], fa, fcal,
                 max(c["rule"]), r["alpha"], r["cal_ms"]))

    print("\n5. Payback of the rule at S=128 against the alpha=100 default:\n")
    for (ds, np_), c in C.items():
        fx = {a: (r, q) for a, r, q in c["fixed"]}
        if 100.0 not in fx:
            continue
        r = c["rule"][max(c["rule"])]
        base_q, rule_q = fx[100.0][1], r["qps"]
        if rule_q <= base_q:
            print("   %-14s np=%-4d no gain over alpha=100" % (style.PRETTY[ds], np_))
            continue
        nq = 500.0                      # the held-out half is the timed batch
        per = nq * (1.0 / base_q - 1.0 / rule_q) * 1000.0   # ms saved a batch
        print("   %-14s np=%-4d gain %.3fx, calibration %.1f ms, payback %.1f batches"
              % (style.PRETTY[ds], np_, rule_q / base_q, r["cal_ms"],
                 r["cal_ms"] / per))


if __name__ == "__main__":
    main()
