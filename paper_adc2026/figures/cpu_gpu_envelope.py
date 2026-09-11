#!/usr/bin/env python3
"""CPU-vs-GPU speedup with *both* sides' (alpha, nprobe) envelopes.

Why this exists: the first CPU baseline ran only alpha=100, so it compared the
GPU's chosen alpha against a CPU pinned to the most expensive one.  Sweeping
alpha on the CPU too turns out to move the answer by 0.3-42%, not by the ~25x
the excess residual work suggested -- because the CPU spends most of its time
in the scan, so refinement is a smaller share of its total.  That asymmetry is
itself the result worth reporting (see alpha_lever() below).

Three things this script refuses to do, each of which produced a wrong number
once already:

  * mix GPU binaries.  paper_fronts.log is v57, alpha_ds.log is v53, alpha6.log
    is v47.  A "GPU envelope" unioned across them is not a measurement of any
    build.  Only v57 sources are read here.
  * compare `max(qps | recall >= R)` across two sparse grids.  That estimator
    reads off whichever grid happens to have a point just above R, and it made
    vogue jump from 42x at R>=0.98 to 81x at R>=0.99.  Both fronts are
    interpolated at the *same* recall instead.
  * extrapolate.  The CPU envelope stops at R=0.9911 (vogue) / 0.9904
    (openai3-3072); GPU points above that are reported as out of range, not as
    a ratio against the CPU's last point.

And one thing it drops: CPU rows with nprobe < 128.  At nprobe <= 64 the 1000
queries finish fast enough that OpenMP spin-up dominates the timer and QPS
*rises* with nprobe -- 12,857 -> 53,686 QPS from nprobe 1 to 2 on vogue.  The
recall column is fine there; the QPS column is not.
"""
import collections
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data")

# cpu5_remeasured.log is cpu5.log with one cell replaced: vogue-768, 32 threads,
# nprobe=1024, alpha in {32, 100}.  That cell carries the top vogue ratio, and
# cpu5.log measured it once -- at alpha=100 it drew 929 QPS, which five repeats
# (data/anchors5.log) place at the bottom of a 765-1015 spread whose median is
# 1003.  Every other cell is cpu5.log's single measurement, unchanged.
CPU_LOG = (sys.argv[1] if len(sys.argv) > 1
           else os.path.join(DATA, "cpu6.log"))

CPU_NPROBE_FLOOR = 128
GPU_V57 = ["paper_fronts.log", "openai3072_v57_front.log"]


def read_cpu(path):
    """(dataset, threads, nprobe) -> {alpha: (recall, qps)}, JHQ rows only.

    Each invocation prints two tables: JQ (M bytes/vec, primary code only) and
    JHQ (M + d*Br/8 bytes/vec, with residuals).  Only the second is the method
    under test; a parser that keys on the numeric row shape alone silently
    keeps whichever came last.
    """
    out = collections.defaultdict(dict)
    inv = meth = None
    for ln in open(path, errors="ignore"):
        m = re.match(r"### (\S+) .*threads=(\d+)", ln)
        if m:
            inv = (m.group(1), int(m.group(2)))
            continue
        m = re.match(r"^(JQ|JHQ) \(", ln)
        if m:
            meth = m.group(1)
            continue
        m = re.match(r"\s*([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", ln)
        if m and inv and meth == "JHQ":
            a, np_, r, q = (float(m.group(1)), int(m.group(2)),
                            float(m.group(3)), float(m.group(4)))
            out[(inv[0], inv[1], np_)][a] = (r, q)
    return out


def read_gpu():
    """dataset -> arm -> [(recall, qps)] from the v57 logs only."""
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for ln in open(os.path.join(DATA, "paper_fronts.log"), errors="ignore"):
        m = re.search(r"FIX\s+(\S+).*np=(\d+)\s+a=100 recall=([\d.]+)\s+qps=(\d+)", ln)
        if m:
            g[m.group(1)]["FIX a=100"].append((float(m.group(3)), int(m.group(4))))
        m = re.search(r"RULE\s+(\S+).*np=(\d+).*recall_picked=([\d.]+)\s+qps_picked=(\d+)", ln)
        if m:
            g[m.group(1)]["RULE"].append((float(m.group(3)), int(m.group(4))))
    p = os.path.join(DATA, "openai3072_v57_front.log")
    for ln in open(p, errors="ignore"):
        m = re.search(r"v57 (\S+)\s+np=(\d+)\s+a=([\d.]+)\s+recall=([\d.]+)\s+qps=(\d+)", ln)
        if m:
            arm = "FIX a=100" if float(m.group(3)) == 100.0 else "ENV alpha<=8"
            g[m.group(1)][arm].append((float(m.group(4)), int(m.group(5))))
    return g


# Recall@10 over 1000 queries has resolution 1e-4, and one ID position moving
# under a tie is worth exactly that.  A raw Pareto front lets such a point win:
# the 16-thread openai3-3072 run at nprobe=256 reads R=0.9633 / 541 QPS against
# the 32-thread run's R=0.9632 / 759 QPS, so it is "optimal" by 1e-4 of recall
# while giving up 29% of throughput -- and it pulled the reported speedup at
# R=0.9647 from 76x to 102x.  Buckets make a point earn its place by a recall
# margin larger than the documented noise floor.
RECALL_EPS = 1e-3


def front(pts):
    """Pareto front over (recall, qps), recall bucketed at RECALL_EPS.

    Within a bucket only the fastest point survives, so a tie-break-sized
    recall difference cannot buy a place on the front.
    """
    best = {}
    for t in pts:
        b = round(t[0] / RECALL_EPS)
        if b not in best or t[1] > best[b][1]:
            best[b] = t
    out = []
    for t in sorted(best.values(), key=lambda t: (-t[0], -t[1])):
        if not out or t[1] > out[-1][1]:
            out.append(t)
    return sorted(out)


def interp(f, R):
    """Log-linear QPS at recall R on front f, or None outside its range."""
    if not f or R < f[0][0] or R > f[-1][0]:
        return None
    for i in range(1, len(f)):
        if f[i][0] >= R:
            (r0, q0), (r1, q1) = f[i - 1][:2], f[i][:2]
            if r1 == r0:
                return max(q0, q1)
            t = (R - r0) / (r1 - r0)
            return math.exp(math.log(q0) + t * (math.log(q1) - math.log(q0)))
    return None


def alpha_lever(rows, label):
    """How much throughput alpha=100 gives up, at equal recall.

    'Equal' means within 3e-3, one recall grid step, of what alpha=100 reaches:
    the cheapest alpha that still lands there is the one the rule would pick.
    """
    print("\n%s -- cost of pinning alpha=100, at matched recall:" % label)
    for key in sorted(rows):
        d = rows[key]
        if 100.0 not in d:
            continue
        r100, q100 = d[100.0]
        cand = [(a, r, q) for a, (r, q) in d.items() if r >= r100 - 3e-3]
        a, r, q = max(cand, key=lambda t: t[2])
        print("    %-30s a=100: R=%.4f Q=%8.0f | a=%-5g R=%.4f Q=%8.0f  -> %+5.1f%%"
              % (key, r100, q100, a, r, q, 100.0 * (q / q100 - 1.0)))


def main():
    cpu = read_cpu(CPU_LOG)
    gpu = read_gpu()

    cpu_pts = collections.defaultdict(list)
    for (ds, th, np_), d in cpu.items():
        if np_ < CPU_NPROBE_FLOOR:
            continue
        for a, (r, q) in d.items():
            cpu_pts[ds].append((r, q, "a=%g np=%d t%d" % (a, np_, th)))

    for ds in sorted(cpu_pts):
        cf = front(cpu_pts[ds])
        print("=" * 78)
        print("%s -- CPU envelope, nprobe>=%d, alpha and threads both swept"
              % (ds, CPU_NPROBE_FLOOR))
        for r, q, s in cf:
            print("    R=%.4f %8.0f qps   %s" % (r, q, s))
        # A second denominator, because the body describes one comparison as
        # "pinning alpha=100 on both sides" and the envelope above is not that:
        # it is swept over alpha *and* threads.  Quoting the envelope's ratio
        # under that sentence describes a comparison nobody ran.  Both are
        # printed and the body says which it means.
        pf = front([(r, q, "a=100 np=%d t%d" % (np_, th))
                    for (d2, th, np_), d in cpu.items()
                    if d2 == ds and np_ >= CPU_NPROBE_FLOOR and 100.0 in d
                    for r, q in [d[100.0]]])
        print("    (CPU pinned at alpha=100, threads swept: %d points, "
              "max R=%.4f)" % (len(pf), pf[-1][0] if pf else float("nan")))
        for arm in sorted(gpu.get(ds, {})):
            gf = front([(r, q, arm) for r, q in gpu[ds][arm]])
            print("  GPU %s vs that envelope, at matched recall:" % arm)
            for r, q, _ in gf:
                c = interp(cf, r)
                p100 = interp(pf, r)
                tail = ("" if p100 is None
                        else "   | CPU a=100 %7.0f = %5.1fx" % (p100, q / p100))
                if c is None:
                    print("    R=%.4f  GPU %8d   -- above the CPU envelope "
                          "(max R=%.4f)%s" % (r, q, cf[-1][0], tail))
                else:
                    print("    R=%.4f  GPU %8d   CPU %7.0f   = %5.1fx%s"
                          % (r, q, c, q / c, tail))

    # The asymmetry, on one dataset and one nprobe grid so it is a comparison
    # and not two separate observations.
    alpha_lever({("openai3-3072 CPU t32 np=%d" % np_): d
                 for (ds, th, np_), d in cpu.items()
                 if ds == "openai3-3072" and th == 32 and np_ >= CPU_NPROBE_FLOOR},
                "CPU, 32 threads")
    grows = collections.defaultdict(dict)
    for ln in open(os.path.join(DATA, "openai3072_v57_front.log"), errors="ignore"):
        m = re.search(r"v57 (\S+)\s+np=(\d+)\s+a=([\d.]+)\s+recall=([\d.]+)\s+qps=(\d+)", ln)
        if m:
            grows["openai3-3072 GPU np=%s" % m.group(2)][float(m.group(3))] = (
                float(m.group(4)), int(m.group(5)))
    alpha_lever(grows, "GPU, v57")


if __name__ == "__main__":
    main()
