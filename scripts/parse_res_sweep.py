#!/usr/bin/env python3
"""Turn the residual-training sweep log into a table and the 100K-vs-ALL deltas."""
import re, sys, collections

def parse(path):
    cur = None; rec = qps = tr = None; rows = []
    for line in open(path):
        m = re.match(r"--- (\S+) ---", line)
        if m:
            cur, rec, qps, tr = m.group(1), None, None, None
            continue
        if not cur:
            continue
        r = re.match(r"Recall@10 *: *([0-9.]+)", line)
        if r: rec = float(r.group(1))
        q = re.match(r"QPS *: *([0-9]+)", line)
        if q: qps = int(q.group(1))
        t = re.match(r"\s*train: *([0-9.]+)", line)
        if t: tr = float(t.group(1))
        if rec is not None and qps is not None:
            mm = re.match(r"(\w+?)_Br(\d+)_nt(\d+)_np(\d+)$", cur)
            if mm:
                rows.append(dict(ds=mm.group(1), br=int(mm.group(2)),
                                 nt=int(mm.group(3)), np=int(mm.group(4)),
                                 recall=rec, qps=qps, train=tr))
            cur = None
    return rows

rows = parse(sys.argv[1] if len(sys.argv) > 1 else "/root/rt.log")
print(f"{'dataset':<8}{'Br':>3}{'n_res':>9}{'np':>5}{'Recall@10':>11}{'QPS':>9}{'train_ms':>10}")
for r in sorted(rows, key=lambda x: (x['ds'], -x['br'], x['np'], x['nt'])):
    print(f"{r['ds']:<8}{r['br']:>3}{r['nt']:>9}{r['np']:>5}"
          f"{r['recall']:>11.4f}{r['qps']:>9}{(r['train'] or 0):>10.0f}")

print()
print("=== Recall@10 delta against ALL, per (dataset, Br, nprobe) ===")
by = collections.defaultdict(dict)
for r in rows:
    by[(r['ds'], r['br'], r['np'])][r['nt']] = r['recall']
worst = collections.defaultdict(float)
print(f"{'dataset':<8}{'Br':>3}{'np':>5}   " +
      "".join(f"{n:>11}" for n in ("10K", "50K", "100K", "250K", "500K")) + f"{'ALL':>11}")
for key in sorted(by):
    d = by[key]
    if not d: continue
    alln = max(d)
    base = d[alln]
    line = f"{key[0]:<8}{key[1]:>3}{key[2]:>5}   "
    for n in sorted(d):
        if n == alln: continue
        line += f"{d[n]-base:>+11.4f}"
        worst[(key[0], key[1])] = max(worst[(key[0], key[1])], abs(d[n]-base))
    line += f"{base:>11.4f}"
    print(line)

print()
print("=== |100K - ALL|, the quantity the decision rule asks for ===")
mx = 0.0
for key in sorted(by):
    d = by[key]
    if not d: continue
    alln = max(d)
    if 100000 in d:
        delta = abs(d[100000] - d[alln])
        mx = max(mx, delta)
        print(f"  {key[0]:<8} Br={key[1]} nprobe={key[2]:<4} "
              f"100K={d[100000]:.4f}  ALL={d[alln]:.4f}  |delta|={delta:.4f}")
band = ("< 0.0005" if mx < 0.0005 else "0.0005-0.001" if mx < 0.001
        else "0.001-0.005" if mx < 0.005 else "> 0.005")
print(f"\n  max |100K - ALL| over all configurations = {mx:.4f}   band: {band}")
