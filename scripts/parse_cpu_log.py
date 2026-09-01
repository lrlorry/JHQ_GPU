#!/usr/bin/env python3
"""Turn the CPU JHQ sweep log into the same CSV schema the GPU harness writes.

The demo prints two tables per build -- JHQ-CPU-IVF (primary + residual) and
JQ-CPU-IVF (primary only) -- under a "###### <dataset> threads=<n>" banner the
runner emits. Both recall columns are kept: `recall` is the set intersection
against the true top-k, `recall_official` is the looser measure the reference
implementation uses, which tests each returned id for membership anywhere in
the ground-truth row and is therefore not comparable across datasets whose
ground truth differs in width.
"""
import csv, re, sys

BANNER = re.compile(r"^######\s+(\S+)\s+threads=(\S+)\s+M=(\d+)\s+nlist=(\d+)")
METHOD = re.compile(r"^===\s*(\S+)\s*===")
ROW    = re.compile(r"^(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")

def parse(path):
    ds = th = method = None
    M = nlist = ""
    out = []
    for line in open(path, errors="replace"):
        b = BANNER.match(line)
        if b:
            ds, th, M, nlist = b.groups(); method = None; continue
        m = METHOD.match(line)
        if m:
            method = m.group(1); continue
        r = ROW.match(line)
        if r and ds and method:
            nprobe, recall, official, qps = r.groups()
            out.append(dict(method=method, dataset=ds, threads=th, M=M,
                            nlist=nlist, nprobe=int(nprobe),
                            recall=f"{float(recall):.4f}",
                            recall_official=f"{float(official):.4f}",
                            qps=f"{float(qps):.1f}"))
    return out

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rows = parse(sys.argv[1])
    w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]) if rows else
                       ["method","dataset","threads","M","nlist","nprobe",
                        "recall","recall_official","qps"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
    print(f"# {len(rows)} rows", file=sys.stderr)
