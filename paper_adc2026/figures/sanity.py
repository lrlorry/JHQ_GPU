#!/usr/bin/env python3
"""Cheap invariants over a fresh log, run before anything is believed.

    python3 sanity.py ../data/batch3.log ../data/cpu3.log

Both re-runs in flight this session exist because a result sat on screen for
an hour looking plausible. Neither needed a clever check to catch.

* QPS is non-decreasing in batch and non-increasing in nprobe. A row that
  breaks either is not a finding, it is a measurement that failed. The RaBitQ
  column of batch2.log breaks both -- 140,297 to 27,802 from batch 512 to 1024
  -- because a 20-core CPU job shared the host, and the timed region is
  host-queries-in to host-results-out.
* Recall on the same dataset must agree between CPU and GPU where the
  parameters match. The CPU bench read 1.0000 at nprobe=16 on vogue-768
  against 0.9939 at nprobe=1024 on the GPU, because its Recall@10 scanned the
  whole 100-wide ground-truth row.

Exit code 1 if anything fails, so a runner can stop instead of producing four
more hours of numbers nobody can use.
"""
import sys, re, collections

FAIL = []


def note(msg):
    FAIL.append(msg); print("  FAIL " + msg)


def monotone(path):
    """QPS up with batch, down with nprobe, per (system, dataset)."""
    d = collections.defaultdict(dict)
    for ln in open(path):
        m = re.search(r"^\s*(\S+)\s+(\S+)\s+np=(\d+)\s+(?:a=\S+\s+)?"
                      r"batch=(\d+)\s+recall=([\d.]+)\s+qps=(\d+)", ln)
        if m:
            sysn, ds, np_, b, _, q = m.groups()
            d[(sysn, ds)][(int(np_), int(b))] = int(q)
    if not d:
        print("  (no batch rows)"); return
    for key, pts in sorted(d.items()):
        nps = sorted({k[0] for k in pts})
        bs = sorted({k[1] for k in pts})
        for np_ in nps:                       # QPS should rise with batch
            row = [(b, pts[(np_, b)]) for b in bs if (np_, b) in pts]
            for (b0, q0), (b1, q1) in zip(row, row[1:]):
                if q1 < q0 * 0.85:
                    note("%s %s np=%d: qps falls %d->%d from batch %d to %d"
                         % (key[0], key[1], np_, q0, q1, b0, b1))
        for b in bs:                          # and fall with nprobe
            col = [(n, pts[(n, b)]) for n in nps if (n, b) in pts]
            for (n0, q0), (n1, q1) in zip(col, col[1:]):
                if q1 > q0 * 1.15:
                    note("%s %s batch=%d: qps rises %d->%d from np=%d to %d"
                         % (key[0], key[1], b, q0, q1, n0, n1))


# what the GPU measures, for the datasets the CPU bench also runs
GPU_REF = {("vogue-768", 128): 0.9645, ("vogue-768", 1024): 0.9939,
           ("openai3-3072", 128): 0.9414, ("openai3-3072", 1024): 0.9926}


def cpu_recall(path):
    """The CPU's Recall@10 against the GPU's, where nprobe matches."""
    tag, arm, seen = None, None, False
    for ln in open(path):
        m = re.match(r"### (\S+) M=", ln)
        if m:
            tag, arm = m.group(1), None; continue
        t = ln.strip()
        if t.startswith("JHQ"):
            arm = "JHQ"; continue
        if t.startswith("JQ "):
            arm = "JQ"; continue
        m = re.match(r"\s*(\d+)\s+([\d.]+)\s+[\d.]+\s*$", ln)
        if m and arm == "JHQ" and tag:
            np_, r = int(m.group(1)), float(m.group(2))
            ref = GPU_REF.get((tag, np_))
            if ref is not None:
                seen = True
                if abs(r - ref) > 0.02:
                    note("%s JHQ np=%d: CPU recall %.4f vs GPU %.4f -- the two "
                         "sides are not measuring the same thing"
                         % (tag, np_, r, ref))
    if not seen:
        print("  (no CPU rows at an nprobe the GPU reference covers)")


for p in sys.argv[1:]:
    print(p)
    try:
        monotone(p); cpu_recall(p)
    except FileNotFoundError:
        print("  (not written yet)")
print("\n%d failure(s)" % len(FAIL))
sys.exit(1 if FAIL else 0)
