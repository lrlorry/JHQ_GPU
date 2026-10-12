"""Shared loading for the sequential-calibration replay.

Everything here is read-only over paper_adc2026/data/. Nothing in that tree is
written by this experiment.

Per-query inputs (paper_adc2026/data/alpha_perquery/, dumped by demo_jhq_v57,
see data/alpha_perquery.log):
    <ds>_np<np>_a<alpha>.ivecs   1000 x 10 returned ids, one frozen index per
                                 (ds, np), every alpha on the grid
    gt_<ds>.ivecs                ground truth; only the first k=10 are used

Two matrices are derived per configuration and used separately:
    DIS[a][q] = k - |R_a(q) n R_ref(q)|      label-free, what a policy may see
    REC[a][q] = |R_a(q) n GT(q)| / k         labels, held-out scoring only
"""
import os
import re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(REPO, "paper_adc2026", "data")
PQ = os.path.join(DATA, "alpha_perquery")

K = 10
GRID = [200, 100, 64, 32, 16, 8, 4, 2]       # descending, A[0] = alpha_max
CONFIGS = [("vogue-768", 128), ("vogue-768", 512),
           ("openai3-3072", 128), ("openai3-3072", 512)]


def ivecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    return raw.reshape(-1, d + 1)[:, 1:]


def _overlap(a, b):
    """|a_q n b_q| per row, set semantics (duplicate ids counted once, as the
    C++ loop and alpha_resample.py both do)."""
    return np.array([len(set(x) & set(y)) for x, y in zip(a, b)], dtype=np.int64)


class Config:
    """One frozen index at one nprobe: ids for every alpha, plus ground truth."""

    def __init__(self, ds, np_):
        self.ds, self.np = ds, np_
        self.name = "%s/np%d" % (ds, np_)
        self.ids = {a: ivecs(os.path.join(PQ, "%s_np%d_a%d.ivecs" % (ds, np_, a)))[:, :K]
                    for a in GRID}
        self.gt = ivecs(os.path.join(PQ, "gt_%s.ivecs" % ds))[:, :K]
        self.n = min(len(self.gt), *(len(v) for v in self.ids.values()))
        # labels: per-query recall@10 for every alpha
        self.rec = {a: _overlap(v[:self.n], self.gt[:self.n]) / float(K)
                    for a, v in self.ids.items()}
        self._dis = {}

    def dis(self, ref):
        """DIS[a][q] against reference alpha `ref` (label-free)."""
        if ref not in self._dis:
            r = self.ids[ref][:self.n]
            self._dis[ref] = {a: K - _overlap(v[:self.n], r) for a, v in self.ids.items()}
        return self._dis[ref]


def parse_arms_log(path):
    """ARM lines from a demo_jhq_budget_arms log, grouped by '### ds ... np=' header."""
    out, cur = {}, None
    for ln in open(path, errors="ignore"):
        m = re.match(r"### (\S+) M=\d+ nlist=(\d+) np=(\d+)", ln)
        if m:
            cur = (m.group(1), int(m.group(3)), int(m.group(2)))
            out[cur] = {"fixed": {}, "rule": {}, "enum": {}, "full": None, "oracle": None}
            continue
        if cur is None or not ln.startswith("ARM"):
            continue
        kv = dict(re.findall(r"(\w[\w\-\.]*)=([\-\d\.e/]+)", ln))
        if ln.startswith("ARM FIXED"):
            out[cur]["fixed"][int(float(kv["alpha"]))] = (float(kv["recall"]), float(kv["qps"]))
        elif ln.startswith("ARM RULE"):
            out[cur]["rule"][int(kv["S"])] = kv
        elif ln.startswith("ARM ENUM"):
            out[cur]["enum"][int(kv["S"])] = kv
        elif ln.startswith("ARM FULL"):
            out[cur]["full"] = kv
        elif ln.startswith("ARM ORACLE"):
            out[cur]["oracle"] = kv
    return out
