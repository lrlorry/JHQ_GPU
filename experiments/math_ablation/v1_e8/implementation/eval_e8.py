#!/usr/bin/env python3
"""V1 offline evaluation on the paper index's probed candidates (calibration + validation; --test once at the end).

  python3 eval_e8.py <out dir> [--test]

Methods (JHQ = the paper index itself: its FP64 D_prim orders, its D_H refines):
  jhq_re        JHQ's primary (+-c cube) with the residual retrained offline (the rank-optimal study's var_jhq_re):
                the reference for anything with a retrained residual
  E8|JHQ-DH     the E8 primary orders the candidates, the paper index's own D_H refines: the primary alone
  E8            the E8 primary and its own retrained residual (the complete V1 representation)
  E8g           the same with one scale per subspace
Outputs: eval_e8.csv / .json, eval_e8_<tag>_perquery.npz, paired differences in the json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import ma_common as MC  # noqa: E402
from e8 import CB  # noqa: E402

C = MC.C
V = {}


def e8_dist(vd, own_dh=True):
    P = V[vd]

    def f(q, qr, pos):
        M, s = P["M"], P["scales"]
        qm = qr.reshape(M, 8)
        T = (qm ** 2).sum(1)[:, None] - 2 * s[:, None] * (qm @ CB.T) + (s ** 2)[:, None] * (CB ** 2).sum(1)[None, :]
        n = len(pos)
        key = np.empty(n); dh = np.empty(n)
        rows = np.arange(M)[None, :]
        off = (np.arange(M) * 256).repeat(8)[None, :]
        cbf = P["res_cb"].reshape(-1)
        for a in range(0, n, 4096):
            ps = pos[a:a + 4096]
            cd = np.asarray(P["codes"][ps]).astype(np.int64)
            key[a:a + 4096] = T[rows, cd].sum(1)
            if own_dh:
                yh = (s[None, :, None] * CB[cd]).reshape(len(ps), -1)
                rh = cbf[np.asarray(P["res"][ps]).astype(np.int64) + off]
                e = qr[None, :] - yh - rh
                dh[a:a + 4096] = np.einsum("ij,ij->i", e, e)
        return key, (dh if own_dh else q["DH"])
    return f


def jhq_re_dist(tag):
    m = C.meta(tag)
    N, d, M = m["N"], m["d"], m["M"]
    vd = os.path.join(C.RQ, tag, "var_jhq_re")
    pz = np.load(os.path.join(vd, "params.npz"))
    lo, hi, cb = pz["lo"].astype(np.float64), pz["hi"].astype(np.float64), pz["res_cb"].astype(np.float64)
    bits_mm = np.memmap(os.path.join(vd, "bits.u8"), dtype=np.uint8, mode="r", shape=(N, M))
    res_mm = np.memmap(os.path.join(vd, "res.u8"), dtype=np.uint8, mode="r", shape=(N, d))
    off = (np.arange(M) * cb.shape[1]).repeat(8)[None, :]
    cbf = cb.reshape(-1)

    def f(q, qr, pos):
        n = len(pos)
        dp = np.empty(n); dh = np.empty(n)
        for a in range(0, n, 4096):
            ps = pos[a:a + 4096]
            bits = np.unpackbits(np.asarray(bits_mm[ps]), axis=1).astype(bool)
            yh = np.where(bits, hi[None, :], lo[None, :])
            rh = cbf[np.asarray(res_mm[ps]).astype(np.int64) + off]
            e = qr[None, :] - yh
            dp[a:a + 4096] = np.einsum("ij,ij->i", e, e)
            e -= rh
            dh[a:a + 4096] = np.einsum("ij,ij->i", e, e)
        return dp, dh
    return f


def main():
    out = sys.argv[1]
    test = "--test" in sys.argv
    os.makedirs(out, exist_ok=True)
    rows, summ = [], {}
    label = "eval_e8" + ("_test" if test else "")
    for tag in C.TAGS:
        m = C.meta(tag)
        N, d, M = m["N"], m["d"], m["M"]
        V.clear()
        methods = {"jhq_re": jhq_re_dist(tag)}
        for name, sub in (("E8", "v1_e8"), ("E8g", "v1_e8_grouped")):
            vd = os.path.join(MC.MA, tag, sub)
            if not os.path.exists(os.path.join(vd, "params.npz")):
                continue
            pz = np.load(os.path.join(vd, "params.npz"))
            V[sub] = dict(M=M, scales=pz["scales"].astype(np.float64), res_cb=pz["res_cb"].astype(np.float64),
                          codes=np.memmap(os.path.join(vd, "codes.u8"), dtype=np.uint8, mode="r", shape=(N, M)),
                          res=np.memmap(os.path.join(vd, "res.u8"), dtype=np.uint8, mode="r", shape=(N, d)))
            if name == "E8":
                methods["E8|JHQ-DH"] = e8_dist(sub, own_dh=False)
            methods[name] = e8_dist(sub)
        r, s = MC.evaluate(out, label, tag, methods, test=test)
        rows += r
        s["paired_calib_val"] = {a + " - " + b: MC.paired(out, label, tag, a, b) for a, b in
                                 (("E8|JHQ-DH", "JHQ"), ("E8", "jhq_re"), ("E8", "JHQ"), ("E8g", "E8"), ("jhq_re", "JHQ")) if (a in methods or a == "JHQ") and (b in methods or b == "JHQ")}
        if test:
            s["paired_test"] = {a + " - " + b: MC.paired(out, label, tag, a, b, splits=("test",)) for a, b in (("E8", "jhq_re"), ("E8", "JHQ"))}
        summ[tag] = s
        print(tag, json.dumps({k: v for k, v in s["methods"].items()}), flush=True)
        print("  paired", json.dumps(s["paired_calib_val"]), flush=True)
        MC.write_rows(os.path.join(out, label + ".csv"), rows)
        json.dump(summ, open(os.path.join(out, label + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
