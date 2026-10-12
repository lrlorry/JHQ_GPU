#!/usr/bin/env python3
"""How tight would a bound have to be? From the same per-candidate files.

A bound LB = rho * D_H prunes candidate x at the oracle threshold tau_full iff
rho * D_H(x) > tau_full, i.e. iff rho > tau_full / D_H(x). So the share of
probed candidates any bound of tightness rho could prune is the share with
D_H > tau_full / rho, and the tightness needed to prune a share f is the
f-quantile of tau_full / D_H. Both come from D_H alone: they hold for every
bound of that tightness, not just this one. The achieved tightness is the
measured LB_cond / D_H.

  python3 tightness.py <work dir> <results dir>
Writes tightness.json and figures/fig6_required_tightness.png.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load  # noqa: E402

K = 10


def main():
    W, R = sys.argv[1], sys.argv[2]
    tags = sorted(t for t in os.listdir(W) if not t.startswith("v16_") and os.path.isdir(os.path.join(W, t)))
    rhos = np.linspace(0.05, 1.0, 96)
    out = {}
    curves = {}
    for t in tags:
        ratio_all, lbr_all, dist_all = [], [], []
        prune = np.zeros(len(rhos))
        nq = 0
        for p in sorted(glob.glob(os.path.join(W, t, "rb_q*.bin"))):
            q = load(p)
            DH, LC = q["DH"], q["LC"]
            tau = float(np.sort(DH)[K - 1])
            r = tau / np.maximum(DH, 1e-300)
            ratio_all.append(r.astype(np.float32))
            dist_all.append((DH / tau).astype(np.float32))
            ok = DH > 1e-12
            lbr_all.append((LC[ok] / DH[ok]).astype(np.float32))
            prune += np.array([(DH * rho > tau).mean() for rho in rhos])
            nq += 1
        r = np.concatenate(ratio_all); lbr = np.concatenate(lbr_all); dd = np.concatenate(dist_all)
        prune /= nq
        need = {"prune_%d%%" % int(f * 100): float(np.quantile(r, f)) for f in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}
        out[t] = dict(queries=nq, required_tightness=need,
                      achieved_LBcond_over_DH={"p50": float(np.median(lbr)), "p99": float(np.quantile(lbr, 0.99)), "max": float(lbr.max())},
                      DH_over_tau={"p1": float(np.quantile(dd, 0.01)), "p10": float(np.quantile(dd, 0.1)), "p50": float(np.median(dd)),
                                   "p90": float(np.quantile(dd, 0.9)), "p99": float(np.quantile(dd, 0.99)), "max": float(dd.max())},
                      prunable_share_at_rho={"%.2f" % rho: float(v) for rho, v in zip(rhos, prune) if abs(rho * 20 - round(rho * 20)) < 1e-6})
        curves[t] = (prune, float(np.median(lbr)), float(lbr.max()))
        print(t, json.dumps(out[t]["required_tightness"]), "achieved median %.3f max %.3f" % (np.median(lbr), lbr.max()),
              "| D_H/tau p50 %.2f p99 %.2f max %.2f" % (np.median(dd), np.quantile(dd, 0.99), dd.max()))
    json.dump(out, open(os.path.join(R, "tightness.json"), "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for i, (t, (pr, med, mx)) in enumerate(curves.items()):
        c = plt.cm.tab10(i)
        ax.plot(rhos, 100 * pr, color=c, lw=1.5, label=t.replace("_np", " np").replace("_a", " α="))
        ax.axvline(mx, color=c, ls=":", lw=1)
    ax.axvspan(0.0, max(m for _, m, _ in curves.values()), color="#cccccc", alpha=0.5,
               label="achieved: LB_cond / D_H (median ... max, dotted)")
    ax.set_xlabel("bound tightness ρ = LB / D_H (a hypothetical bound at least this tight)")
    ax.set_ylabel("probed candidates it could prune at τ_full (%)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 100); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("How tight a bound must be, against how tight this one is", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(R, "figures", "fig6_required_tightness.png"), dpi=110)


if __name__ == "__main__":
    main()
