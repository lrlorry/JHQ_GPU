#!/usr/bin/env python3
"""The three figures the argument rests on.

1. The crossover. Same comparison at three dimensions, so the trend is the
   subject rather than something the reader assembles from six panels.
2. Scale. Measured device memory against the card, including the two runs that
   ran out of it -- one of them ours.
3. The hierarchy. Residual level off, on at 4 bits, on at 8 bits, which
   separates what the quantiser contributes from what the kernel work does.
"""
import csv, glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "final")
JHQ, CAG, INT8, IVF, JQ = "#12507B", "#2E8B57", "#8A9A2B", "#E07B39", "#9B7FC7"
CARD_GIB = 31.36


def load(pats, ds):
    out = []
    for pat in pats:
        for f in sorted(glob.glob(os.path.join(D, pat.format(d=ds)))):
            with open(f) as fh:
                body = [l for l in fh if not l.startswith("#")]
            for r in csv.DictReader(body):
                if r["status"] != "ok":
                    continue
                try:
                    out.append((float(r["recall"]), float(r["qps_mean"]),
                                float(r["vram_mib"] or 0)))
                except ValueError:
                    pass
    return out


def front(pts):
    best, keep = 0.0, []
    for r, q, *_ in sorted(pts, key=lambda p: -p[0]):
        if q > best:
            best = q
            keep.append((r, q))
    return sorted(keep)


def crossing(a, b):
    """Recall where front a rises above front b, by linear scan on the union grid."""
    if not a or not b:
        return None
    xs = sorted({r for r, _ in a} | {r for r, _ in b})
    def at(f, x):
        pts = [p for p in f if p[0] <= x] or [f[0]]
        return pts[-1][1]
    prev = None
    for x in xs:
        if x < max(a[0][0], b[0][0]):
            continue
        d = at(a, x) - at(b, x)
        if prev is not None and prev < 0 <= d:
            return x
        prev = d
    return None


# ── 1. the crossover, one panel per dimension ────────────────────────────────
def fig_crossover():
    panels = [("vogue", 768, "Vogue-768"), ("openai3-1536", 1536, "OpenAI3-1536"),
              ("openai3-3072", 3072, "OpenAI3-3072")]
    series = [("JHQ (this work)", JHQ, "o", "-", 2.8, 9,
               ["p0_{d}*jhq*.csv", "sat_{d}*jhq*.csv"]),
              ("cuVS CAGRA fp32", CAG, "D", "-", 2.0, 7,
               ["p0_{d}*_cagra.csv", "sat_{d}*_cagra.csv"]),
              ("cuVS CAGRA int8", INT8, "^", "--", 1.7, 6,
               ["p0_{d}*cagra_int8.csv", "sat_{d}*cagra_int8.csv"]),
              ("cuVS IVF-PQ", IVF, "s", ":", 1.5, 5,
               ["p0_{d}*ivfpq*.csv", "sat_{d}*ivfpq*.csv"])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=False)
    for ax, (ds, dim, title) in zip(axes, panels):
        fronts = {}
        for label, colour, mk, ls, lw, ms, pats in series:
            f = [p for p in front(load(pats, ds)) if p[0] >= 0.86]
            if not f:
                continue
            fronts[label] = f
            ax.plot([p[0] for p in f], [p[1] for p in f], marker=mk, ls=ls, color=colour,
                    lw=lw, ms=ms, label=label, markeredgecolor="white", markeredgewidth=.6,
                    zorder=5 if label.startswith("JHQ") else 3)
        x = crossing(fronts.get("JHQ (this work)", []), fronts.get("cuVS CAGRA fp32", []))
        if x:
            ax.axvline(x, color="#A2452E", ls="-.", lw=1.6, zorder=2)
            ax.annotate(f"JHQ overtakes\nfp32 CAGRA\nat R={x:.3f}", (x, ax.get_ylim()[1]),
                        xytext=(-6, -46), textcoords="offset points", ha="right",
                        fontsize=9, color="#A2452E", weight="bold")
        else:
            ax.text(.97, .93, "fp32 CAGRA ahead\nthroughout this range",
                    transform=ax.transAxes, fontsize=9.5, color="#A2452E",
                    ha="right", va="top", weight="bold")
        ax.set_yscale("log")
        ax.set_xlim(0.86, 1.005)
        ax.set_title(f"{title}   ·   d = {dim}", fontsize=12.5)
        ax.set_xlabel("Recall@10", fontsize=10.5)
        ax.set_ylabel("QPS (batch 1000)", fontsize=10.5)
        ax.grid(alpha=.22, which="both")
    axes[0].legend(fontsize=9, loc="lower left", framealpha=.95)
    fig.suptitle("The same comparison at three dimensions: JHQ's distance work is O(M) with M "
                 "fixed, CAGRA's is O(d)", fontsize=13, y=.99)
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(os.path.join(D, "core1_crossover.png"), dpi=155)
    print("core1_crossover.png")


# ── 2. scale: what fits on one card ──────────────────────────────────────────
def fig_scale():
    sets = [("bge-m3", "BGE-M3\n10.1M × 1024", 10091524),
            ("stella-trec24", "Stella-TREC24\n17.8M × 1024", 17776615)]
    METH = [("JHQ Br=4", JHQ, ["p0_{d}_jhq_Br4.csv"]),
            ("JHQ Br=8", "#3D7FA6", ["p0_{d}_jhq_Br8.csv"]),
            ("cuVS IVF-PQ", IVF, ["p0_{d}_ivfpq.csv"]),
            ("CAGRA int8", INT8, ["p0_{d}_cagra_int8.csv"]),
            ("CAGRA fp32", CAG, ["p0_{d}_cagra.csv"])]

    def failure(ds, pats):
        for pat in pats:
            for f in glob.glob(os.path.join(D, pat.format(d=ds))):
                with open(f) as fh:
                    body = [l for l in fh if not l.startswith("#")]
                for r in csv.DictReader(body):
                    if r["status"] != "ok":
                        try:
                            msg = " ".join(x for p in json.loads(r["failures"] or "[]")
                                           for x in p[1])
                        except Exception:
                            msg = r["failures"] or ""
                        if "out_of_memory" in msg or "out of memory" in msg.lower():
                            g = 41.33 if "41334882304" in msg else None
                            return g
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for ax, (ds, title, n) in zip(axes, sets):
        names, vals, cols, notes = [], [], [], []
        for label, colour, pats in METH:
            pts = load(pats, ds)
            if pts:
                best = max(pts, key=lambda p: p[0])
                names.append(label); vals.append(best[2] / 1024); cols.append(colour)
                notes.append(f"R={best[0]:.4f}")
            else:
                g = failure(ds, pats)
                names.append(label); cols.append(colour)
                vals.append(g if g else 0)
                notes.append(f"OOM, asked {g:.1f} GiB" if g else
                             ("OOM during add" if "jhq" in pats[0] else "not run"))
        y = np.arange(len(names))[::-1]
        bars = ax.barh(y, vals, .6, color=cols, zorder=3,
                       hatch=["///" if "OOM" in nt else "" for nt in notes],
                       alpha=.95)
        for b, nt in zip(bars, notes):
            if "OOM" in nt:
                b.set_edgecolor("#A2452E"); b.set_linewidth(1.4); b.set_alpha(.45)
        for yy, v, nt in zip(y, vals, notes):
            ax.text(max(v, .3) + .5, yy, nt, va="center", fontsize=9.5,
                    color="#A2452E" if "OOM" in nt else "#222")
        ax.axvline(CARD_GIB, color="#A2452E", ls="--", lw=1.8, zorder=4)
        ax.text(CARD_GIB + .6, .15, "RTX 5090\n31.4 GiB", ha="left", va="bottom",
                fontsize=9.5, color="#A2452E")
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
        ax.set_xlim(0, 46)
        ax.set_xlabel("device memory held, GiB (measured after build)")
        ax.set_title(title, fontsize=12)
        # the point is not that others fail to fit -- int8 CAGRA and IVF-PQ both
        # do -- but that the ones which fit cap out below 0.95 while the one with
        # the higher ceiling cannot be loaded at all
        ok = [(nm, nt) for nm, nt in zip(names, notes) if nt.startswith("R=")]
        if ok:
            best = max(ok, key=lambda t: float(t[1][2:]))
            ax.text(.98, .04, f"highest recall here: {best[0]}  {best[1]}",
                    transform=ax.transAxes, ha="right", fontsize=9.5,
                    color="#12507B", weight="bold")
        ax.grid(alpha=.22, axis="x", zorder=0)
    fig.suptitle("At ten million vectors everything but fp32 CAGRA fits — and only JHQ still "
                 "reaches 0.95", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(os.path.join(D, "core2_scale.png"), dpi=155)
    print("core2_scale.png")


# ── 3. the hierarchy, three settings of the same index ───────────────────────
def fig_hierarchy():
    panels = [("vogue", "Vogue-768"), ("openai3-3072", "OpenAI3-3072"),
              ("bge-m3", "BGE-M3, 10.1M")]
    variants = [("residual off (JQ)", JQ, "v", ":", ["p0_{d}*_jq.csv"]),
                ("residual 4 bits/dim", JHQ, "o", "-",
                 ["p0_{d}*jhq*Br4*.csv", "p0_{d}_jhq_M96Br4.csv"]),
                ("residual 8 bits/dim", "#7EC8E3", "s", "-",
                 ["p0_{d}*jhq*Br8*.csv", "sat_{d}*jhq_Br8.csv"])]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5))
    for ax, (ds, title) in zip(axes, panels):
        for label, colour, mk, ls, pats in variants:
            f = front(load(pats, ds))
            if not f:
                continue
            ax.plot([p[0] for p in f], [p[1] for p in f], marker=mk, ls=ls, color=colour,
                    lw=2.4, ms=7, label=label, markeredgecolor="white", markeredgewidth=.6)
            rmax, qmax = max(f, key=lambda p: p[0])
            ax.annotate(f"{rmax:.3f}", (rmax, qmax), textcoords="offset points",
                        xytext=(6, -2), fontsize=9, color=colour, weight="bold",
                        annotation_clip=False)
        ax.axvline(0.95, color="#888", ls=":", lw=1.2)
        ax.set_yscale("log"); ax.set_xlim(0.55, 1.04)
        ax.set_xlabel("Recall@10", fontsize=10.5); ax.set_ylabel("QPS", fontsize=10.5)
        ax.set_title(title, fontsize=12)
        # the point is not that others fail to fit -- int8 CAGRA and IVF-PQ both
        # do -- but that the ones which fit cap out below 0.95 while the one with
        # the higher ceiling cannot be loaded at all
        ok = [(nm, nt) for nm, nt in zip(names, notes) if nt.startswith("R=")]
        if ok:
            best = max(ok, key=lambda t: float(t[1][2:]))
            ax.text(.98, .04, f"highest recall here: {best[0]}  {best[1]}",
                    transform=ax.transAxes, ha="right", fontsize=9.5,
                    color="#12507B", weight="bold"); ax.grid(alpha=.22, which="both")
    axes[0].legend(fontsize=9.5, loc="lower left")
    fig.suptitle("Same kernel, same IVF, same occupancy and selection — only the residual level "
                 "differs.  Without it nothing reaches 0.95.", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.savefig(os.path.join(D, "core3_hierarchy.png"), dpi=155)
    print("core3_hierarchy.png")


if __name__ == "__main__":
    fig_crossover(); fig_scale(); fig_hierarchy()
