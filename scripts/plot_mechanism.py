#!/usr/bin/env python3
"""The figures that explain why, rather than restating what.

The comparison figures say JHQ is ahead in a particular regime. These say what
produced that and where the design is pinned: the three changes that made the
scan faster, where the build time went, what the two selection parameters buy,
and the shared-memory wall that stops the obvious next step.

Everything is read from results/final/*.csv, including the sweeps that predate
the harness -- scripts/parse_stage_logs.py turns those logs into CSVs first.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "final")
BLUE, GREY, RED, GREEN = "#12507B", "#9AA5B1", "#A2452E", "#2E6B50"


def rows(name):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        # harness CSVs carry a leading "# {...}" line of provenance
        body = [l for l in fh if not l.startswith("#")]
    return list(csv.DictReader(body))


def num(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


# ── 3. where the scan speedup comes from ─────────────────────────────────────
def fig_ablation():
    oc = rows("abl_occupancy_cascade.csv")
    def find(v, b, p):
        for r in oc:
            if r["variant"] == v and r["block"] == b and r["prefix"] == p:
                return num(r["qps"]), num(r["recall"])
        return None, None
    base, rbase = find("v16", "256", "-")
    off_1024, r1 = find("v21", "1024", "1/1")
    on_1024, r2 = find("v21", "1024", "1/2")
    # the bitonic step is measured in the harness runs, same dataset and settings
    bit = max((num(r["qps_mean"]) for r in rows("p0_vogue_jhq_M96Br4.csv")
               if r["status"] == "ok" and abs(num(r["recall"], 0) - 0.945) < 0.004),
              default=None)
    steps = [("v16 published port\nBLOCK=256", base, rbase),
             ("+ BLOCK=1024\noccupancy 17% → 67%", off_1024, r1),
             ("+ cascade, prefix 1/2\nhalf the table lookups", on_1024, r2)]
    if bit:
        steps.append(("+ bitonic selection\n12000 barriers → 78", bit, 0.9452))

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ys = np.arange(len(steps))[::-1]
    vals = [s[1] for s in steps]
    ax.barh(ys, vals, .62, color=[GREY] + [BLUE] * (len(steps) - 1), zorder=3)
    for y, (label, v, rec) in zip(ys, steps):
        ax.text(v + base * .04, y, f"{v:,.0f} QPS   {v/base:.2f}×"
                + (f"   R={rec:.4f}" if rec else ""), va="center", fontsize=9.5)
    ax.set_yticks(ys); ax.set_yticklabels([s[0] for s in steps], fontsize=9.5)
    ax.set_xlim(0, max(vals) * 1.5)
    ax.set_xlabel("QPS at Recall@10 ≈ 0.945, nprobe=128, batch 1000 — Vogue-768")
    ax.set_title("Where the scan speedup comes from")
    ax.grid(alpha=.25, axis="x", zorder=0)
    fig.tight_layout(); fig.savefig(os.path.join(D, "fig3_ablation.png"), dpi=160)
    print("fig3_ablation.png")


# ── 4. build time by phase ───────────────────────────────────────────────────
def fig_build():
    rs = [r for r in rows("build_phases.csv") if r["phase"] != "total"]
    if not rs:
        return
    labels = [r["phase"] for r in rs]
    before = [num(r["before_ms"], 0) / 1000 for r in rs]
    after = [num(r["after_ms"], 0) / 1000 for r in rs]
    y = np.arange(len(rs))[::-1]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.barh(y + .19, before, .36, color=GREY, label="before", zorder=3)
    ax.barh(y - .19, after, .36, color=BLUE, label="after", zorder=3)
    for yy, b, a in zip(y, before, after):
        if b: ax.text(b + .15, yy + .19, f"{b:.1f}s", va="center", fontsize=8.5)
        if a: ax.text(a + .15, yy - .19, f"{a:.1f}s", va="center", fontsize=8.5, color=BLUE)
    ax.axvline(6.1, color=GREEN, ls="--", lw=1.4, zorder=2)
    ax.text(6.3, len(rs) - .6, "cuVS IVF-PQ 384B\nbuilds in 6.1 s", fontsize=8.5, color=GREEN)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("seconds — Vogue-768, 100k training vectors, 208-core host")
    ax.set_title("Index build: the residual codebook was 80% of it, on one core")
    ax.legend(fontsize=9); ax.grid(alpha=.25, axis="x", zorder=0)
    fig.tight_layout(); fig.savefig(os.path.join(D, "fig4_build.png"), dpi=160)
    print("fig4_build.png")


# ── 5. alpha, i.e. how many candidates get refined ───────────────────────────
def fig_alpha():
    rs = rows("abl_alpha.csv")
    if not rs:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))
    for v, colour, mk in (("v16", GREY, "s"), ("v21", BLUE, "o")):
        pts = sorted((num(r["ck"]), num(r["recall"]), num(r["qps"]))
                     for r in rs if r["variant"] == v)
        if not pts:
            continue
        a1.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", color=colour, label=v, ms=6)
        a2.plot([p[0] for p in pts], [p[2] for p in pts], mk + "-", color=colour, label=v, ms=6)
    a1.set_xscale("log"); a2.set_xscale("log")
    a1.set_xlabel("ck = α·k, candidates sent to residual refinement")
    a2.set_xlabel("ck = α·k")
    a1.set_ylabel("Recall@10"); a2.set_ylabel("QPS")
    a1.axvline(320, color=RED, ls=":", lw=1.2); a2.axvline(320, color=RED, ls=":", lw=1.2)
    a1.text(340, a1.get_ylim()[0] + .002, "α=32", fontsize=8.5, color=RED)
    a1.set_title("Recall saturates by α=32"); a2.set_title("Cost keeps rising past it")
    for a in (a1, a2):
        a.grid(alpha=.25); a.legend(fontsize=9)
    fig.suptitle("What the refinement budget buys — Vogue-768, nprobe=128", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(D, "fig5_alpha.png"), dpi=160)
    print("fig5_alpha.png")


# ── 6. bytes: primary code against residual precision ────────────────────────
def fig_bytes():
    rs = rows("abl_bytes.csv")
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for Br, colour, mk in ((4, GREY, "s"), (8, BLUE, "o")):
        pts = sorted((num(r["bytes_per_vec"]), num(r["recall"]), r["M"])
                     for r in rs if int(r["Br"]) == Br)
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", color=colour,
                ms=8, lw=2, label=f"residual Br={Br} bits/dim")
        for b, rec, M in pts:
            ax.annotate(f"M={M}", (b, rec), textcoords="offset points",
                        xytext=(6, -11), fontsize=8, color=colour)
    ax.axhline(0.98, color=RED, ls="--", lw=1.2)
    ax.text(ax.get_xlim()[0], .9815, " 0.98", fontsize=8.5, color=RED)
    ax.set_xlabel("index payload, bytes per vector  (M + d·Br/8 + 4, d=768)")
    ax.set_ylabel("Recall@10 at nprobe=256")
    ax.set_title("Residual precision buys more recall per byte than a longer primary code")
    ax.grid(alpha=.25); ax.legend(fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(D, "fig6_bytes.png"), dpi=160)
    print("fig6_bytes.png")


# ── 7. the shared-memory wall ────────────────────────────────────────────────
def fig_klocal():
    rs = rows("abl_klocal.csv")
    if not rs:
        return
    blocks = sorted({int(r["block"]) for r in rs})
    kls = sorted({int(r["k_local"]) for r in rs})
    grid = np.full((len(kls), len(blocks)), np.nan)
    for r in rs:
        i, j = kls.index(int(r["k_local"])), blocks.index(int(r["block"]))
        v = num(r["recall"])
        if v: grid[i, j] = v
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    im = ax.imshow(grid, cmap="Blues", vmin=0.945, vmax=0.951, aspect="auto")
    for i in range(len(kls)):
        for j in range(len(blocks)):
            if np.isnan(grid[i, j]):
                # (2*K_LOCAL*BLOCK + 2*BLOCK) floats of scratch beside a 48 KB table
                kb = (2 * kls[i] * blocks[j] + 2 * blocks[j]) * 4 / 1024 + 48
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, hatch="///",
                                           facecolor="#F2F2F2", edgecolor="#BBB"))
                ax.text(j, i, f"{kb:.0f} KB\nwon't launch", ha="center", va="center",
                        fontsize=8, color=RED)
            else:
                ax.text(j, i, f"{grid[i, j]:.4f}", ha="center", va="center", fontsize=9.5,
                        color="white" if grid[i, j] > 0.9495 else "#123")
    ax.set_xticks(range(len(blocks))); ax.set_xticklabels(blocks)
    ax.set_yticks(range(len(kls))); ax.set_yticklabels(kls)
    ax.set_xlabel("BLOCK (threads per block)"); ax.set_ylabel("K_LOCAL (candidates kept per thread)")
    ax.set_title("Occupancy and candidate retention cannot both be large\n"
                 "Recall@10 at nprobe=256; the 48 KB table leaves 51 KB for scratch",
                 fontsize=11)
    fig.colorbar(im, ax=ax, label="Recall@10", shrink=.85)
    fig.tight_layout(); fig.savefig(os.path.join(D, "fig7_klocal.png"), dpi=160)
    print("fig7_klocal.png")


# ── 8. probing every list ────────────────────────────────────────────────────
def fig_nprobe():
    rs = rows("abl_nprobe_ceiling.csv")
    if not rs:
        return
    pts = sorted((num(r["nprobe"]), num(r["recall"]), num(r["qps"])) for r in rs)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=BLUE, ms=8, lw=2.2)
    for n, rec, _ in pts:
        ax.annotate(f"{rec:.4f}", (n, rec), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9)
    ax.axvline(1024, color=RED, ls="--", lw=1.2)
    ax.text(1024 * .93, pts[0][1] + .001, "every list probed", rotation=90,
            fontsize=8.5, color=RED, ha="right")
    ax.set_xscale("log", base=2); ax.set_xticks([p[0] for p in pts])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("nprobe  (nlist = 1024)"); ax.set_ylabel("Recall@10")
    ax.set_title("Probing every list reaches 0.9528 — the ceiling is the code, not the search")
    ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(os.path.join(D, "fig8_nprobe.png"), dpi=160)
    print("fig8_nprobe.png")


# ── 9. CPU against GPU at matched recall ─────────────────────────────────────
def fig_cpu():
    rs = rows("cpu_jhq.csv")
    if not rs:
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    styles = {("JHQ-CPU-IVF", "1"): (GREY, "s", "JHQ CPU, 1 thread"),
              ("JHQ-CPU-IVF", "all"): ("#5C6773", "D", "JHQ CPU, all 208 cores"),
              ("JQ-CPU-IVF", "1"): ("#C4B8E0", "v", "JQ CPU (residual off), 1 thread")}
    for (meth, th), (colour, mk, label) in styles.items():
        pts = sorted((num(r["recall"]), num(r["qps"])) for r in rs
                     if r["dataset"] == "vogue-768" and r["method"] == meth and r["threads"] == th)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", color=colour,
                    label=label, ms=6)
    gpu = sorted((num(r["recall"]), num(r["qps_mean"])) for r in rows("p0_vogue_jhq_M96Br4.csv")
                 if r["status"] == "ok")
    if gpu:
        ax.plot([p[0] for p in gpu], [p[1] for p in gpu], "o-", color=BLUE, lw=2.4, ms=7,
                label="JHQ GPU (this work)")
    ax.set_yscale("log"); ax.set_xlabel("Recall@10"); ax.set_ylabel("QPS")
    ax.set_title("GPU against the CPU reference at matched settings — Vogue-768, α=100")
    ax.grid(alpha=.25, which="both"); ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(D, "fig9_cpu_gpu.png"), dpi=160)
    print("fig9_cpu_gpu.png")


if __name__ == "__main__":
    fig_ablation(); fig_build(); fig_alpha(); fig_bytes()
    fig_klocal(); fig_nprobe(); fig_cpu()
