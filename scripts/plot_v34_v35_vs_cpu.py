#!/usr/bin/env python3
"""
HBlock v30 / v34 / v35 vs CPU baselines — Vogue-768 + arXiv-768.
Layout:
  [0,0] arXiv full scale          [0,1] Vogue full scale (v30/v34/v35 + CPU)
  [1,0] arXiv ≤70K zoom vs CPU   [1,1] Vogue recall≥0.90 zoom (v30/v34/v35 detail)
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS = "/Users/apple/github/JHQ_GPU/results"

# ── Load data ─────────────────────────────────────────────────────────────────
v17_arxiv = pd.read_csv(f"{RESULTS}/hblock_v17_arxiv.csv")
v17_vogue = pd.read_csv(f"{RESULTS}/hblock_v17_vogue.csv")
v30_arxiv = pd.read_csv(f"{RESULTS}/hblock_v30_arxiv.csv")
v30_vogue = pd.read_csv(f"{RESULTS}/hblock_v30_vogue.csv")
cpu_arxiv = pd.read_csv(f"{RESULTS}/cpu_arxiv.csv")
cpu_vogue = pd.read_csv(f"{RESULTS}/cpu_vogue.csv")

# v34: beam=min(ef,128), expanded-block output, HNSW early termination (Vogue-768)
# ef=64 QPS from v32 (kernel identical in this range)
v34_vogue = pd.DataFrame({
    "recall": [0.7000, 0.9270, 0.9707, 0.9907, 0.9907],
    "qps":    [503247, 311000, 208000, 130000, 127160],
    "ef":     [8,      32,     64,     128,    256],
})

# v35: beam=ef (no cap), expanded-block output, NO early termination (ceiling)
v35_vogue = pd.DataFrame({
    "recall": [0.9904, 0.9968],
    "qps":    [130000, 72033],
    "ef":     [128,    256],
})

# ── Pareto ────────────────────────────────────────────────────────────────────
def pareto(df, rc="recall@10", qc="qps"):
    df = df.sort_values(rc, ascending=False).reset_index(drop=True)
    pts, best = [], -1
    for _, r in df.iterrows():
        if r[qc] > best:
            best = r[qc]; pts.append(r)
    return pd.DataFrame(pts).sort_values(rc)

def pareto_s(df, rc="recall", qc="qps"):
    df = df.sort_values(rc, ascending=False).reset_index(drop=True)
    pts, best = [], -1
    for _, r in df.iterrows():
        if r[qc] > best:
            best = r[qc]; pts.append(r)
    return pd.DataFrame(pts).sort_values(rc)

v30_ax_p  = pareto(v30_arxiv)
v30_vg_p  = pareto(v30_vogue)
v34_vg_p  = pareto_s(v34_vogue)
v35_vg_p  = pareto_s(v35_vogue)

# ── Colors ────────────────────────────────────────────────────────────────────
C = {
    "v17":  "#e6194b",
    "v30":  "#ff7f0e",
    "v34":  "#1f77b4",
    "v35":  "#2ca02c",
    "JHQ+IVF":          "#3cb44b",
    "JQ+IVF":           "#4363d8",
    "FAISS-IVFPQ-384B": "#f58231",
    "FAISS-IVFPQ-192B": "#911eb4",
    "FAISS-IVFPQ-96B":  "#808080",
}

def plot_cpu(ax, cpu_df, kw={}):
    for method, grp in cpu_df.groupby("method"):
        grp = grp.sort_values("recall")
        base = dict(linewidth=1.6, markersize=5)
        base.update(kw)
        if method == "JHQ+IVF":
            ax.plot(grp["recall"], grp["qps"]/1000, "s--", color=C["JHQ+IVF"],
                    label="JHQ+IVF (CPU)", **base)
        elif method == "JQ+IVF":
            ax.plot(grp["recall"], grp["qps"]/1000, "^--", color=C["JQ+IVF"],
                    label="JQ+IVF (CPU)", **base)
        elif method == "FAISS-IVFPQ-384B":
            ax.plot(grp["recall"], grp["qps"]/1000, "D:",
                    color=C["FAISS-IVFPQ-384B"], label="FAISS-IVFPQ-384B (CPU)", **base)
        elif method == "FAISS-IVFPQ-192B":
            ax.plot(grp["recall"], grp["qps"]/1000, "v:",
                    color=C["FAISS-IVFPQ-192B"], label="FAISS-IVFPQ-192B (CPU)", **base)
        elif method == "FAISS-IVFPQ-96B":
            ax.plot(grp["recall"], grp["qps"]/1000, "x:",
                    color=C["FAISS-IVFPQ-96B"],  label="FAISS-IVFPQ-96B (CPU)", **base)

def decorate(ax, title, xlim=(0.3, 1.01), ylim=None, legend_loc="upper right"):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall@10", fontsize=11)
    ax.set_ylabel("QPS (thousands)", fontsize=11)
    ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(0, ylim)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}K"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc=legend_loc)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
GKW = dict(linewidth=2.5, markersize=7, zorder=5)

# ── [0,0] arXiv full scale ────────────────────────────────────────────────────
ax = axes[0][0]
ax.plot(v17_arxiv["recall"],    v17_arxiv["qps"]/1000,  "o-", color=C["v17"],
        label="HBlock-v17 (GPU)", **GKW)
ax.plot(v30_ax_p["recall@10"], v30_ax_p["qps"]/1000,  "s-", color=C["v30"],
        label="HBlock-v30 (GPU)", **GKW)
plot_cpu(ax, cpu_arxiv)
decorate(ax, "arXiv-Abstracts-768 (full scale)")

# ── [0,1] Vogue full scale ────────────────────────────────────────────────────
ax = axes[0][1]
ax.plot(v17_vogue["recall"],   v17_vogue["qps"]/1000,  "o-", color=C["v17"],
        label="HBlock-v17 (GPU)", **GKW)
ax.plot(v30_vg_p["recall@10"], v30_vg_p["qps"]/1000,  "s-", color=C["v30"],
        label="HBlock-v30 (GPU)", **GKW)
ax.plot(v34_vg_p["recall"],    v34_vg_p["qps"]/1000,  "D-", color=C["v34"],
        label="HBlock-v34 (beam=min(ef,128))", **GKW)
ax.plot(v35_vg_p["recall"],    v35_vg_p["qps"]/1000,  "^-", color=C["v35"],
        label="HBlock-v35 (beam=ef, no early-stop)", **GKW)
plot_cpu(ax, cpu_vogue)
decorate(ax, "Vogue-768 (full scale)", legend_loc="upper left")

# ── [1,0] arXiv ≤70K zoom vs CPU ─────────────────────────────────────────────
ax = axes[1][0]
sub17 = v17_arxiv[v17_arxiv["qps"] <= 70000].sort_values("recall")
ax.plot(sub17["recall"], sub17["qps"]/1000, "o-", color=C["v17"],
        label="HBlock-v17 (GPU)", **GKW)
ax.text(0.32, 66, "↑ GPU: more points above this range", fontsize=8, color=C["v17"])
sub30 = v30_ax_p[v30_ax_p["qps"] <= 70000]
ax.plot(sub30["recall@10"], sub30["qps"]/1000, "s-", color=C["v30"],
        label="HBlock-v30 (GPU)", **GKW)
plot_cpu(ax, cpu_arxiv)
decorate(ax, "arXiv-Abstracts-768 (≤70K QPS zoom)", ylim=70)

# ── [1,1] Vogue recall ≥ 0.90 zoom: v30 / v34 / v35 detail ──────────────────
ax = axes[1][1]

# v30 pareto points in range
sub30v = v30_vg_p[v30_vg_p["recall@10"] >= 0.90]
ax.plot(sub30v["recall@10"], sub30v["qps"]/1000, "s-", color=C["v30"],
        label="HBlock-v30 (GPU)", **GKW)

# v34: ef=128 gives 0.9907/130K; ef=256 same recall at 127K → not on pareto
sub34 = v34_vg_p[v34_vg_p["recall"] >= 0.90]
ax.plot(sub34["recall"], sub34["qps"]/1000, "D-", color=C["v34"],
        label="HBlock-v34 (beam=min(ef,128))", **GKW)

# v35: both points are ≥ 0.90
ax.plot(v35_vg_p["recall"], v35_vg_p["qps"]/1000, "^-", color=C["v35"],
        label="HBlock-v35 (beam=ef, ceiling)", **GKW)

# annotate key endpoints — stagger vertically to avoid overlap
annots = [
    (0.9907, 130, "v34 ef=128 · 130K QPS", C["v34"], (-8, 50),  "right"),
    (0.9914, 107, "v30 ef=256 · 107K QPS", C["v30"], (-8, -50), "right"),
    (0.9968, 72,  "v35 ef=256 · 72K QPS",  C["v35"], (-8, -50), "right"),
]
for rx, qy, lbl, col, off, ha in annots:
    ax.annotate(lbl, xy=(rx, qy), xytext=off,
                textcoords="offset points",
                fontsize=8.5, color=col, ha=ha,
                arrowprops=dict(arrowstyle="->", color=col, lw=1.1))

# no CPU baselines here — they don't reach recall 0.90 and would clutter the panel
decorate(ax, "Vogue-768 (Recall@10 ≥ 0.90 zoom)",
         xlim=(0.90, 1.005), ylim=200, legend_loc="upper right")

# ── Suptitle & save ───────────────────────────────────────────────────────────
fig.suptitle(
    "HBlock GPU vs CPU baselines — Vogue-768 & arXiv-768 (Linear Scale)\n"
    "v34: beam=min(ef,128), early-stop  |  v35: beam=ef, no early-stop (recall ceiling)",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
out = f"{RESULTS}/hblock_v34_v35_vs_cpu_linear.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
