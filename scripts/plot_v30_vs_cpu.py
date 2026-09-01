#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS = "/Users/apple/github/JHQ_GPU/results"

v17_arxiv = pd.read_csv(f"{RESULTS}/hblock_v17_arxiv.csv")
v17_vogue = pd.read_csv(f"{RESULTS}/hblock_v17_vogue.csv")
v30_arxiv = pd.read_csv(f"{RESULTS}/hblock_v30_arxiv.csv")
v30_vogue = pd.read_csv(f"{RESULTS}/hblock_v30_vogue.csv")
cpu_arxiv = pd.read_csv(f"{RESULTS}/cpu_arxiv.csv")
cpu_vogue = pd.read_csv(f"{RESULTS}/cpu_vogue.csv")

def pareto(df, recall_col="recall@10", qps_col="qps"):
    # Sort descending by recall; keep points with QPS strictly higher than any
    # higher-recall point (i.e., non-dominated upper-left frontier).
    df = df.sort_values(recall_col, ascending=False).reset_index(drop=True)
    pts = []; best_qps = -1
    for _, row in df.iterrows():
        if row[qps_col] > best_qps:
            best_qps = row[qps_col]; pts.append(row)
    return pd.DataFrame(pts).sort_values(recall_col)

v30_arxiv_p = pareto(v30_arxiv)
v30_vogue_p = pareto(v30_vogue)

COLORS = {
    "v17":              "#e6194b",
    "v30":              "#ff7f0e",
    "JHQ+IVF":          "#3cb44b",
    "JQ+IVF":           "#4363d8",
    "FAISS-IVFPQ-384B": "#f58231",
    "FAISS-IVFPQ-192B": "#911eb4",
    "FAISS-IVFPQ-96B":  "#808080",
}

def plot_cpu_methods(ax, cpu_df, style_kw):
    for method, grp in cpu_df.groupby("method"):
        grp = grp.sort_values("recall")
        kw = dict(linewidth=1.8, markersize=6, **style_kw)
        if method == "JHQ+IVF":
            ax.plot(grp["recall"], grp["qps"]/1000, "s--", color=COLORS["JHQ+IVF"],   label="JHQ+IVF (CPU)", **kw)
        elif method == "JQ+IVF":
            ax.plot(grp["recall"], grp["qps"]/1000, "^--", color=COLORS["JQ+IVF"],    label="JQ+IVF (CPU)", **kw)
        elif method == "FAISS-IVFPQ-384B":
            ax.plot(grp["recall"], grp["qps"]/1000, "D:",  color=COLORS["FAISS-IVFPQ-384B"], label="FAISS-IVFPQ-384B (CPU)", **kw)
        elif method == "FAISS-IVFPQ-192B":
            ax.plot(grp["recall"], grp["qps"]/1000, "v:",  color=COLORS["FAISS-IVFPQ-192B"], label="FAISS-IVFPQ-192B (CPU)", **kw)
        elif method == "FAISS-IVFPQ-96B":
            ax.plot(grp["recall"], grp["qps"]/1000, "x:",  color=COLORS["FAISS-IVFPQ-96B"],  label="FAISS-IVFPQ-96B (CPU)", **kw)

def decorate(ax, title, ylim=None):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall@10", fontsize=11)
    ax.set_ylabel("QPS (thousands)", fontsize=11)
    ax.set_xlim(0.3, 1.01)
    if ylim: ax.set_ylim(0, ylim)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}K"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# ── Row 1: full scale ────────────────────────────────────────────────────────
for ax, v17, v30p, cpu, title in [
    (axes[0][0], v17_arxiv, v30_arxiv_p, cpu_arxiv, "arXiv-Abstracts-768 (full scale)"),
    (axes[0][1], v17_vogue, v30_vogue_p, cpu_vogue, "Vogue-768 (full scale)"),
]:
    ax.plot(v17["recall"], v17["qps"]/1000, "o-",
            color=COLORS["v17"], linewidth=2.5, markersize=7,
            label="HBlock-v17 (GPU, RTX 5090)", zorder=5)
    ax.plot(v30p["recall@10"], v30p["qps"]/1000, "s-",
            color=COLORS["v30"], linewidth=2.5, markersize=7,
            label="HBlock-v30 (GPU, RTX 5090)", zorder=5)
    plot_cpu_methods(ax, cpu, {})
    decorate(ax, title)

# ── Row 2: ≤70K zoom ─────────────────────────────────────────────────────────
for ax, v17, v30p, cpu, title in [
    (axes[1][0], v17_arxiv, v30_arxiv_p, cpu_arxiv, "arXiv-Abstracts-768 (≤70K QPS zoom)"),
    (axes[1][1], v17_vogue, v30_vogue_p, cpu_vogue, "Vogue-768 (≤70K QPS zoom)"),
]:
    sub17 = v17[v17["qps"] <= 70000].sort_values("ck")
    ax.plot(sub17["recall"], sub17["qps"]/1000, "o-",
            color=COLORS["v17"], linewidth=2.5, markersize=7,
            label="HBlock-v17 (GPU)", zorder=5)
    ax.text(0.32, 66, "↑ GPU: more points above this range",
            fontsize=8, color=COLORS["v17"])

    sub30 = v30p[v30p["qps"] <= 70000]
    ax.plot(sub30["recall@10"], sub30["qps"]/1000, "s-",
            color=COLORS["v30"], linewidth=2.5, markersize=7,
            label="HBlock-v30 (GPU)", zorder=5)

    plot_cpu_methods(ax, cpu, {})
    decorate(ax, title, ylim=70)

fig.suptitle("HBlock-v17 vs v30 (GPU) vs CPU baselines — Linear Scale",
             fontsize=14, fontweight="bold")
plt.tight_layout()
out = f"{RESULTS}/hblock_v30_vs_cpu_linear.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
