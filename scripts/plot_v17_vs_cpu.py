#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS = "/Users/apple/github/JHQ_GPU/results"

v17_arxiv = pd.read_csv(f"{RESULTS}/hblock_v17_arxiv.csv")
v17_vogue = pd.read_csv(f"{RESULTS}/hblock_v17_vogue.csv")
cpu_arxiv = pd.read_csv(f"{RESULTS}/cpu_arxiv.csv")
cpu_vogue = pd.read_csv(f"{RESULTS}/cpu_vogue.csv")

COLORS = {
    "HBlock-v17 GPU": "#e6194b",
    "JHQ+IVF":        "#3cb44b",
    "JQ+IVF":         "#4363d8",
    "FAISS-IVFPQ-384B": "#f58231",
    "FAISS-IVFPQ-192B": "#911eb4",
    "FAISS-IVFPQ-96B":  "#808080",
}

def plot_dataset(ax, v17_df, cpu_df, title):
    # GPU v17
    ax.plot(v17_df["recall"], v17_df["qps"] / 1000,
            "o-", color=COLORS["HBlock-v17 GPU"],
            linewidth=2.5, markersize=7, label="HBlock-v17 (GPU, RTX 5090)", zorder=5)

    for method, grp in cpu_df.groupby("method"):
        grp = grp.sort_values("recall")
        style = dict(linewidth=1.8, markersize=6)
        if method == "JHQ+IVF":
            ax.plot(grp["recall"], grp["qps"] / 1000,
                    "s--", color=COLORS["JHQ+IVF"], label="JHQ+IVF (CPU)", **style)
        elif method == "JQ+IVF":
            ax.plot(grp["recall"], grp["qps"] / 1000,
                    "^--", color=COLORS["JQ+IVF"], label="JQ+IVF (CPU)", **style)
        elif method == "FAISS-IVFPQ-384B":
            ax.plot(grp["recall"], grp["qps"] / 1000,
                    "D:", color=COLORS["FAISS-IVFPQ-384B"], label="FAISS-IVFPQ-384B (CPU)", **style)
        elif method == "FAISS-IVFPQ-192B":
            ax.plot(grp["recall"], grp["qps"] / 1000,
                    "v:", color=COLORS["FAISS-IVFPQ-192B"], label="FAISS-IVFPQ-192B (CPU)", **style)
        elif method == "FAISS-IVFPQ-96B":
            ax.plot(grp["recall"], grp["qps"] / 1000,
                    "x:", color=COLORS["FAISS-IVFPQ-96B"], label="FAISS-IVFPQ-96B (CPU)", **style)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall@10", fontsize=11)
    ax.set_ylabel("QPS (thousands)", fontsize=11)
    ax.set_xlim(0.3, 1.01)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}K"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Row 1: full scale (GPU visible)
plot_dataset(axes[0][0], v17_arxiv, cpu_arxiv, "arXiv-Abstracts-768 (full scale)")
plot_dataset(axes[0][1], v17_vogue, cpu_vogue, "Vogue-768 (full scale)")

# Row 2: CPU scale only (zoom in, no GPU line)
def plot_cpu_only(ax, cpu_df, v17_df, title):
    # v17 GPU — only points within ylim, arrow indicates curve continues upward
    sub = v17_df[v17_df["qps"] <= 70000].sort_values("ck")
    ax.plot(sub["recall"], sub["qps"] / 1000,
            "o-", color=COLORS["HBlock-v17 GPU"],
            linewidth=2.5, markersize=7, label="HBlock-v17 (GPU)", zorder=5)
    ax.text(0.32, 66, "↑ GPU: more points above this range",
            fontsize=8, color=COLORS["HBlock-v17 GPU"])

    for method, grp in cpu_df.groupby("method"):
        grp = grp.sort_values("recall")
        style = dict(linewidth=1.8, markersize=6)
        if method == "JHQ+IVF":
            ax.plot(grp["recall"], grp["qps"] / 1000, "s--", color=COLORS["JHQ+IVF"],   label="JHQ+IVF", **style)
        elif method == "JQ+IVF":
            ax.plot(grp["recall"], grp["qps"] / 1000, "^--", color=COLORS["JQ+IVF"],    label="JQ+IVF", **style)
        elif method == "FAISS-IVFPQ-384B":
            ax.plot(grp["recall"], grp["qps"] / 1000, "D:",  color=COLORS["FAISS-IVFPQ-384B"], label="FAISS-IVFPQ-384B", **style)
        elif method == "FAISS-IVFPQ-192B":
            ax.plot(grp["recall"], grp["qps"] / 1000, "v:",  color=COLORS["FAISS-IVFPQ-192B"], label="FAISS-IVFPQ-192B", **style)
        elif method == "FAISS-IVFPQ-96B":
            ax.plot(grp["recall"], grp["qps"] / 1000, "x:",  color=COLORS["FAISS-IVFPQ-96B"],  label="FAISS-IVFPQ-96B", **style)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall@10", fontsize=11)
    ax.set_ylabel("QPS (thousands)", fontsize=11)
    ax.set_xlim(0.3, 1.01)
    ax.set_ylim(0, 70)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}K"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

plot_cpu_only(axes[1][0], cpu_arxiv, v17_arxiv, "arXiv-Abstracts-768 (≤70K QPS zoom)")
plot_cpu_only(axes[1][1], cpu_vogue, v17_vogue, "Vogue-768 (≤70K QPS zoom)")

fig.suptitle("HBlock-v17 (GPU) vs CPU baselines — Linear Scale", fontsize=14, fontweight="bold")
plt.tight_layout()
out = f"{RESULTS}/v17_vs_cpu_linear.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
