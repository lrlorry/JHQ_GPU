#!/usr/bin/env python3
"""
Generate one JHQ-GPU vs JHQ-CPU-IVF recall/QPS figure per dataset,
matching the style of the existing results/jhq_v12_vs_cpu.png (which only
covers Vogue-768), for every dataset that has both a GPU and a CPU CSV
under results/. GPU source is jhq_v12_* where present, else jhq_v14_*
(bge-m3/stella-trec24) -- see discover_datasets().

Silently skips a dataset if either its GPU or CPU CSV is missing -- rerun
this after adding more datasets' results, no need to list them here.

Usage: python3 scripts/plot_v12_vs_cpu_all.py
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

GPU_COLOR = "#e6194b"
# demo_jhq_ivf now reports two CPU methods per dataset: full JHQ (primary +
# residual refine) and a JQ-only ablation (primary code alone, no residual
# stage) -- give them distinct colors so the two dashed lines aren't
# indistinguishable at a glance.
CPU_METHOD_COLORS = {
    "JHQ-CPU-IVF": "#3cb44b",  # green
    "JQ-CPU-IVF": "#4363d8",   # blue
}
CPU_COLOR_FALLBACK = "#808000"

# (display title, gpu csv, cpu csv) -- discovered dynamically below instead
# of hardcoded, so this script doesn't need editing as more datasets land.
# v12_transposed is the original/validated GPU line for the first 4
# datasets; v14_streaming_add exists only because v12's add() can't fit
# bge-m3/stella-trec24 in VRAM (see jhq_v14_streaming_add/jhq_gpu_index.cu),
# so it's the GPU source for those two specifically. Where both exist for
# the same dataset (arxiv/vogue got a v14 run too, just to cross-validate
# it against v12 -- see the recall-match check before trusting v14 on the
# big datasets), v12 wins: it's the one with the longer track record.
GPU_PREFIXES = ["jhq_v12_", "jhq_v14_"]
CPU_PREFIX = "jhq_cpu_ivf_"


def discover_datasets():
    found = {}
    for prefix in GPU_PREFIXES:
        for path in glob.glob(os.path.join(RESULTS, f"{prefix}*.csv")):
            name = os.path.basename(path)[len(prefix):-len(".csv")]
            entry = found.setdefault(name, {})
            entry.setdefault("gpu", path)  # first prefix wins -- v12 before v14
    for path in glob.glob(os.path.join(RESULTS, f"{CPU_PREFIX}*.csv")):
        name = os.path.basename(path)[len(CPU_PREFIX):-len(".csv")]
        found.setdefault(name, {})["cpu"] = path
    # vogue768's GPU csv predates this naming convention -- special-case it.
    vogue_gpu = os.path.join(RESULTS, "jhq_v12_vogue.csv")
    if os.path.exists(vogue_gpu):
        found.setdefault("vogue-768", {})["gpu"] = vogue_gpu
    return {name: paths for name, paths in found.items() if "gpu" in paths and "cpu" in paths}


def plot_one(ax, gpu_csv, cpu_csv, title):
    gpu_df = pd.read_csv(gpu_csv).sort_values("recall")
    gpu_method = gpu_df["method"].iloc[0]
    ax.plot(gpu_df["recall"], gpu_df["qps"] / 1000, "o-",
            color=GPU_COLOR, linewidth=2.5, markersize=7,
            label=f"{gpu_method} (RTX 5090)", zorder=5)

    cpu_df = pd.read_csv(cpu_csv).sort_values("recall")
    for method, grp in cpu_df.groupby("method"):
        color = CPU_METHOD_COLORS.get(method, CPU_COLOR_FALLBACK)
        ax.plot(grp["recall"], grp["qps"] / 1000, "s--",
                color=color, linewidth=1.8, markersize=6,
                label=f"{method} (CPU)")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall@10", fontsize=11)
    ax.set_ylabel("QPS (thousands)", fontsize=11)
    # Size the x-axis to the actual data range instead of a fixed 0.3 floor --
    # the JQ-only ablation line's recall never even gets close to 0.3, so a
    # hardcoded floor just wastes the left third of every panel as blank space.
    xmin = min(gpu_df["recall"].min(), cpu_df["recall"].min())
    ax.set_xlim(xmin - 0.02, 1.01)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}K"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")


def pretty_title(name):
    special = {
        "vogue-768": "Vogue-768",
        "arxiv-abstracts-768": "Arxiv-Abstracts-768",
        "openai3-1536": "OpenAI3-1536",
        "openai3-3072": "OpenAI3-3072",
        "bge-m3": "BGE-M3-1024",
        "stella-trec24": "Stella-TREC24",
    }
    return special.get(name, name)


def main():
    datasets = discover_datasets()
    if not datasets:
        print("No dataset has both a GPU and a CPU CSV yet -- nothing to plot.")
        return

    for name, paths in sorted(datasets.items()):
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_one(ax, paths["gpu"], paths["cpu"], pretty_title(name))
        plt.tight_layout()
        out = os.path.join(RESULTS, f"jhq_v12_vs_cpu_{name}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved -> {out}")

    # Combined grid, one panel per dataset, for a single at-a-glance figure.
    names = sorted(datasets.keys())
    ncols = 2
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes = axes.flatten() if len(names) > 1 else [axes]
    for i, name in enumerate(names):
        plot_one(axes[i], datasets[name]["gpu"], datasets[name]["cpu"], pretty_title(name))
    for j in range(len(names), len(axes)):
        axes[j].axis("off")
    fig.suptitle("JHQ-GPU vs JHQ-CPU-IVF", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(RESULTS, "jhq_v12_vs_cpu_all.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
