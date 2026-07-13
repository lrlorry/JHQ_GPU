#!/usr/bin/env python3
"""
Plot recall-QPS comparison: HBlock-v17 (GPU) vs CPU baselines.

Usage:
  python3 scripts/plot_v17_comparison.py \
      results/hblock_v17_vogue.csv \
      results/cpu_vogue.csv \
      [results/hblock_v17_arxiv.csv]   # optional second GPU dataset
  Output: results/hblock_v17_comparison.pdf / .png
"""
import sys, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family":        "DejaVu Serif",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     11.5,
    "legend.fontsize":    8.5,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "axes.linewidth":     0.9,
    "lines.linewidth":    2.0,
    "lines.markersize":   7,
    "grid.linewidth":     0.45,
    "grid.alpha":         0.45,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.06,
})

STYLES = {
    "HBlock-v17 (Vogue-768)":  dict(color="#E31A1C", marker="o",  ls="-",  lw=2.4, zorder=9),
    "HBlock-v17 (Arxiv-768)":  dict(color="#FF7F00", marker="D",  ls="-",  lw=2.4, zorder=8),
    "JHQ+IVF":                 dict(color="#1F77B4", marker="s",  ls="--", lw=2.0, zorder=5),
    "JQ+IVF":                  dict(color="#2CA02C", marker="^",  ls="--", lw=2.0, zorder=5),
    "FAISS-IVFPQ-96B":         dict(color="#9467BD", marker="v",  ls=":",  lw=1.6, zorder=3),
    "FAISS-IVFPQ-192B":        dict(color="#8C564B", marker="<",  ls=":",  lw=1.6, zorder=3),
    "FAISS-IVFPQ-384B":        dict(color="#7F7F7F", marker=">",  ls=":",  lw=1.6, zorder=3),
}


def load_csv(path):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            m = row["method"]
            if m not in data:
                data[m] = {"recall": [], "qps": [], "build_time": float(row.get("build_time", 0))}
            data[m]["recall"].append(float(row["recall"]))
            data[m]["qps"].append(float(row["qps"]))
    return data


def plot_panel(ax, data, title, ds_tag=""):
    # GPU first, then CPU
    order = [k for k in STYLES if k in data and k.startswith("HBlock")] + \
            [k for k in STYLES if k in data and not k.startswith("HBlock")]
    for mname in order:
        if mname not in data:
            continue
        d     = data[mname]
        style = STYLES.get(mname, dict(color="black", marker="o", ls="-", lw=1.5, zorder=2))
        if mname.startswith("HBlock"):
            label = f"{mname.replace('HBlock-v17 ', 'HBlock-v17 GPU ')}"
        elif mname.startswith("FAISS"):
            label = f"{mname} CPU"
        else:
            label = f"{mname} CPU ({ds_tag})" if ds_tag else f"{mname} CPU"
        pairs = sorted(zip(d["recall"], d["qps"]))
        rs    = [p[0] * 100 for p in pairs]
        qs    = [p[1]        for p in pairs]
        ax.semilogy(rs, qs,
                    marker=style["marker"],
                    color=style["color"],
                    linestyle=style["ls"],
                    linewidth=style["lw"],
                    markerfacecolor=style["color"],
                    markeredgecolor=style["color"],
                    markeredgewidth=1.2,
                    zorder=style["zorder"],
                    label=label)

    ax.set_xlabel("Recall@10 (%)")
    ax.set_ylabel("QPS")
    ax.set_title(title, fontweight="bold", pad=6)
    ax.set_xlim(0, 102)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else
                      f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))
    ax.grid(True, which="major", ls="-",  alpha=0.35)
    ax.grid(True, which="minor", ls=":",  alpha=0.20)
    for xv in [70, 80, 90, 95, 99]:
        ax.axvline(xv, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.92, edgecolor="0.75",
              handlelength=2.2, labelspacing=0.35, ncol=1)


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: plot_v17_comparison.py <gpu_vogue.csv> <cpu_vogue.csv> "
              "[gpu_arxiv.csv] [cpu_arxiv.csv] [output_prefix]")
        sys.exit(1)

    csv_args = [a for a in args if a.endswith(".csv")]
    prefix   = next((a for a in args if not a.endswith(".csv")),
                    "results/hblock_v17_comparison")

    gpu_vogue_csv = csv_args[0]
    cpu_vogue_csv = csv_args[1]
    gpu_arxiv_csv = csv_args[2] if len(csv_args) > 2 else None
    cpu_arxiv_csv = csv_args[3] if len(csv_args) > 3 else None

    data_vogue = load_csv(cpu_vogue_csv)
    data_vogue.update(load_csv(gpu_vogue_csv))

    has_arxiv = gpu_arxiv_csv and os.path.exists(gpu_arxiv_csv)

    if has_arxiv:
        data_arxiv = {}
        if cpu_arxiv_csv and os.path.exists(cpu_arxiv_csv):
            data_arxiv.update(load_csv(cpu_arxiv_csv))
        data_arxiv.update(load_csv(gpu_arxiv_csv))
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5))
        plot_panel(ax0, data_vogue,  "Vogue-768  (N=933K, d=768, k=10)", "Vogue-768")
        plot_panel(ax1, data_arxiv,  "Arxiv-768  (N=2.3M, d=768, k=10)", "Arxiv-768")
        fig.suptitle("HBlock-v17 GPU vs CPU Baselines", fontsize=12,
                     fontweight="bold", y=1.02)
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(8, 5))
        plot_panel(ax0, data_vogue, "Vogue-768  (N=933K, d=768, k=10)  —  GPU vs CPU", "Vogue-768")
        fig.suptitle("HBlock-v17 GPU vs CPU Baselines", fontsize=12,
                     fontweight="bold", y=1.02)

    os.makedirs(os.path.dirname(os.path.abspath(prefix)) or ".", exist_ok=True)
    fig.savefig(prefix + ".pdf")
    fig.savefig(prefix + ".png", dpi=300)
    print(f"Saved: {prefix}.pdf")
    print(f"Saved: {prefix}.png")


if __name__ == "__main__":
    main()
