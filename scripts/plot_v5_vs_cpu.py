"""
v5 CUDA Graph vs CPU 480B/vec baselines (Official-JHQ + JHQ Repro).
Usage:
    python3 scripts/plot_v5_vs_cpu.py \
        ../JHQ_repro/results/vogue768_results.csv \
        results/jhq_v5_vogue.csv \
        results/jhq_v5_vs_cpu_vogue
"""
import sys, csv
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
    "legend.fontsize":    9.0,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "axes.linewidth":     0.9,
    "lines.linewidth":    2.2,
    "lines.markersize":   8,
    "grid.linewidth":     0.45,
    "grid.alpha":         0.45,
    "figure.dpi":         180,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.06,
})

METHODS = {
    "JHQ-GPU-v5-CUDAGraph": dict(color="#BCBD22", marker="D", ls="-",  lw=2.4, zorder=10,
                                  label="JHQ-GPU v5 CUDA Graph (480 B/vec)",
                                  mfc="#BCBD22", mew=1.6, ms=8),
    "Official-JHQ":          dict(color="#D62728", marker="o", ls="--", lw=2.2, zorder=6,
                                  label="JHQ Official CPU (480 B/vec)",
                                  mfc="white", mew=2.0, ms=8),
    "JHQ+IVF":               dict(color="#D62728", marker="o", ls="-",  lw=2.2, zorder=5,
                                  label="JHQ Repro CPU (480 B/vec)",
                                  mfc="#D62728", mew=1.8, ms=8),
}

def load_csv(path):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            m = row["method"]
            if m not in METHODS:
                continue
            if m not in data:
                data[m] = {"recall": [], "qps": [], "nprobe": [],
                            "build_time": float(row.get("build_time", 0))}
            data[m]["recall"].append(float(row["recall"]))
            data[m]["qps"].append(float(row["qps"]))
            data[m]["nprobe"].append(float(row["nprobe"]))
    return data

def _qps_fmt(x, _):
    return f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"

def plot_speed_accuracy(ax, data):
    for mname, style in METHODS.items():
        if mname not in data:
            continue
        d = data[mname]
        recalls = [r * 100 for r in d["recall"]]
        ax.semilogy(recalls, d["qps"],
                    marker=style["marker"],
                    color=style["color"],
                    linestyle=style["ls"],
                    linewidth=style["lw"],
                    markerfacecolor=style["mfc"],
                    markeredgewidth=style["mew"],
                    markeredgecolor=style["color"],
                    markersize=style["ms"],
                    zorder=style["zorder"],
                    label=style["label"])

    ax.set_xlabel("Recall@10 (%)")
    ax.set_ylabel("Queries per Second (QPS)")
    ax.set_title("(a) Speed–Accuracy Trade-off", fontweight="bold", pad=6)
    ax.set_xlim(85, 101)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_qps_fmt))
    ax.grid(True, which="major", ls="-",  alpha=0.35)
    ax.grid(True, which="minor", ls=":",  alpha=0.20)
    for xv, lbl in [(90, "90%"), (95, "95%"), (99, "99%")]:
        ax.axvline(xv, color="gray", lw=0.7, ls=":", alpha=0.6)
        ax.text(xv + 0.15, ax.get_ylim()[0] * 1.5, lbl, fontsize=7.5,
                color="gray", rotation=90, va="bottom")
    ax.legend(loc="upper left", framealpha=0.92, edgecolor="0.75",
              handlelength=2.2, labelspacing=0.4)

def plot_build_time(ax, data):
    bar_order = [
        ("JHQ-GPU-v5-CUDAGraph", "JHQ-GPU v5\nCUDA Graph"),
        ("Official-JHQ",          "JHQ Official\nCPU"),
        ("JHQ+IVF",               "JHQ Repro\nCPU"),
    ]
    order  = [(k, lbl) for k, lbl in bar_order if k in data]
    labels = [lbl for _, lbl in order]
    colors = [METHODS[k]["color"] for k, _ in order]
    hatches = ["//" if k.startswith("Official") else "" for k, _ in order]
    times  = [data[k]["build_time"] for k, _ in order]

    x    = np.arange(len(order))
    bars = ax.bar(x, times, color=colors, hatch=hatches,
                  edgecolor="white", linewidth=0.9,
                  width=0.5, zorder=3, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Index Build Time (s)")
    ax.set_title("(b) Index Construction Time", fontweight="bold", pad=6)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.0f}s" if v >= 10 else f"{v:.1f}s"))
    ax.grid(True, which="both", axis="y", ls=":", alpha=0.4)
    ax.set_ylim(top=max(times) * 6)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                t * 1.6, f"{t:.0f}s",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

def main():
    args = sys.argv[1:]
    cpu_csv = args[0]
    gpu_csv = args[1]
    prefix  = args[2] if len(args) > 2 else "results/jhq_v5_vs_cpu_vogue"

    data = load_csv(cpu_csv)
    data.update(load_csv(gpu_csv))

    fig = plt.figure(figsize=(12, 4.8))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.8, 1], wspace=0.30)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    plot_speed_accuracy(ax0, data)
    plot_build_time(ax1, data)

    fig.suptitle(
        "JHQ-GPU v5 vs CPU  —  Vogue-768  (d=768, N=932K, k=10)",
        fontsize=11, fontweight="bold", y=1.03
    )
    fig.savefig(prefix + ".pdf")
    fig.savefig(prefix + ".png", dpi=300)
    print(f"Saved: {prefix}.pdf")
    print(f"Saved: {prefix}.png")

if __name__ == "__main__":
    main()
