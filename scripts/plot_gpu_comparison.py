"""
GPU v1_plain vs CPU baselines comparison figure.

Usage (run locally after scp-ing jhq_v1_vogue.csv):
    python3 scripts/plot_gpu_comparison.py \
        ../JHQ_repro/results/vogue768_results.csv \
        jhq_v1_vogue.csv \
        [output_prefix]
"""
import sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":          "DejaVu Serif",
    "font.size":            11,
    "axes.labelsize":       12,
    "axes.titlesize":       11.5,
    "legend.fontsize":      8.0,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "axes.linewidth":       0.9,
    "lines.linewidth":      2.0,
    "lines.markersize":     7,
    "grid.linewidth":       0.45,
    "grid.alpha":           0.45,
    "figure.dpi":           180,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.06,
})

# ── Method palette ────────────────────────────────────────────────────────────
METHODS = {
    # ---- GPU (this work) ---------------------------------------------------
    "JHQ-GPU-v1":        dict(color="#17BECF", marker="*",  ls="-",  lw=2.4, zorder=7,
                              label="JHQ-GPU v1 (96 B/vec)",
                              mfc="#17BECF", mew=1.8, ms=10),
    "JHQ-GPU-v2":        dict(color="#111111", marker="P",  ls="-",  lw=2.2, zorder=8,
                              label="JHQ-GPU v2 top-k (480 B/vec)",
                              mfc="#111111", mew=1.6, ms=8),
    "JHQ-GPU-v3-IVF":    dict(color="#E377C2", marker="*",  ls="-",  lw=2.4, zorder=9,
                              label="JHQ-GPU v3 IVF (480 B/vec)",
                              mfc="#E377C2", mew=1.8, ms=10),
    "JHQ-GPU-v4-Batched": dict(color="#8C564B", marker="X",  ls="-",  lw=2.3, zorder=10,
                               label="JHQ-GPU v4 batched (480 B/vec)",
                               mfc="#8C564B", mew=1.6, ms=8),
    "JHQ-GPU-v5-CUDAGraph":  dict(color="#BCBD22", marker="D", ls="-", lw=2.3, zorder=11,
                                   label="JHQ-GPU v5 CUDA Graph (480 B/vec)",
                                   mfc="#BCBD22", mew=1.6, ms=7),
    # ---- Official source results -------------------------------------------
    "Official-JHQ":      dict(color="#D62728", marker="o",  ls="--", lw=2.2, zorder=6,
                              label="JHQ Official (480 B/vec)",
                              mfc="white", mew=2.0),
    "Official-JQ":       dict(color="#1F77B4", marker="s",  ls="--", lw=2.2, zorder=6,
                              label="JQ Official (96 B/vec)",
                              mfc="white", mew=2.0),
    # ---- Our CPU reproduction ----------------------------------------------
    "JHQ+IVF":           dict(color="#D62728", marker="o",  ls="-",  lw=2.2, zorder=5,
                              label="JHQ Repro (480 B/vec)",
                              mfc="#D62728", mew=1.8),
    "JQ+IVF":            dict(color="#1F77B4", marker="s",  ls="-",  lw=2.2, zorder=5,
                              label="JQ Repro (96 B/vec)",
                              mfc="#1F77B4", mew=1.8),
    # ---- FAISS baselines ---------------------------------------------------
    "FAISS-IVFPQ-96B":   dict(color="#2CA02C", marker="^",  ls=":",  lw=1.6, zorder=3,
                              label="FAISS IVFPQ (96 B/vec)",
                              mfc="white", mew=1.6),
    "FAISS-IVFPQ-192B":  dict(color="#FF7F0E", marker="D",  ls=":",  lw=1.6, zorder=3,
                              label="FAISS IVFPQ (192 B/vec)",
                              mfc="white", mew=1.6),
    "FAISS-IVFPQ-384B":  dict(color="#9467BD", marker="v",  ls=":",  lw=1.6, zorder=3,
                              label="FAISS IVFPQ (384 B/vec)",
                              mfc="white", mew=1.6),
}

# ── Load ─────────────────────────────────────────────────────────────────────
def load_csv(path):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            m = row["method"]
            if m not in data:
                data[m] = {"recall": [], "qps": [],
                            "nprobe": [], "build_time": float(row.get("build_time", 0))}
            data[m]["recall"].append(float(row["recall"]))
            data[m]["qps"].append(float(row["qps"]))
            data[m]["nprobe"].append(float(row["nprobe"]))
    return data

# ── QPS axis formatter ────────────────────────────────────────────────────────
def _qps_fmt(x, _):
    return f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"

# ── Panel (a): speed-accuracy ─────────────────────────────────────────────────
def plot_speed_accuracy(ax, data):
    for mname, style in METHODS.items():
        if mname not in data:
            continue
        d = data[mname]
        recalls = [r * 100 for r in d["recall"]]
        kw = dict(
            marker=style["marker"],
            color=style["color"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            markerfacecolor=style["mfc"],
            markeredgewidth=style["mew"],
            markeredgecolor=style["color"],
            markersize=style.get("ms", 7),
            zorder=style["zorder"],
            label=style["label"],
        )
        if mname == "JHQ-GPU-v1":
            # Plot all points as individual scatter markers, no connecting line.
            kw_scatter = dict(kw)
            kw_scatter["linestyle"] = "none"
            ax.semilogy(recalls, d["qps"], **kw_scatter)
            # Annotate only the two non-100%-recall points
            for r, q, a in zip(recalls, d["qps"], d["nprobe"]):
                if a in (1.0, 1.5):
                    ax.annotate(f"α={a:.1f}\n{r:.1f}%", xy=(r, q),
                                xytext=(-38, 4), textcoords="offset points",
                                fontsize=7, color=style["color"],
                                fontweight="bold")
        else:
            ax.semilogy(recalls, d["qps"], **kw)

    ax.set_xlabel("Recall@10 (%)")
    ax.set_ylabel("Queries per Second (QPS)")
    ax.set_title("(a) Speed–Accuracy Trade-off", fontweight="bold", pad=6)
    ax.set_xlim(55, 101)
    ax.set_ylim(bottom=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_qps_fmt))
    ax.grid(True, which="major", ls="-",  alpha=0.35)
    ax.grid(True, which="minor", ls=":",  alpha=0.20)
    for xv, lbl in [(90, "90%"), (95, "95%"), (99, "99%")]:
        ax.axvline(xv, color="gray", lw=0.7, ls=":", alpha=0.6)
        ax.text(xv + 0.3, ax.get_ylim()[0] * 1.5, lbl, fontsize=7.5,
                color="gray", rotation=90, va="bottom")
    ax.legend(loc="upper left", framealpha=0.92, edgecolor="0.75",
              handlelength=2.2, labelspacing=0.35, ncol=1, fontsize=7.5)

# ── Panel (b): build time ─────────────────────────────────────────────────────
def plot_build_time(ax, data):
    bar_order = [
        ("JHQ-GPU-v1",        "JHQ-GPU\nv1 (Ours)"),
        ("JHQ-GPU-v2",        "JHQ-GPU\nv2"),
        ("JHQ-GPU-v3-IVF",    "JHQ-GPU\nv3 IVF"),
        ("JHQ-GPU-v4-Batched", "JHQ-GPU\nv4"),
        ("JHQ-GPU-v5-CUDAGraph",  "JHQ-GPU\nv5"),
        ("Official-JHQ",      "Official\nJHQ"),
        ("Official-JQ",       "Official\nJQ"),
        ("JHQ+IVF",           "JHQ\nRepro"),
        ("JQ+IVF",            "JQ\nRepro"),
        ("FAISS-IVFPQ-96B",   "FAISS PQ\n96B"),
        ("FAISS-IVFPQ-192B",  "FAISS PQ\n192B"),
        ("FAISS-IVFPQ-384B",  "FAISS PQ\n384B"),
    ]
    order   = [(k, lbl) for k, lbl in bar_order if k in data]
    labels  = [lbl for _, lbl in order]
    colors  = [METHODS[k]["color"] for k, _ in order]
    hatches = ["//" if k.startswith("Official") else "" for k, _ in order]
    times   = [data[k]["build_time"] for k, _ in order]

    x    = np.arange(len(order))
    bars = ax.bar(x, times, color=colors, hatch=hatches,
                  edgecolor="white", linewidth=0.9,
                  width=0.62, zorder=3, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Index Build Time (s)")
    ax.set_title("(b) Index Construction Time", fontweight="bold", pad=6)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.0f}s" if v >= 10 else f"{v:.1f}s"))
    ax.grid(True, which="both", axis="y", ls=":", alpha=0.4)
    ax.set_ylim(top=max(times) * 6)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                t * 1.6, f"{t:.0f}s",
                ha="center", va="bottom", fontsize=8.0, fontweight="bold")

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: plot_gpu_comparison.py <cpu_results.csv> <gpu_results.csv> [output_prefix]")
        sys.exit(1)

    cpu_csv = args[0]
    gpu_csv = args[1]
    prefix  = args[2] if len(args) > 2 else "results/jhq_vogue768_comparison"

    data = load_csv(cpu_csv)
    data.update(load_csv(gpu_csv))

    fig = plt.figure(figsize=(14, 4.8))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.70, 1], wspace=0.30)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    plot_speed_accuracy(ax0, data)
    plot_build_time(ax1, data)

    fig.suptitle(
        "JHQ-GPU vs CPU Methods  —  Vogue-768  (d=768, N=932K, k=10)",
        fontsize=11, fontweight="bold", y=1.03
    )

    fig.savefig(prefix + ".pdf")
    fig.savefig(prefix + ".png", dpi=300)
    print(f"Saved: {prefix}.pdf")
    print(f"Saved: {prefix}.png")

if __name__ == "__main__":
    main()
