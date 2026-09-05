#!/usr/bin/env python3
"""The index build: train + add on six datasets against the CPU reference and
the first GPU port, and where the largest add's time goes.

Reads results/pre_freeze_v22_s2b1/build_gpu.csv and build_stella_add_phases.csv, writes
results/pre_freeze_v22_s2b1/fig4_build.png.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "final")
ACC, MUTED, NO, SOFT = "#1F5F8B", "#5C6773", "#A2452E", "#9AA5B1"
LABEL = {"vogue": "Vogue-768\n0.93M", "arxiv-768": "arXiv-768\n2.25M",
         "openai3-1536": "OpenAI3-1536\n1.0M", "openai3-3072": "OpenAI3-3072\n1.0M",
         "bge-m3": "BGE-M3-1024\n10.1M", "stella-trec24": "Stella-1024\n17.8M"}


def read(name):
    with open(os.path.join(D, name)) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def main():
    rows = read("build_gpu.csv")
    # one bar per dataset: the single-pass row where there are two settings
    seen, sel = set(), []
    for r in rows:
        if r["dataset"] in seen or r["pass"] != "single":
            continue
        seen.add(r["dataset"]); sel.append(r)
    names = [LABEL.get(r["dataset"], r["dataset"]) for r in sel]
    cpu = [float(r["cpu_ref_s"]) for r in sel]
    first = [float(r["first_port_s"]) for r in sel]
    now = [(float(r["train_ms"]) + float(r["add_ms"])) / 1000 for r in sel]
    stella = next(r for r in sel if r["dataset"] == "stella-trec24")
    stella_add = float(stella["add_ms"]) / 1000

    fig = plt.figure(figsize=(15.4, 6.2), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0]); bx = fig.add_subplot(gs[0, 1])

    y = np.arange(len(names))[::-1]; h = 0.26
    ax.barh(y + h, cpu, h, color=SOFT,
            label="JHQ-CPU-IVF (reference, Xeon 8470Q, 2.41 TFLOP/s at its 32-thread peak)")
    ax.barh(y, first, h, color="#7FA6C2", label="first GPU port (streaming add)")
    ax.barh(y - h, now, h, color=ACC, label="now")
    for yy, v in zip(y + h, cpu):
        ax.text(v * 1.08, yy, f"{v:.0f} s", va="center", fontsize=9.5, color=MUTED)
    for yy, v in zip(y, first):
        ax.text(v * 1.08, yy, f"{v:.0f} s", va="center", fontsize=9.5, color="#4F6E85")
    for yy, v in zip(y - h, now):
        ax.text(v * 1.08, yy, f"{v:.2f} s", va="center", fontsize=9.5, color=ACC,
                fontweight="bold")
    # cuVS IVF-PQ on Vogue, same 96-byte code, is the fastest builder measured.
    ax.plot([1.58], [y[0] - h], marker="v", color=NO, ms=8, ls="none", zorder=5)
    ax.text(1.58 * 1.25, y[0] - h, "cuVS IVF-PQ, same 96-byte code: 1.58 s",
            va="center", fontsize=8.8, color=NO)
    ax.set_xscale("log"); ax.set_xlim(0.2, 900)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("index build, train + add, seconds (log)", fontsize=10.5)
    ax.grid(axis="x", alpha=0.3, which="both"); ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.11), ncol=3, fontsize=9,
              frameon=False)
    r_cpu = [c / n for c, n in zip(cpu, now)]
    r_first = [f / n for f, n in zip(first, now)]
    # The ratio alone invites the obvious objection, so the title carries the
    # part of it that is the machine: at 17.8M the two implementations reach
    # 50% and 2.8% of their own arithmetic bounds, and the GPU is doing 13.6x
    # the arithmetic at nlist 16384 against 256. See gpu_operating_point/roofline.csv.
    ax.set_title(f"Build time: {min(r_cpu):.0f}x to {max(r_cpu):.0f}x below the CPU "
                 f"reference — of which about 10x is the arithmetic bound and the "
                 f"rest is distance from it",
                 fontsize=11.5, loc="left")

    # Stella add, pipelined: what the seconds are, and what runs under them.
    phases = read("build_stella_add_phases.csv")
    colours = [ACC, "#4F87B0", "#7FA6C2", "#B3C7D6"]
    gpu = [(p["phase"], float(p["ms"]) / 1000) for p in phases if p["where"] == "gpu"]
    over = [(p["phase"], float(p["ms"]) / 1000) for p in phases if p["where"] != "gpu"]
    parts = [(lab, w, c) for (lab, w), c in zip(gpu, colours)]
    parts.append(("list build, pipeline fill", stella_add - sum(w for _, w in gpu), SOFT))
    left = 0.0
    for lab, w, c in parts:
        bx.barh(1.0, w, 0.5, left=left, color=c, edgecolor="white", linewidth=0.6)
        if w > 0.25:
            bx.text(left + w / 2, 1.0, f"{w:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if c in (ACC, "#4F87B0") else "#141A21")
        left += w
    for (lab, w), yy, hatch in zip(over, (0.3, -0.2), ("////", "....")):
        bx.barh(yy, w, 0.32, color="none", edgecolor=MUTED, hatch=hatch, linewidth=0.8)
        bx.text(w + 0.08, yy, f"{lab}: {w:.2f} s", va="center", fontsize=9, color=MUTED)
    bx.set_xlim(0, stella_add * 1.06); bx.set_ylim(-0.7, 1.7)
    bx.set_yticks([1.0, 0.3, -0.2])
    bx.set_yticklabels(["on the GPU\n(critical path)", "overlapped", "overlapped"],
                       fontsize=9.5)
    bx.set_xlabel(f"Stella-TREC24 add, 17.8M x 1024, {stella_add:.2f} s pipelined",
                  fontsize=10.5)
    bx.grid(axis="x", alpha=0.3); bx.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts]
    bx.legend(handles, [p[0] for p in parts], loc="upper center", bbox_to_anchor=(0.5, 1.0),
              fontsize=8.6, ncol=2, frameon=False)
    bx.set_title("Where the largest add goes", fontsize=11.5, loc="left")
    for a in (ax, bx):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    out = os.path.join(D, "fig4_build.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(os.path.basename(out))


if __name__ == "__main__":
    main()
