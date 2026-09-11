#!/usr/bin/env python3
"""What the refinement budget is worth on a GPU against a CPU.

Section 6.5 states a 52-108x GPU-over-CPU range in one clause, which needs no
figure. This is the other half of that experiment and the paper makes its
argument nowhere else: the same alpha knob, on the same dataset, at the same
matched recall, is worth 2.5x to 3.0x more on the GPU than on the CPU at every
probe depth.

That is the reason Section 4 exists. On the CPU the scan dominates and the
budget barely moves throughput; on the GPU the scan is what Section 3 makes
cheap, so refinement takes the larger share and the same reduction in
survivors buys more. Selective refinement is therefore a GPU design decision
rather than a CPU idea carried across -- a claim the stage shares in Section
6.2 support from inside JHQ, and this supports from outside it.

Panel (a) of the original two-panel version drew both throughput envelopes,
which is the 52-108x clause as a picture. It is dropped: at a 12-page limit a
figure has to say something the prose cannot.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import *          # noqa: F401,F403
import style
import fig_cpu


def main():
    nps, cpu, gpu = fig_cpu.alpha_lever()
    fig, b = plt.subplots(figsize=(WIDE * 0.62, 2.05))
    xs = range(len(nps))
    w = 0.36
    b.bar([x - w / 2 for x in xs], [cpu[n] for n in nps], w,
          color=fig_cpu.CPU_C, label="CPU, 32 threads")
    b.bar([x + w / 2 for x in xs], [gpu[n] for n in nps], w,
          color=DS_COLOR["openai3-3072"], label="GPU, RTX 5090")
    for x, n in zip(xs, nps):
        b.text(x, max(cpu[n], gpu[n]) + 3, r"$%.1f\times$" % (gpu[n] / cpu[n]),
               ha="center", fontsize=7, color="#52514e")
    b.set_xticks(list(xs))
    b.set_xticklabels([str(n) for n in nps])
    b.set_xlabel("nprobe")
    b.set_ylabel(r"QPS given up by $\alpha{=}100$ (\%)")
    b.set_ylim(0, max(gpu.values()) * 1.30)
    b.legend(loc="upper right", fontsize=7)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_alphalever")
    for n in nps:
        print("  nprobe=%-5d CPU %+5.1f%%  GPU %+6.1f%%  ratio %.2fx"
              % (n, cpu[n], gpu[n], gpu[n] / cpu[n]))


if __name__ == "__main__":
    main()
