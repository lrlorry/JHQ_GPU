#!/usr/bin/env python3
"""Two sensitivity results that were two tables of small numbers.

Section 6.5 had a six-column table whose content was a failure rate falling
with the calibration sample, and Section 6.6 a nine-column one whose content
was a ratio crossing 1 as the batch grows. Both are one quantity against one
knob, which is a line, and both were hard to read as digits: the budget table's
interesting column was buried among near-constant ones, and the batch table's
thirty-two numbers had to be scanned pairwise to find the crossing.

(a) The share of 64 random calibration samples that give up more than 1e-3 of
held-out recall, against S. It falls on every cell and reaches zero on neither
vogue-768 cell, which is the reason Section 4's S=32 is withdrawn and S=128 is
described as small rather than safe.

(b) JHQ over IVF-RaBitQ at matched recall, against batch size, on the four
datasets IVF-RaBitQ builds on. Above 1 is JHQ ahead. vogue-768 crosses parity
between 128 and 512, The batch experiment and main frontier are independent measurements.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from style import *          # noqa: F401,F403
import budget_arms
import style


def batch_ratios():
    """(dataset, batch) -> JHQ/RaBitQ at each matched recall target."""
    import collections
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for ln in open(style.datafile("batch_matched4.log"), errors="ignore"):
        m = re.search(r"^\s*(JHQ|RaBitQ)\s+(\S+)\s+np=(\d+)\s+(?:a=[\d.]+\s+)?"
                      r"batch=(\d+)\s+recall=([\d.]+)\s+qps=(\d+)", ln)
        if m:
            rows[(m.group(2), int(m.group(4)))][m.group(1)].append(
                (float(m.group(5)), int(m.group(6))))
    out = collections.defaultdict(dict)
    for (ds, b), cell in rows.items():
        for tgt in (0.90, 0.95):
            v = [style.interp(sorted(cell.get(s, [])), tgt)
                 for s in ("JHQ", "RaBitQ")]
            if all(v):
                out[(ds, tgt)][b] = v[0] / v[1]
    return out


def main():
    C = budget_arms.load()
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.15))

    # ── (a) sampling risk against S ────────────────────────────────────────
    SS = [32, 64, 128]
    for (ds, np_), c in C.items():
        y = [100 * c["rule"][S]["over"] for S in SS]
        a.plot(SS, y, color=DS_COLOR[ds], marker=DS_MARK[ds], mec="white",
               ls="-" if np_ == 128 else "--",
               label=r"%s, $\mathit{np}{=}%d$" % (PRETTY[ds].split("-")[0], np_))
    a.set_xscale("log", base=2)
    a.set_xticks(SS); a.set_xticklabels([str(s) for s in SS])
    a.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
    a.set_xlabel(r"calibration sample $S$")
    a.set_ylabel(r"draws losing $>10^{-3}$ (\%)")
    a.set_ylim(-4, 78)
    a.legend(loc="upper right", fontsize=7, labelspacing=0.2,
             handlelength=1.5, borderpad=0.25)
    a.text(0.03, 0.05, "(a)", transform=a.transAxes, fontsize=8)

    # ── (b) the batch ratio ───────────────────────────────────────────────
    br = batch_ratios()
    BATCH = [32, 128, 512, 1024]
    # One recall target, not two.  Eight lines made the panel unreadable and
    # both targets tell the same story: vogue-768 crosses parity between 128
    # and 512 and the other three stay above it.  0.95 is the harder target
    # and the one Section 6.2's frontier ratios sit at.
    for (ds, tgt), d in sorted(br.items()):
        if tgt != 0.95:
            continue
        xs = [x for x in BATCH if x in d]
        if len(xs) < 2:
            continue
        # "openai3" alone names two different datasets.
        b.plot(xs, [d[x] for x in xs], color=DS_COLOR[ds], marker=DS_MARK[ds],
               mec="white", ls="-", label=PRETTY[ds])
    b.axhline(1.0, color="0.45", lw=0.7, ls=":")
    b.set_xscale("log", base=2)
    b.set_xticks(BATCH); b.set_xticklabels([str(x) for x in BATCH])
    b.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
    b.set_xlabel("batch size")
    b.set_ylabel(r"JHQ $\div$ IVF-RaBitQ at $R{=}0.95$")
    b.legend(loc="upper left", fontsize=7, ncol=1, labelspacing=0.2,
             handlelength=1.5, borderpad=0.25)
    b.text(0.97, 0.05, "(b)", transform=b.transAxes, fontsize=8, ha="right")

    fig.tight_layout(pad=0.3, w_pad=1.4)
    save(fig, "fig_knobs")


if __name__ == "__main__":
    main()
