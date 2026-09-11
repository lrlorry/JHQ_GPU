#!/usr/bin/env python3
"""Two LaTeX tables the body needs, generated so no number is typed by hand.

`frontier`  -- who is fastest at each matched recall, every baseline included.
              Counts are recomputed from the current frozen inputs. A reviewer reading Figure 5
              works this out in a minute; stating it is the only way the paper
              keeps its credibility while claiming the region it does own.

`batch`     -- the matched-recall batch ratios. This replaces fig_batch, which
              was drawn from batch_sweep.log (a fixed-nprobe experiment) while
              the caption and the surrounding text described batch_matched.log.
              The figure was showing a different experiment from the one the
              text reported.

Both use log-linear interpolation of each system's curve at the target recall.
The cruder max(qps | recall >= target) disagrees -- 2.14 where interpolation
gives 2.23 on openai3-3072 at batch 32 -- because the two systems land at
slightly different recalls on a shared nprobe grid.
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style

# The table and figure include all main GPU baselines.
NAME = {"ivfpq": "IVF-PQ", "cagra": "CAGRA-fp32", "cagra8": "CAGRA-int8"}
RECALLS = (0.90, 0.93, 0.95, 0.97, 0.98, 0.99)


MARGINS = collections.defaultdict(list)
MARGINS_LOW = collections.defaultdict(list)
# Wins and margins are different counts: a cell can have a winner and no
# JHQ point to measure the margin against.  Reporting the margin count as
# the win count loses those cells and the totals stop summing to the table.
WINS = collections.Counter()


def frontier_table():
    f, rq, bl = style.load_fronts(), style.load_rabitq(), style.load_baselines()
    lines, win, tot = [], 0, 0
    for ds in style.DATASETS:
        if ds not in f:
            continue
        cells = []
        for R in RECALLS:
            row = []
            q = style.interp(f[ds]["rule"], R)
            if q:
                row.append((q, "JHQ"))
            if ds in rq:
                q = style.interp(rq[ds], R)
                if q:
                    row.append((q, "RaBitQ"))
            for nm, pts in bl.get(ds, {}).items():
                if nm not in NAME:
                    continue
                q = style.interp(style.pareto(pts), R)
                if q:
                    row.append((q, NAME[nm]))
            if len(row) < 2:
                cells.append("--")
                continue
            row.sort(reverse=True)
            tot += 1
            best = row[0][1]
            jhq = dict((n, v) for v, n in row).get("JHQ")
            if best == "JHQ":
                win += 1
                cells.append(r"\textbf{JHQ}")
            else:
                WINS[best] += 1
                # "CAGRA-int8 $2.7\times$" made the six-column table 30pt too
                # wide for the LNCS block; the caption expands the names.
                short = best.replace("CAGRA-", "").replace("RaBitQ", "RaBitQ")
                if jhq:
                    MARGINS[best].append(row[0][0] / jhq)
                    # The introduction makes a narrower claim than the table --
                    # the cells below R=0.95 -- and it needs its own range, or
                    # it gets written by hand from a different subset than the
                    # one it names.
                    if R < 0.95:
                        MARGINS_LOW[best].append(row[0][0] / jhq)
                cells.append("%s $%.1f\\times$" % (short, row[0][0] / jhq)
                             if jhq else short)
        lines.append("%s & %s \\\\" % (style.PRETTY[ds], " & ".join(cells)))
    return lines, win, tot


def batch_table():
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for ln in open(style.datafile("batch_matched4.log"), errors="ignore"):
        m = re.search(r"^\s*(JHQ|RaBitQ)\s+(\S+)\s+np=(\d+)\s+(?:a=[\d.]+\s+)?"
                      r"batch=(\d+)\s+recall=([\d.]+)\s+qps=(\d+)", ln)
        if m:
            rows[(m.group(2), int(m.group(4)))][m.group(1)].append(
                (float(m.group(5)), int(m.group(6))))
    out = []
    DS4 = ("vogue-768", "arxiv-768", "openai3-1536", "openai3-3072")
    for b in (32, 128, 512, 1024):
        cells = []
        for ds in DS4:
            cell = rows.get((ds, b), {})
            for tgt in (0.90, 0.95):
                v = [style.interp(sorted(cell.get(s, [])), tgt)
                     for s in ("JHQ", "RaBitQ")]
                cells.append("$%.2f$" % (v[0] / v[1]) if all(v) else "--")
        out.append("%d & %s \\\\" % (b, " & ".join(cells)))
    return out


def main():
    lines, win, tot = frontier_table()
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, "tex", "tables")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "frontier.tex"), "w") as fh:
        fh.write("%% generated by figures/table_frontier.py -- do not edit\n")
        fh.write("\\footnotesize\n\\begin{tabular}{l%s}\n\\toprule\n"
                 % ("r" * len(RECALLS)))
        fh.write("& \\multicolumn{%d}{c}{fastest system at Recall@10}\\\\\n"
                 % len(RECALLS))
        fh.write("\\cmidrule(l){2-%d}\n" % (len(RECALLS) + 1))
        fh.write("dataset & %s \\\\\n\\midrule\n"
                 % " & ".join("$%.2f$" % r for r in RECALLS))
        fh.write("\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(dst, "batch.tex"), "w") as fh:
        fh.write("%% generated by figures/table_frontier.py -- do not edit\n")
        # "batch" and the first ratio collided at uniform spacing -- 32 0.68
        # read as one number -- and the four dataset pairs had nothing
        # separating them.
        fh.write("\\begin{tabular}{@{}r@{\\qquad}r@{\\ }r@{\\qquad}"
                 "r@{\\ }r@{\\qquad}r@{\\ }r@{\\qquad}r@{\\ }r@{}}\n"
                 "\\toprule\n")
        fh.write("& \\multicolumn{2}{c}{vogue-768} & \\multicolumn{2}{c}{arxiv-768}"
                 " & \\multicolumn{2}{c}{openai3-1536}"
                 " & \\multicolumn{2}{c}{openai3-3072}\\\\\n"
                 "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}"
                 "\\cmidrule(lr){6-7}\\cmidrule(l){8-9}\n")
        fh.write("batch & $.90$ & $.95$ & $.90$ & $.95$ & $.90$ & $.95$ "
                 "& $.90$ & $.95$ \\\\\n\\midrule\n")
        fh.write("\n".join(batch_table()) + "\n\\bottomrule\n\\end{tabular}\n")
    print("  wrote tex/tables/frontier.tex  (JHQ fastest in %d of %d cells)"
          % (win, tot))
    print("  ranges for the body, so they are not typed by hand:")
    for nm, v in sorted(MARGINS.items()):
        print("     %-11s leads in %2d cells (%d with a JHQ point to compare), "
              "margin over JHQ %.1fx to %.1fx"
              % (nm, WINS[nm], len(v), min(v), max(v)))
    print("     JHQ leads in %d; wins sum to %d of %d cells"
          % (win, win + sum(WINS.values()), tot))
    print("  and over the cells below Recall@10=0.95 only:")
    for nm, v in sorted(MARGINS_LOW.items()):
        print("     %-11s leads in %2d cells, margin over JHQ %.1fx to %.1fx"
              % (nm, len(v), min(v), max(v)))
    # CAGRA is off the frontier table by scope, so its numbers need their own
    # source or they go back to being typed by hand.  Same interpolation, same
    # recall grid, JHQ against each CAGRA variant one at a time.
    f, rq, bl = style.load_fronts(), style.load_rabitq(), style.load_baselines()
    print("  the graph baseline, reported in its own subsection:")
    for key, nm in (("cagra8", "CAGRA-int8"), ("cagra", "CAGRA-fp32")):
        lead, low, mg, mglow = 0, 0, [], []
        cells = 0
        for ds in style.DATASETS:
            for R in RECALLS:
                j = style.interp(f[ds]["rule"], R)
                c = style.interp(style.pareto(bl.get(ds, {}).get(key, [])), R)
                if not (j and c):
                    continue
                cells += 1
                if c > j:
                    lead += 1; mg.append(c / j)
                    if R < 0.95:
                        low += 1; mglow.append(c / j)
        if mg:
            print("     %-11s leads JHQ in %d of %d comparable cells, "
                  "by %.1fx to %.1fx" % (nm, lead, cells, min(mg), max(mg)))
            if mglow:
                print("       below R=0.95: %d cells, %.1fx to %.1fx"
                      % (low, min(mglow), max(mglow)))

    q = style._quarantined()
    if q:
        print("  excluded as not status=ok before any envelope was taken:")
        for r, v in sorted(q):
            print("     recall=%.4f qps=%.0f" % (r, v))
    print("  wrote tex/tables/batch.tex")


if __name__ == "__main__":
    main()
