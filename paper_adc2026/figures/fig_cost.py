#!/usr/bin/env python3
"""Figure: what the index costs to build and to hold.

Section 6.7. Two panels, both of them numbers that go against this work and
are reported anyway.

(a) Resident GPU memory, same call on both sides: cudaMemGetInfo after the
    index is built and the search workspace allocated. IVF-RaBitQ is smaller
    wherever it builds, by more than its bits per dimension alone predict --
    8 against 9 is 11%, and the measured gap is 22-39%, because JHQ also
    carries the selection buffer and the factorised table in its workspace.
    On the two largest sets IVF-RaBitQ does not build at the parameters its
    defaults choose; that is a bar that is absent, not a bar at zero.
(b) Build time, JHQ, split into training and encoding.
"""
import sys, os, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import numpy as np

rq_vram = {}
for ln in open(datafile("vram.log")):
    m = re.search(r"^  (\S+)\s+nlist=\d+\s+vram=([\d.]+)", ln)
    if m:
        rq_vram[m.group(1)] = float(m.group(2))

jhq_vram, train, add = {}, {}, {}
for ln in open(datafile("paper_fronts.log")):
    m = re.search(r"^  FIX\s+(\S+)\s+M=\d+\s+nlist=\d+\s+np=\d+\s+a=100 "
                  r"recall=[\d.]+\s+qps=\d+\s+train=([\d.]+)\s+add=([\d.]+)\s+"
                  r"vram=([\d.]+)", ln)
    if m:
        ds = m.group(1)
        jhq_vram[ds] = float(m.group(4))
        # train() hits the cache after the first run of a dataset, so the
        # first row is the one that trained.
        train.setdefault(ds, float(m.group(2)))
        add.setdefault(ds, float(m.group(3)))

order = [d for d in DATASETS if d in jhq_vram]
y = np.arange(len(order))
fig, (a, b) = plt.subplots(1, 2, figsize=(WIDE, 2.1))

jv = [jhq_vram[d] / 1024 for d in order]
rv = [rq_vram.get(d, np.nan) / 1024 if d in rq_vram else np.nan for d in order]
a.barh(y - 0.19, rv, height=0.36, color=S["rabitq"]["color"], label="IVF-RaBitQ")
a.barh(y + 0.19, jv, height=0.36, color=S["jhq"]["color"], label="JHQ")
for i, d in enumerate(order):
    if d not in rq_vram:
        a.text(0.4, i - 0.19, "does not build", va="center", fontsize=6,
               color=S["rabitq"]["color"], style="italic")
    else:
        a.text(jv[i] + 0.3, i + 0.19, f"+{100*(jv[i]/rv[i]-1):.0f}\\%",
               va="center", fontsize=6)
a.set_yticks(y); a.set_yticklabels([PRETTY[d] for d in order])
a.set_xlabel("resident GPU memory (GiB)"); a.invert_yaxis()
a.grid(axis="y", visible=False); a.legend(loc="lower right", fontsize=6.5)
a.set_xlim(0, max(jv) * 1.25)
a.text(0.97, 0.05, "(a)", transform=a.transAxes, fontsize=8, ha="right")

tv = [train[d] / 1000 for d in order]
av = [add[d] / 1000 for d in order]
b.barh(y, tv, height=0.5, color="#4a3aa7", label="train")
b.barh(y, av, height=0.5, left=tv, color="#1baf7a", label="encode")
for i in range(len(order)):
    b.text(tv[i] + av[i] + 0.4, i, f"{tv[i]+av[i]:.0f}s", va="center", fontsize=6)
b.set_yticks(y); b.set_yticklabels([])
b.set_xlabel("JHQ index build (s)"); b.invert_yaxis()
b.grid(axis="y", visible=False); b.legend(loc="lower right", fontsize=6.5)
b.set_xlim(0, max(t + a2 for t, a2 in zip(tv, av)) * 1.2)
b.text(0.97, 0.05, "(b)", transform=b.transAxes, fontsize=8, ha="right")

fig.tight_layout(pad=0.3)
save(fig, "fig_cost")
