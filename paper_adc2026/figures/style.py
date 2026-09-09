"""Shared figure style and data loading for the ADC 2026 figures.

Camera-ready conventions, so a figure never has to be redrawn for the
submission:

* PDF out, with Type-42 fonts embedded -- Type-3 is rejected by most
  proceedings pipelines and is matplotlib's PDF default.
* Serif text at the size it will be *printed*, not the size it is drawn.
  A figure sized COL wide goes into a one-column slot at scale 1.0, so 8pt
  here is 8pt on the page. Never scale a figure in LaTeX; change the size
  here instead.
* No titles inside the axes. The caption is the title, and a duplicated one
  wastes a line of column.
* Every series carries a marker as well as a colour, so the figure survives
  greyscale printing and the two most common colour-vision deficiencies.
"""
import re, os, math, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT  = os.path.join(HERE, "out")

# Column widths in inches for a two-column ACM/IEEE layout.
COL, WIDE = 3.33, 6.9

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4,
    "lines.linewidth": 1.1, "lines.markersize": 3.2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "legend.handlelength": 1.8,
    "legend.borderpad": 0.2, "legend.labelspacing": 0.25,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})

# Hue order chosen so adjacent pairs stay separable under deuteranopia and
# protanopia; markers carry the same information again.
S = {
    "jhq":    dict(color="#2a78d6", marker="o", label="JHQ (this work)"),
    "jhq_fix":dict(color="#a9a7a0", marker="",  label=r"JHQ, fixed $\alpha{=}100$", ls="--"),
    "rabitq": dict(color="#e34948", marker="x", label="IVF-RaBitQ"),
    "cagra":  dict(color="#eb6834", marker="^", label="CAGRA fp32"),
    "cagra8": dict(color="#1baf7a", marker="s", label="CAGRA int8"),
    "ivfpq":  dict(color="#4a3aa7", marker="D", label="IVF-PQ"),
    "jq":     dict(color="#8f8d86", marker="v", label="JQ (primary only)"),
}
DATASETS = ["vogue-768", "arxiv-768", "bge-m3", "stella",
            "openai3-1536", "openai3-3072"]

# One hue and one marker per dataset, in DATASETS order, used by every figure
# that plots datasets as series -- the same dataset must not change colour
# between figures. Validated as a categorical palette against a light surface:
# lightness band, chroma floor and normal-vision separation all pass. Two
# results shape how it must be used:
#
#   * openai3-1536 and openai3-3072 (#e34948, #008300) separate by only
#     dE 7.2 under protanopia -- the 6-8 floor band, which is legal only with
#     a secondary encoding. The markers are that encoding, so never drop them.
#   * bge-m3 (#1baf7a) sits at 2.74:1 against the surface, under the 3:1 bar,
#     so a figure that leans on it needs a visible label, not a legend swatch
#     alone.
DS_COLOR = dict(zip(DATASETS, ["#2a78d6", "#eb6834", "#1baf7a",
                               "#4a3aa7", "#e34948", "#008300"]))
DS_MARK = dict(zip(DATASETS, ["o", "s", "^", "D", "x", "v"]))
PRETTY = {"vogue-768": "vogue-768", "arxiv-768": "arxiv-768", "bge-m3": "bge-m3",
          "stella": "stella-trec24", "openai3-1536": "openai3-1536",
          "openai3-3072": "openai3-3072"}


def datafile(name):
    return os.path.join(DATA, name)


def pareto(points):
    """Upper-left envelope: keep a point only if nothing cheaper reaches its recall."""
    out = []
    for r, q in sorted(points, key=lambda t: (-t[1], -t[0])):
        if not out or r > out[-1][0]:
            out.append((r, q))
    return sorted(out)


def load_fronts():
    """paper_fronts.log -> {dataset: {'rule': [...], 'fix': [...]}}, 72 runs, one build."""
    fix = collections.defaultdict(list)
    rule = collections.defaultdict(list)
    for ln in open(datafile("paper_fronts.log")):
        m = re.search(r"^  FIX\s+(\S+)\s+M=\d+\s+nlist=\d+\s+np=(\d+)\s+a=100 "
                      r"recall=([\d.]+)\s+qps=(\d+)", ln)
        if m:
            fix[m.group(1)].append((float(m.group(3)), int(m.group(4))))
        m = re.search(r"^  RULE\s+(\S+).*recall_picked=([\d.]+) qps_picked=(\d+)", ln)
        if m:
            rule[m.group(1)].append((float(m.group(2)), int(m.group(3))))
    return {d: {"rule": pareto(rule[d]), "fix": pareto(fix[d])} for d in fix}


def load_rabitq():
    """paper_rabitq.log plus the openai3-3072 QUANT4 sweep measured separately."""
    rq = collections.defaultdict(list)
    for ln in open(datafile("paper_rabitq.log")):
        m = re.search(r"^  (\S+)\s+nlist=\d+\s+np=(\d+)\s+\S+\s+recall=([\d.]+)\s+qps=(\d+)", ln)
        if m:
            rq[m.group(1)].append((float(m.group(3)), int(m.group(4))))
    rq["openai3-3072"] = [(0.8606, 76165), (0.9055, 67883), (0.9414, 58790),
                          (0.9666, 45244), (0.9839, 31218), (0.9943, 22095)]
    return {d: pareto(v) for d, v in rq.items() if v}


def load_baselines():
    """cuVS fronts from the v47 sweep; unchanged since, and not re-run."""
    import json
    p = os.path.join(HERE, "..", "..", "report_adc2026", "v47", "fronts.json")
    j = json.load(open(p))
    out = {}
    for ds, v in j.items():
        out[ds] = {"cagra": pareto([tuple(x) for x in v["series"].get("CAGRA fp32", [])]),
                   "cagra8": pareto([tuple(x) for x in v["series"].get("CAGRA int8", [])]),
                   "ivfpq": pareto([tuple(x) for x in v["series"].get("IVF-PQ", [])]),
                   "N": v["N"], "d": v["d"]}
    return out


def interp(points, r):
    """QPS at recall r, log-interpolated between measured points; None outside."""
    p = sorted(points)
    if not p or r < p[0][0] or r > p[-1][0]:
        return None
    for i in range(1, len(p)):
        if p[i][0] >= r:
            (r0, q0), (r1, q1) = p[i - 1][:2], p[i][:2]
            if r1 == r0:
                return q1
            f = (r - r0) / (r1 - r0)
            return 10 ** (math.log10(q0) + f * (math.log10(q1) - math.log10(q0)))
    return None


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  wrote out/{name}.pdf and .png")
