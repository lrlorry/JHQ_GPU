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
import re, os, math, collections, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT  = os.path.join(HERE, "out")

# Column width in inches. The target is the ADC LNCS template: single column,
# \textwidth = 347.12pt = 4.803in. A figure MUST be drawn at this width and
# included with width=\linewidth, so the scale factor is exactly 1 and the
# sizes below are the sizes on the page.
#
# They were 3.33/6.9 for a two-column ACM slot. Included at \linewidth in LNCS
# that is a 0.696x reduction, which turned 8pt axis text into 5.6pt and 6.5pt
# annotations into 4.5pt. Never scale a figure in LaTeX; change the size here.
COL = WIDE = 4.803

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "axes.edgecolor": "#4a4844",
    "axes.labelcolor": "#26241f", "text.color": "#26241f",
    "lines.linewidth": 1.15, "lines.markersize": 3.4,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    # A marker sitting directly on its own line, and on whatever line crosses
    # it, is the thing that makes a multi-series plot look muddy.  A thin
    # background-coloured rim around every marker separates them, which is why
    # every published frontier plot has one.
    # Set per series below, not here: "x" and "+" are drawn as edges with no
    # face, so a global white edge colour erases them outright.
    "lines.markeredgewidth": 0.5,
    # Ticks inward and short: outward ticks push the axes apart and read as a
    # default plot.  Minor ticks on both sides so a log axis looks like one.
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
    "xtick.color": "#4a4844", "ytick.color": "#4a4844",
    "xtick.labelcolor": "#26241f", "ytick.labelcolor": "#26241f",
    "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "legend.handlelength": 1.8,
    "legend.borderpad": 0.2, "legend.labelspacing": 0.25,
    "legend.handletextpad": 0.5,
    # The grid is a reading aid, not a element of the data: dotted, faint, and
    # underneath everything drawn.
    "axes.grid": True, "grid.alpha": 0.5, "grid.linestyle": ":",
    "grid.linewidth": 0.4, "grid.color": "#b8b5ae",
    "axes.axisbelow": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})

# Hue order chosen so adjacent pairs stay separable under deuteranopia and
# protanopia; markers carry the same information again.
S = {
    # Neither "(this work)" nor "(ours)".  The first claimed the method, which
    # the paper explicitly does not -- the transform, the primary codebook and
    # the residual hierarchy are JHQ's, and Related Work concedes the grouped
    # table too.  "(ours)" makes the same claim in a softer voice, on a curve
    # whose name is the prior method's.  What is ours is the execution path and
    # the calibration rule, and the paper says so in words; a legend that
    # annotates one of four series and not the others reads as advocacy rather
    # than labelling, and the thickest blue line is not hard to find.
"jhq":    dict(color="#1f6fc4", marker="o", label="JHQ-GPU",
                   mec="white"),
    "jhq_fix":dict(color="#9c9a93", marker="",
                   label=r"JHQ-GPU, fixed $\alpha{=}100$", ls=(0, (4, 2))),
    # Filled markers throughout, each with the page's white as a rim, so a
    # crossing does not merge two series into one shape.
    "rabitq": dict(color="#d1443f", marker="s", label="IVF-RaBitQ", mec="white"),
    "cagra":  dict(color="#e07b27", marker="^", label="CAGRA fp32", mec="white"),
    "cagra8": dict(color="#17a074", marker="D", label="CAGRA int8", mec="white"),
    "ivfpq":  dict(color="#5b4bb8", marker="v", label="IVF-PQ", mec="white"),
    "jq":     dict(color="#8f8d86", marker="P", label="JQ (primary only)",
                   mec="white"),
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
    """The frontier: paper_fronts.log, with openai3-3072 re-measured.

    nlist was tuned per dataset -- results/front6/nlist6.log sweeps four values
    each -- and five of the six datasets run at the value that sweep picked.
    openai3-3072 ran at 4096 while its own sweep picks 8192 at every recall
    target it resolves.  run_front6.sh's second block took both OpenAI sets
    from 1024 to 4096; openai3-1536 was later moved to 8192 and this one was
    not, so the highest-dimensional dataset was measured at a partition its own
    experiment rejected.

    o30_nlist8192.log re-runs it under /root/_pa.sh's protocol, changing only
    nlist and the 39-points-a-centroid training count that follows it.  At
    matched recall it is worth 1.04x to 1.19x, and it removes a confound from
    Section 6.2's dimension argument: at 4096 this dataset held 244 vectors a
    list against openai3-1536's 122, so the same probe depth scanned twice the
    candidates on the higher-dimensional set.  Both hold 122 now.

    The 4096 rows stay in paper_fronts.log; they are skipped here rather than
    deleted, so the pair remains available as the measurement of what the
    partition is worth.
    """
    fix = collections.defaultdict(list)
    rule = collections.defaultdict(list)
    REPLACED = {"openai3-3072"}          # superseded by the file below
    for src in ("paper_fronts.log", "o30_nlist8192.log"):
        p = datafile(src)
        if not os.path.exists(p):
            continue
        for ln in open(p, errors="ignore"):
            m = re.search(r"^\s*FIX\s+(\S+)\s+M=\d+\s+nlist=(\d+)\s+np=(\d+)\s+a=100 "
                          r"recall=([\d.]+)\s+qps=(\d+)", ln)
            if m and not (src == "paper_fronts.log" and m.group(1) in REPLACED):
                fix[m.group(1)].append((float(m.group(4)), int(m.group(5))))
            m = re.search(r"^\s*RULE\s+(\S+).*recall_picked=([\d.]+) qps_picked=(\d+)", ln)
            if m and not (src == "paper_fronts.log" and m.group(1) in REPLACED):
                rule[m.group(1)].append((float(m.group(2)), int(m.group(3))))
    return {d: {"rule": pareto(rule[d]), "fix": pareto(fix[d])} for d in fix}


def load_rabitq():
    """IVF-RaBitQ at the search kernel it is fastest on, per dataset.

    The archived sweep (paper_rabitq.log) ran every dataset at cuVS's QUANT4
    mode and never recorded that it had.  A mode sweep afterwards found QUANT4
    is *not* the best kernel on three of the four datasets RaBitQ builds on --
    LUT32 wins on vogue-768 and arxiv-768 by 14% and 6%, LUT16 on openai3-1536
    by 6% -- so the published frontier compared JHQ against a baseline below
    its own best.  rq_best.log re-runs each dataset at its winner, and is used
    wherever it covers a dataset; bge-m3 and stella are absent because the tested build paths failed.
    """
    rq = collections.defaultdict(list)
    for ln in open(datafile("paper_rabitq.log")):
        m = re.search(r"^  (\S+)\s+nlist=\d+\s+np=(\d+)\s+\S+\s+recall=([\d.]+)\s+qps=(\d+)", ln)
        if m:
            rq[m.group(1)].append((float(m.group(3)), int(m.group(4))))
    rq["openai3-3072"] = [(0.8606, 76165), (0.9055, 67883), (0.9414, 58790),
                          (0.9666, 45244), (0.9839, 31218), (0.9943, 22095)]
    best = collections.defaultdict(list)
    p = datafile("rq_best.log")
    if os.path.exists(p):
        for ln in open(p, errors="ignore"):
            m = re.match(r"\s*RQ\s+(\S+)\s+nlist=\d+\s+tpl=\d+\s+used=\S+\s+"
                         r"mode=\d+\s+np=\d+\s+recall=([\d.]+)\s+qps=(\d+)", ln)
            if m:
                best[m.group(1)].append((float(m.group(2)), int(m.group(3))))
    for d, v in best.items():
        if len(v) >= 3:                 # a partial run must not replace a full one
            rq[d] = v
    return {d: pareto(v) for d, v in rq.items() if v}


def _quarantined():
    """(recall, qps) pairs whose source row is not status=ok.

    fronts.json was built by scripts/make_figures.py, which drops a row only
    when status == "FAILED".  One row in the sweep is marked CONTAMINATED
    instead -- stella-trec24, CAGRA-int8, itopk=64, whose own failures field
    reads "QPS spread 6.6% against 0.02-0.64% on an idle card", i.e. the card
    was not idle while it was timed -- and it went into the frontier at
    618,859 QPS, which is 1.6x the next point down.  A timing taken on a busy
    card is not a slower measurement of the same thing; it is not a
    measurement of this system at all, and it carried a 2.6x lead at
    stella's R=0.97 cell.

    The exclusion is derived from the CSVs rather than hard-coded, so a row
    that is re-run and cleared stops being excluded without an edit here.
    """
    import csv, glob
    bad = set()
    root = os.path.join(HERE, "..", "..", "results", "pre_freeze_v22_s2b1")
    for f in glob.glob(os.path.join(root, "*.csv")):
        try:
            rows = csv.DictReader(l for l in open(f) if not l.startswith("#"))
            for r in rows:
                st = (r.get("status") or "").strip()
                if st in ("ok", "FAILED", ""):
                    continue
                if r.get("recall") and r.get("qps_mean"):
                    bad.add((round(float(r["recall"]), 6),
                             round(float(r["qps_mean"]), 6)))
        except Exception:
            pass
    return bad


# Which n_lists each dataset's reported IVF-PQ front came from, filled in by
# load_baselines.  None means the archived sweep's value.  fig_build reads it
# so that the build bars count only the partition the frontier reports: the
# re-run at JHQ's nlist builds a different index, and on stella-trec24 that
# moved IVF-PQ's median build from 43.0 s to 55.7 s for a configuration the
# frontier does not use.
PQ_NLIST = {}
PQ_REPORTED = {}


def load_baselines():
    """cuVS fronts from the v47 sweep; unchanged since, and not re-run."""
    import json
    p = os.path.join(HERE, "..", "..", "report_adc2026", "v47", "fronts.json")
    j = json.load(open(p))
    bad = _quarantined()
    def keep(series):
        return [tuple(x) for x in series
                if (round(float(x[0]), 6), round(float(x[1]), 6)) not in bad]
    out = {}
    for ds, v in j.items():
        out[ds] = {"cagra": pareto(keep(v["series"].get("CAGRA fp32", []))),
                   "cagra8": pareto(keep(v["series"].get("CAGRA int8", []))),
                   "ivfpq": pareto(keep(v["series"].get("IVF-PQ", []))),
                   "N": v["N"], "d": v["d"]}
    # The default Stella IVF-PQ path failed, but this archived reduced-training
    # path completed. Include its successful search points in the same envelope.
    import csv
    extra = os.path.join(HERE, "..", "..", "results", "pre_freeze_v22_s2b1",
                         "fair_stella_ivfpq_f0.02.csv")
    with open(extra) as fh:
        rows = csv.DictReader(l for l in fh if not l.startswith("#"))
        pts = [(float(r["recall"]), float(r["qps_mean"])) for r in rows
               if r.get("status") == "ok"]
    out["stella"]["ivfpq"] = pareto(out["stella"]["ivfpq"] + pts)

    # IVF-PQ at the partition JHQ runs at, where that was measured.
    #
    # The archived sweep varies pq_dim over four code sizes and n_probes over
    # six depths and pins n_lists at one value a dataset -- 1024 on vogue-768
    # and both OpenAI sets, 2048 on arxiv-768 -- while JHQ's own nlist was
    # swept over four values and the frontier reports the winner.  One side was
    # tuned on a parameter the other was not.  At openai3-1536 that is 976
    # vectors a list against JHQ's 122.
    #
    # results/pq_nlist/ re-runs IVF-PQ at JHQ's value.  Each dataset is then
    # reported at whichever of its two partitions gives the better front, which
    # is how JHQ is reported and is the reason the two are not merged into one
    # envelope: JHQ's curve is a single nlist, so pooling partitions on one side
    # only would hand the baseline a choice its opponent does not get.
    pq_dir = os.path.join(HERE, "..", "..", "results", "pq_nlist")
    if os.path.isdir(pq_dir):
        alias = {"stella-trec24": "stella"}
        by_ds = collections.defaultdict(list)
        for f in sorted(os.listdir(pq_dir)):
            if not f.endswith(".csv"):
                continue
            with open(os.path.join(pq_dir, f)) as fh:
                rows = list(csv.DictReader(l for l in fh if not l.startswith("#")))
            pts = [(float(r["recall"]), float(r["qps_mean"])) for r in rows
                   if (r.get("status") or "").strip() == "ok"
                   and r.get("qps_mean") not in (None, "")]
            if not rows:
                continue
            ds = alias.get(rows[0].get("dataset"), rows[0].get("dataset"))
            if pts and ds in out:
                fr = pareto(pts)
                by_ds[ds].append(fr)
                try:
                    import json as _j
                    nl_seen = _j.loads(rows[0]["params"]).get("n_lists")
                except Exception:
                    nl_seen = None
                PQ_NLIST[(ds, id(fr))] = nl_seen
        nl_of = {}          # front id -> the n_lists that produced it
        for ds, fronts in by_ds.items():
            # "Better front" is area under the curve over the recalls both
            # partitions reach, so one extra high-recall point cannot win it.
            def score(fr, lo, hi):
                xs = [lo + (hi - lo) * i / 20.0 for i in range(21)]
                vs = [interp(fr, x) for x in xs]
                return sum(v for v in vs if v) if any(vs) else 0.0
            cands = fronts + [out[ds]["ivfpq"]]
            cands = [c for c in cands if c]
            lo = max(min(r for r, _ in c) for c in cands)
            hi = min(max(r for r, _ in c) for c in cands)
            if hi > lo:
                best = max(cands, key=lambda c: score(c, lo, hi))
                out[ds]["ivfpq"] = best
                # None means the archived sweep's own nlist, whatever it was.
                PQ_REPORTED[ds] = PQ_NLIST.get((ds, id(best)))

    # fronts.json rounded Vogue's N; use the actual loader/ID metadata.
    out["vogue-768"]["N"] = 932328
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


def save(fig, name, target_in=None):
    """Write out/<name>.pdf and .png at exactly `target_in` inches wide.

    savefig(bbox_inches="tight") crops to the drawn content, which is wider
    than figsize whenever a label overhangs the axes -- by up to 12% here. A
    figure wider than \textwidth is then shrunk by \includegraphics, and the
    point sizes on the page stop being the point sizes set in rcParams. So
    measure what tight actually produced and correct the canvas once. Font
    sizes are in points and do not move when the canvas does, which is the
    whole reason this is the right correction.
    """
    os.makedirs(OUT, exist_ok=True)
    target = target_in or WIDE
    for _ in range(3):
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        pad = plt.rcParams["savefig.pad_inches"]
        got = bb.width + 2 * pad
        if abs(got - target) < 0.01:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * target / got, h, forward=True)
        fig.canvas.draw()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  wrote out/{name}.pdf and .png  ({got:.2f} in wide)")
