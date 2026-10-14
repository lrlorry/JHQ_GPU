#!/usr/bin/env python3
"""COMP8800 poster E1: every raw record -> one table, frontiers, matched-recall ratios, figures.

    python3 analyze.py <RAW_RESULTS dir> [--figdir DIR] [--md OUT.md]

Reads demo_jhq_v69_batch's JSON (jhq_*.json, the scheduled pipeline G = -1 is JHQ-GPU v69; G = 0
is its v61-path reference, used only as a check), and the jsonl of bench_cuvs.py and
bench_rabitq_sweep.cu. Writes E1_points.csv (every measured point, protocol A and B),
E1_failures.csv, E1_matched.csv and the figures. Throughput is protocol A (host to host) unless a
column says B.

Matched recall, two definitions, never mixed:
  measured  QPS at recall >= r: the best measured point of a method whose recall is at least r.
            No interpolation; defined only if the method has such a point.
  interp    log-QPS interpolated linearly in recall between the two points of the method's Pareto
            frontier that bracket r; defined only inside the frontier's recall range (no
            extrapolation). A frontier point within 0.0005 of r counts as measured.
A ratio is reported only where both methods are defined at r.
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys

FAMILY_ORDER = ["JHQ-GPU v69", "CAGRA fp32", "CAGRA int8", "IVF-RaBitQ", "IVF-PQ", "IVF-PQ + refine",
                "GPU brute force"]
STYLE = {  # colour, marker, linestyle
    "JHQ-GPU v69": ("#1f4e79", "o", "-"),
    "CAGRA fp32": ("#2e8b57", "D", "-"),
    "CAGRA int8": ("#8fae2c", "^", "--"),
    "IVF-RaBitQ": ("#c0392b", "s", "-"),
    "IVF-PQ": ("#e67e22", "v", ":"),
    "IVF-PQ + refine": ("#b9770e", "P", ":"),
    "GPU brute force": ("#555555", "*", ""),
}
DS_TITLE = {"vogue-768": "Vogue-768 (0.93M x 768)", "arxiv-768": "arXiv-768 (2.25M x 768)",
            "openai3-1536": "OpenAI3-1536 (1.0M x 1536)", "openai3-3072": "OpenAI3-3072 (1.0M x 3072)"}


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def load(raw):
    points, failures, indexes = [], [], []
    # JHQ: one harness JSON per (dataset, nprobe); alphas and variants inside
    for f in sorted(glob.glob(os.path.join(raw, "jhq_*.json"))):
        try:
            j = json.load(open(f))
        except Exception as ex:
            failures.append({"dataset": os.path.basename(f), "method": "JHQ-GPU v69", "label": "parse",
                             "params": "", "error": f"unreadable JSON: {ex}"})
            continue
        ds = os.path.basename(f).split("_")[1]
        for run in j.get("runs", []):
            ref = {v["G"]: v for v in run["variants"]}
            for v in run["variants"]:
                if v["G"] != -1:
                    continue
                pts = {"dataset": ds, "family": "JHQ-GPU v69", "label": "jhq-v69",
                       "params": json.dumps({"nprobe": j["nprobe"], "alpha": run["alpha"], "ck": run["ck"],
                                             "nlist": j["nlist"], "M": j["M"], "Br": 8, "batch": run["batch"]}),
                       "recall": v["recall"], "qps_A": med(v["qps"]), "qps_A_min": min(v["qps"]),
                       "qps_A_max": max(v["qps"]), "qps_B": "",
                       "lat_med_ms": med(v["lat_med_ms"]), "lat_p95_ms": med(v["lat_p95_ms"]),
                       "rounds": len(v["qps"]), "calls_per_round": run["reps"] * max(1, run["nbatch"]),
                       "mem_used_mib": v["used_mib"], "index_mib": "", "invalid_ids": "", "dup_rows": "",
                       "deterministic": "", "flags": "", "source": os.path.basename(f)}
                g0 = ref.get(0)
                if g0 is not None and abs(g0["recall"] - v["recall"]) > 1e-9:
                    pts["flags"] = f"G=0 recall {g0['recall']:.6f} != G=-1"
                points.append(pts)
    for f in sorted(glob.glob(os.path.join(raw, "cuvs_*.jsonl")) + glob.glob(os.path.join(raw, "rabitq_*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                failures.append({"dataset": os.path.basename(f), "method": "?", "label": "parse",
                                 "params": "", "error": "unparsable line"})
                continue
            ds = r.get("dataset") or os.path.basename(f).split("_")[1].replace(".jsonl", "")
            if r["type"] == "index":
                indexes.append(dict(r, source=os.path.basename(f), dataset=ds))
            elif r["type"] == "failure":
                failures.append({"dataset": ds, "method": r.get("method"), "label": r.get("label", r.get("stage")),
                                 "params": json.dumps({k: r.get(k) for k in ("index", "search", "n_lists",
                                                       "bits_per_dim", "n_probes", "mode") if k in r}),
                                 "error": r.get("error", "")[:300]})
            elif r["type"] == "point":
                m = r["method"]
                fam = {"cagra": "CAGRA fp32", "cagra-int8": "CAGRA int8", "ivfpq": "IVF-PQ",
                       "ivfpq-refine": "IVF-PQ + refine", "bruteforce": "GPU brute force",
                       "ivf-rabitq": "IVF-RaBitQ"}[m]
                A, B = r["A"], r.get("B")
                flags = []
                if r.get("invalid_ids"):
                    flags.append(f"{r['invalid_ids']} invalid ids")
                if r.get("dup_rows"):
                    flags.append(f"{r['dup_rows']} rows repeat an id")
                if r.get("deterministic") is False:
                    flags.append("ids differ between calls")
                if r.get("b_matches_a") is False:
                    flags.append("protocol B ids differ from A")
                params = dict(r.get("index", {}))
                params.update(r.get("search") or {})
                for drop in ("type", "dataset", "method", "utc", "ctx_mib", "mem_used_after_build_mib"):
                    params.pop(drop, None)
                points.append({"dataset": ds, "family": fam, "label": r["label"], "params": json.dumps(params),
                               "recall": r["recall"], "qps_A": A["qps_median"], "qps_A_min": A["qps_min"],
                               "qps_A_max": A["qps_max"], "qps_B": B["qps_median"] if B else "",
                               "lat_med_ms": med(A["lat_med_ms_rounds"]), "lat_p95_ms": med(A["lat_p95_ms_rounds"]),
                               "rounds": r["rounds"], "calls_per_round": r["calls_per_round"],
                               "mem_used_mib": r.get("mem_used_mib", ""),
                               "index_mib": (r.get("index") or {}).get("index_mib", ""),
                               "invalid_ids": r.get("invalid_ids", ""), "dup_rows": r.get("dup_rows", ""),
                               "deterministic": r.get("deterministic", ""), "flags": "; ".join(flags),
                               "source": os.path.basename(f)})
    return points, failures, indexes


def valid(p):
    """A point that may enter a frontier: no id defects, sane numbers."""
    return (not p["flags"] or p["flags"].startswith("G=0")) and p["qps_A"] and p["qps_A"] == p["qps_A"]


def pareto(pts):
    """Points not dominated in (recall up, QPS up), ascending recall."""
    best, out = -1.0, []
    for p in sorted(pts, key=lambda p: (-p["recall"], -p["qps_A"])):
        if p["qps_A"] > best:
            out.append(p)
            best = p["qps_A"]
    return sorted(out, key=lambda p: p["recall"])


def at_recall_measured(pts, r):
    c = [p["qps_A"] for p in pts if p["recall"] >= r]
    return max(c) if c else None


def at_recall_interp(front, r):
    if not front or r < front[0]["recall"] - 5e-4 or r > front[-1]["recall"] + 5e-4:
        return None, None
    for p in front:
        if abs(p["recall"] - r) <= 5e-4:
            return p["qps_A"], "measured"
    for a, b in zip(front, front[1:]):
        if a["recall"] <= r <= b["recall"]:
            t = (r - a["recall"]) / (b["recall"] - a["recall"])
            return math.exp(math.log(a["qps_A"]) + t * (math.log(b["qps_A"]) - math.log(a["qps_A"]))), "interp"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--targets", default="0.90,0.92,0.94,0.95,0.96,0.97,0.98,0.985,0.99,0.995")
    a = ap.parse_args()
    figdir = a.figdir or os.path.join(os.path.dirname(os.path.abspath(a.raw)), "figures")
    os.makedirs(figdir, exist_ok=True)
    out_dir = os.path.dirname(os.path.abspath(a.raw))
    points, failures, indexes = load(a.raw)
    cols = ["dataset", "family", "label", "params", "recall", "qps_A", "qps_A_min", "qps_A_max", "qps_B",
            "lat_med_ms", "lat_p95_ms", "rounds", "calls_per_round", "mem_used_mib", "index_mib",
            "invalid_ids", "dup_rows", "deterministic", "flags", "source"]
    with open(os.path.join(out_dir, "E1_points.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for p in sorted(points, key=lambda p: (p["dataset"], FAMILY_ORDER.index(p["family"]), -p["recall"])):
            w.writerow(p)
    with open(os.path.join(out_dir, "E1_failures.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "method", "label", "params", "error"])
        w.writeheader()
        for f in failures:
            w.writerow(f)
    targets = [float(x) for x in a.targets.split(",")]
    md = []
    matched_rows = []
    datasets = sorted({p["dataset"] for p in points}, key=lambda d: list(DS_TITLE).index(d) if d in DS_TITLE else 9)
    for ds in datasets:
        fams = {}
        for p in points:
            if p["dataset"] == ds and valid(p):
                fams.setdefault(p["family"], []).append(p)
        fronts = {f: pareto(v) for f, v in fams.items()}
        md.append(f"\n## {DS_TITLE.get(ds, ds)}\n")
        md.append("| method | points | recall range (frontier) | best QPS | frontier points |")
        md.append("|---|---:|---|---:|---:|")
        for f in FAMILY_ORDER:
            if f in fronts:
                fr = fronts[f]
                md.append(f"| {f} | {len(fams[f])} | {fr[0]['recall']:.4f}-{fr[-1]['recall']:.4f} | "
                          f"{max(p['qps_A'] for p in fr):,.0f} | {len(fr)} |")
        md.append("\n### Frontier points (protocol A)\n")
        for f in FAMILY_ORDER:
            if f not in fronts:
                continue
            md.append(f"\n**{f}**\n\n| recall | QPS A (median of rounds, min-max) | QPS B | p50 / p95 ms | config |")
            md.append("|---:|---:|---:|---|---|")
            for p in fronts[f]:
                qb = f"{p['qps_B']:,.0f}" if p["qps_B"] != "" else "-"
                md.append(f"| {p['recall']:.4f} | {p['qps_A']:,.0f} ({p['qps_A_min']:,.0f}-{p['qps_A_max']:,.0f}) | "
                          f"{qb} | {p['lat_med_ms']:.3f} / {p['lat_p95_ms']:.3f} | `{p['label']}` {p['params']} |")
        md.append("\n### Matched recall: JHQ-GPU v69 throughput / other method (protocol A)\n")
        md.append("measured = best measured point at recall >= r (no interpolation); interp = log-linear between "
                  "bracketing frontier points; '-' = the method does not cover r.\n")
        others = [f for f in FAMILY_ORDER if f in fronts and f != "JHQ-GPU v69"]
        md.append("| r | JHQ QPS (measured / interp) | " + " | ".join(f"{o}: QPS, ratio" for o in others) + " |")
        md.append("|---:|---|" + "---|" * len(others))
        jf = fams.get("JHQ-GPU v69", [])
        jfr = fronts.get("JHQ-GPU v69", [])
        for r in targets:
            jm = at_recall_measured(jf, r)
            ji, jk = at_recall_interp(jfr, r)
            cells = []
            for o in others:
                om = at_recall_measured(fams[o], r)
                oi, ok = at_recall_interp(fronts[o], r)
                rm = (jm / om) if (jm and om) else None
                ri = (ji / oi) if (ji and oi) else None
                cells.append(f"{om:,.0f} / {oi:,.0f}, x{rm:.2f} / x{ri:.2f}" if (om and oi and rm and ri) else
                             (f"{om:,.0f}, x{rm:.2f} (meas.)" if (om and rm) else
                              (f"{om:,.0f} (JHQ -)" if om else "-")))
                matched_rows.append({"dataset": ds, "recall": r, "other": o,
                                     "jhq_measured": jm or "", "other_measured": om or "",
                                     "ratio_measured": round(rm, 4) if rm else "",
                                     "jhq_interp": round(ji, 1) if ji else "", "jhq_interp_kind": jk or "",
                                     "other_interp": round(oi, 1) if oi else "", "other_interp_kind": ok or "",
                                     "ratio_interp": round(ri, 4) if ri else ""})
            jtxt = (f"{jm:,.0f} / {ji:,.0f}" if (jm and ji) else (f"{jm:,.0f} / -" if jm else "-"))
            md.append(f"| {r} | {jtxt} | " + " | ".join(cells) + " |")
        plot(ds, fams, fronts, figdir)
    with open(os.path.join(out_dir, "E1_matched.csv"), "w", newline="") as fh:
        if matched_rows:
            w = csv.DictWriter(fh, fieldnames=list(matched_rows[0]))
            w.writeheader()
            for r in matched_rows:
                w.writerow(r)
    if failures:
        md.append("\n## Failures and anomalies (kept, never dropped)\n")
        md.append("| dataset | method | label | params | error |")
        md.append("|---|---|---|---|---|")
        for f in failures:
            md.append(f"| {f['dataset']} | {f['method']} | {f['label']} | {f['params'][:80]} | {f['error'][:160]} |")
    flagged = [p for p in points if p["flags"]]
    if flagged:
        md.append("\n## Points with flags\n")
        for p in flagged:
            md.append(f"- {p['dataset']} {p['family']} {p['params']}: {p['flags']}")
    text = "\n".join(md) + "\n"
    if a.md:
        open(a.md, "w").write(text)
    print(text)


def plot(ds, fams, fronts, figdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:
        print(f"(no matplotlib: {ex})", file=sys.stderr)
        return
    plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "axes.titlesize": 18, "legend.fontsize": 13,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    lo = 0.90
    for f in FAMILY_ORDER:
        if f not in fronts:
            continue
        c, mk, ls = STYLE[f]
        allp = [p for p in fams[f] if p["recall"] >= lo - 0.02]
        ax.scatter([p["recall"] for p in allp], [p["qps_A"] for p in allp], s=10, color=c, alpha=0.18, zorder=1)
        fr = [p for p in fronts[f] if p["recall"] >= lo - 0.02]
        if not fr:
            continue
        lw = 3.2 if f == "JHQ-GPU v69" else 2.0
        ax.plot([p["recall"] for p in fr], [p["qps_A"] for p in fr], ls if ls else "none", marker=mk, color=c,
                lw=lw, ms=8 if f != "GPU brute force" else 16, label=f, zorder=3 if f == "JHQ-GPU v69" else 2)
    ax.set_yscale("log")
    ax.set_xlim(lo, 1.0)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Queries per second (log scale)")
    ax.set_title(DS_TITLE.get(ds, ds))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower left", frameon=True)
    ax.text(0.99, 0.98, "RTX 5090 · batch 1000 · host-to-host timing", transform=ax.transAxes, ha="right",
            va="top", fontsize=11, color="#555555")
    fig.tight_layout()
    name = {"vogue-768": "vogue", "arxiv-768": "arxiv", "openai3-3072": "openai3072",
            "openai3-1536": "openai1536"}.get(ds, ds)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(figdir, f"fig_recall_qps_{name}.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
