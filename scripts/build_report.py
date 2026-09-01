#!/usr/bin/env python3
"""Generate the results report from results/final/p0_*.csv.

The previous report was written by hand and one of its tables carried a row
copied from the wrong dataset. Everything data-dependent here is computed from
harness rows, each of which records the commit, parameters and run count that
produced it; the prose is the only hand-written part.

Usage: build_report.py [--out report.html]
"""
import argparse, base64, csv, glob, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, "results", "final")

DATASETS = [("vogue", "Vogue-768", 768, 932328),
            ("arxiv-768", "arXiv-768", 768, 2253000),
            ("openai3-1536", "OpenAI3-1536", 1536, 999000),
            ("openai3-3072", "OpenAI3-3072", 3072, 999000),
            ("bge-m3", "BGE-M3", 1024, 10091524),
            ("stella-trec24", "Stella-TREC24", 1024, 17776615)]

SERIES = [("JHQ Br=4", "p0_{ds}*jhq*Br4*.csv"),
          ("JHQ Br=8", "p0_{ds}*jhq*Br8*.csv"),
          ("JQ (no residual)", "p0_{ds}*_jq.csv"),
          ("cuVS IVF-PQ", "p0_{ds}*ivfpq.csv"),
          ("CAGRA fp32", "p0_{ds}*_cagra.csv"),
          ("CAGRA int8", "p0_{ds}*cagra_int8.csv")]


def rows_for(pattern):
    out, meta = [], {}
    for path in sorted(glob.glob(os.path.join(D, pattern))):
        with open(path) as fh:
            first = fh.readline()
            if first.startswith("#"):
                try: meta = json.loads(first[1:])
                except Exception: pass
                body = fh.readlines()
            else:
                body = [first] + fh.readlines()
        for r in csv.DictReader(body):
            try:
                r["_r"], r["_q"] = float(r["recall"]), float(r["qps_mean"])
                r["_s"] = float(r["qps_std"] or 0)
            except ValueError:
                r["_r"] = r["_q"] = r["_s"] = None
            out.append(r)
    return out, meta


def best_at(rows, target):
    c = [r for r in rows if r["status"] == "ok" and r["_r"] is not None and r["_r"] >= target]
    return max(c, key=lambda r: r["_q"]) if c else None


def fmt(r):
    return f"{r['_q']:,.0f} <span class=\"at\">@{r['_r']:.4f}</span>" if r else \
           '<span class="nr">not reached</span>'


def collect():
    """{dataset: {series: rows}} plus the environment the rows were produced in."""
    data, meta = {}, {}
    for ds, _, _, _ in DATASETS:
        per = {}
        for label, pat in SERIES:
            rs, m = rows_for(pat.format(ds=ds))
            if rs:
                per[label] = rs
                meta = meta or m
        if per:
            data[ds] = per
    return data, meta


def failures(data):
    out = []
    for ds, per in data.items():
        for label, rs in per.items():
            for r in rs:
                if r["status"] != "ok":
                    note = ""
                    try:
                        f = json.loads(r["failures"] or "[]")
                        note = " ; ".join(x for pair in f for x in pair[1])
                    except Exception:
                        note = r["failures"]
                    out.append((ds, label, r["variant"], note))
    return out


def img_tag(name, cap):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        return ""
    b = base64.b64encode(open(p, "rb").read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b}" alt="{cap}">'
            f'<figcaption>{cap}</figcaption></figure>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(D, "report.html"))
    a = ap.parse_args()

    data, meta = collect()
    if not data:
        sys.exit(f"no p0_*.csv under {D}")

    # ---- table: best QPS at each recall gate, per dataset and method ----
    body = []
    for ds, title, dim, N in DATASETS:
        if ds not in data:
            continue
        rows = []
        for label, _ in SERIES:
            rs = data[ds].get(label)
            if not rs:
                continue
            a95, a98, a99 = (best_at(rs, t) for t in (0.95, 0.98, 0.99))
            vram = next((r["vram_mib"] for r in rs if r["vram_mib"]), "")
            cls = ' class="jhq"' if label.startswith("JHQ") else ""
            rows.append(f"<tr{cls}><td class=\"t\">{label}</td>"
                        f"<td class=\"num\">{fmt(a95)}</td>"
                        f"<td class=\"num\">{fmt(a98)}</td>"
                        f"<td class=\"num\">{fmt(a99)}</td>"
                        f"<td class=\"num\">{vram or '&mdash;'}</td></tr>")
        body.append(
            f'<h3>{title} &middot; d={dim}, {N:,} vectors</h3>'
            '<div class="tw"><table><thead><tr><th>method</th>'
            '<th class="num">QPS at R&ge;0.95</th><th class="num">R&ge;0.98</th>'
            '<th class="num">R&ge;0.99</th><th class="num">VRAM MiB</th>'
            '</tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")

    fails = failures(data)
    fail_html = ""
    if fails:
        fr = "".join(f'<tr><td class="t">{d}</td><td class="t">{m}</td>'
                     f'<td class="t">{v}</td><td class="t">{n[:200]}</td></tr>'
                     for d, m, v, n in fails[:40])
        fail_html = ('<h3>Configurations that did not run</h3>'
                     '<p>Recorded rather than dropped, with the device memory read at '
                     'the moment of failure where the cause was capacity.</p>'
                     '<div class="tw"><table><thead><tr><th>dataset</th><th>method</th>'
                     '<th>config</th><th>reason</th></tr></thead><tbody>'
                     + fr + "</tbody></table></div>")

    meta_html = " &middot; ".join(
        f"{k} <code>{v}</code>" for k, v in sorted(meta.items())
        if k in ("commit", "gpu", "driver", "nvcc", "cuvs", "host_cores"))

    html = TEMPLATE.format(
        meta=meta_html or "environment not recorded",
        tables="\n".join(body),
        fronts=img_tag("figA_fronts.png",
                       "QPS against recall for every method on every dataset that finished. "
                       "Five runs per point; the harness records the spread with each."),
        trend=img_tag("figB_dimension_trend.png",
                      "CAGRA scores against stored vectors, so its distance work grows with d "
                      "while JHQ's is fixed at M subspaces. The ratio is measured, not modelled."),
        failures=fail_html,
    )
    with open(a.out, "w") as fh:
        fh.write(html)
    print(f"wrote {a.out} ({len(html)//1024} KB), "
          f"{sum(len(v) for v in data.values())} series, {len(fails)} failed configs")


TEMPLATE = """<title>JHQ on GPU: Measured</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>
:root{{--ground:#F4F6F8;--panel:#FFF;--ink:#141A21;--muted:#5C6773;--rule:#D6DAE0;
--rule-soft:#E6E9EE;--accent:#1F5F8B;--accent-soft:#E3EDF4;--no:#A2452E;
--sans:"IBM Plex Sans Condensed",system-ui,sans-serif;--serif:"Source Serif 4",Georgia,serif;
--mono:"IBM Plex Mono",ui-monospace,monospace;}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--ground:#11151A;
--panel:#171C23;--ink:#E2E7ED;--muted:#8894A2;--rule:#28303A;--rule-soft:#1E242C;
--accent:#5FA8D8;--accent-soft:#172833;--no:#D4795C;}}}}
:root[data-theme="dark"]{{--ground:#11151A;--panel:#171C23;--ink:#E2E7ED;--muted:#8894A2;
--rule:#28303A;--rule-soft:#1E242C;--accent:#5FA8D8;--accent-soft:#172833;--no:#D4795C;}}
*{{box-sizing:border-box}}
body{{background:var(--ground);color:var(--ink);font-family:var(--serif);
font-size:17px;line-height:1.62;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1120px;margin:0 auto;padding:0 28px 90px}}
h1,h2,h3,th,.label{{font-family:var(--sans)}}
h1{{font-size:clamp(2rem,4.4vw,3rem);font-weight:700;letter-spacing:-.015em;
line-height:1.07;margin:0 0 .5rem;text-wrap:balance}}
h2{{font-size:1.5rem;font-weight:700;margin:0 0 .2rem;text-wrap:balance}}
h3{{font-size:1.02rem;font-weight:600;margin:1.9rem 0 .5rem}}
p{{margin:0 0 1rem;max-width:68ch}}
.label{{font-size:.72rem;font-weight:600;letter-spacing:.13em;text-transform:uppercase;color:var(--muted)}}
header{{border-bottom:1px solid var(--rule);padding:54px 0 26px;margin-bottom:36px}}
.env{{font-size:.82rem;color:var(--muted);margin-top:16px}}
section{{border-top:1px solid var(--rule-soft);padding:34px 0}}
.tw{{overflow-x:auto;border:1px solid var(--rule);border-radius:2px;margin:14px 0 4px}}
table{{border-collapse:collapse;width:100%;background:var(--panel);font-size:.88rem}}
th{{font-size:.68rem;font-weight:600;letter-spacing:.09em;text-transform:uppercase;
color:var(--muted);text-align:left;padding:10px 13px;border-bottom:1px solid var(--rule);white-space:nowrap}}
td{{padding:8px 13px;border-bottom:1px solid var(--rule-soft);font-family:var(--mono);
font-size:.82rem;font-variant-numeric:tabular-nums;white-space:nowrap}}
td.t{{font-family:var(--serif);font-size:.88rem;white-space:normal}}
tr:last-child td{{border-bottom:none}}
tr.jhq td{{background:var(--accent-soft)}}
.num{{text-align:right}}
.at{{color:var(--muted);font-size:.92em}}
.nr{{color:var(--no)}}
figure{{margin:22px 0 6px}}
figure img{{width:100%;height:auto;display:block;border:1px solid var(--rule);background:var(--panel)}}
figcaption{{font-size:.85rem;color:var(--muted);margin-top:9px;max-width:74ch}}
code{{font-family:var(--mono);font-size:.85em;background:var(--rule-soft);padding:1px 5px;border-radius:2px}}
footer{{border-top:1px solid var(--rule);margin-top:44px;padding-top:20px;font-size:.84rem;color:var(--muted)}}
:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
</style>
<div class="wrap">
<header>
  <div class="label">Experiment report &middot; generated from the harness CSVs</div>
  <h1>JHQ on GPU: measured</h1>
  <p>Every number below is parsed from a benchmark row that carries its own
  commit, parameter set and run count. Each point is five runs after a warm-up;
  configurations that failed are listed rather than omitted.</p>
  <div class="env">{meta}</div>
</header>

<section>
  <h2>Throughput at each recall gate</h2>
  <p>Best QPS among the configurations of each method that reach the recall on
  the column. JHQ's two residual settings are separate rows because they are
  separate memory budgets, not one curve.</p>
  {tables}
  {failures}
</section>

<section>
  <h2>Fronts</h2>
  {fronts}
</section>

<section>
  <h2>Dimension</h2>
  {trend}
</section>

<footer>
  Recall@10 is set intersection against the true top-10. The official JHQ
  implementation's own recall function instead tests each returned id for
  membership in the whole ground-truth row, which is a looser measure and not
  comparable across datasets whose ground truth differs in width &mdash; 100
  columns for Vogue and arXiv, 20 for the rest. Both are recorded; the tables
  use the set-intersection form.
</footer>
</div>
"""

if __name__ == "__main__":
    main()
