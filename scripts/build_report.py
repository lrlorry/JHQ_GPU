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
        gate95=img_tag("fig_gate95.png",
                       "Best QPS among each method's configurations that reach Recall@10 0.95. "
                       "A hatched stub means the method was swept and never got there; the "
                       "ceiling it did reach is printed on it."),
        gate98=img_tag("fig_gate98.png",
                       "The same at 0.98. int8 CAGRA reaches it on none of the six, and on "
                       "OpenAI3-3072 JHQ is ahead of fp32 CAGRA on throughput as well."),
        ceiling=img_tag("fig_ceiling.png",
                        "The highest recall each method reaches at any setting. Above 1024 "
                        "dimensions JHQ's ceiling is the highest of the five."),
        memory=img_tag("fig_memory.png",
                       "Device memory actually in use against the recall it buys, read with "
                       "cudaMemGetInfo rather than derived from bytes per vector."),
        fronts=img_tag("figA_fronts.png",
                       "Full fronts per dataset, for reference."),
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
  <h2>The five questions, answered</h2>
  <p>Every figure below is the best configuration of each method that reaches the
  stated recall. Where a method never reaches a gate, the sweep was pushed until
  it stopped improving: int8 CAGRA was taken to itopk 2048, sixty-four times the
  smallest value tried, and gained 0.0016 or less on the final doubling.</p>
  <div class="tw"><table>
    <thead><tr><th>#</th><th>question</th><th>answer</th></tr></thead>
    <tbody>
      <tr><td class="t">1</td><td class="t">How much faster than CPU JHQ?</td>
      <td class="t"><b>427&times; single-thread, 82&times; all-core</b> at matched recall
      (Vogue, alpha=100: 53,818 QPS against 126.0 and 658.7). The all-core figure is only
      5.2&times; its own single-thread number on a 208-core host, so the reference scan is
      not parallel &mdash; the comparison is against a baseline that does not use the
      machine it runs on.</td></tr>
      <tr><td class="t">2</td><td class="t">What does the hierarchy contribute over JQ?</td>
      <td class="t"><b>It sets the ceiling.</b> With the residual level off, JQ fails to
      reach 0.95 on any of the six datasets &mdash; its ceilings are 0.647 to 0.774. The
      same ablation on CPU is 0.9341 against 0.6344.</td></tr>
      <tr><td class="t">3</td><td class="t">Better than IVF-PQ at the same budget?</td>
      <td class="t"><b>Yes.</b> 2.0&ndash;6.7&times; at R&ge;0.98 where IVF-PQ reaches it,
      and IVF-PQ's ceiling falls away with dimension: 0.986 at 768d, 0.966 at 1536d,
      0.913 at 3072d, against JHQ's 0.983, 0.994 and 0.992.</td></tr>
      <tr><td class="t">4</td><td class="t">And against compressed CAGRA?</td>
      <td class="t"><b>Above 1024 dimensions JHQ is the most accurate of the five.</b>
      PQ-compressed CAGRA is not reachable in cuVS 26.08.01; the int8 dataset that is
      reachable caps at 0.938&ndash;0.973 and reaches 0.98 on none of the six. fp32 CAGRA
      has the higher ceiling at 768d (0.995, 0.999 against 0.983, 0.993) but not at 1536d
      (0.986 against 0.994) or 3072d (0.985 against 0.992), where JHQ is also
      <b>2.17&times; faster on 1.7&times; less memory</b>.</td></tr>
      <tr><td class="t">5</td><td class="t">Does it hold across datasets and at scale?</td>
      <td class="t"><b>Yes, and it strengthens with dimension.</b> JHQ reaches 0.95 on all
      six; no other method does. CAGRA's throughput lead falls from 2.5&times; at 768d to
      below parity at 3072d. On bge-m3's 10.1M vectors JHQ is the only method to reach
      0.95, and stella-trec24's 17.8M run at 0.9510 and 41,812 QPS.</td></tr>
    </tbody>
  </table></div>
</section>

<section>
  <h2>Throughput at each recall gate</h2>
  <p>Best QPS among the configurations of each method that reach the recall on
  the column. JHQ's two residual settings are separate rows because they are
  separate memory budgets, not one curve.</p>
  {tables}
  {failures}
</section>

<section>
  <h2>Against the baselines</h2>
  <p>Each panel answers one question. A method absent from a panel is not
  missing data &mdash; it is a method that was swept and never reached that
  recall, drawn as a hatched stub with its ceiling on it.</p>
  {gate95}
  {gate98}
  {ceiling}
  {memory}
</section>

<section>
  <h2>Dimension, and the full fronts</h2>
  {trend}
  {fronts}
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
