#!/usr/bin/env python3
"""Draw the findings that only ever had tables.

The recall-vs-QPS fronts have a page; these four do not, and each is a result
the fronts cannot show. Regenerates report_adc2026/v47/findings.html from the
raw logs. Run from the repo root.
"""
import csv, glob, json, math, os, re, sys, collections, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, 'report_adc2026', 'v47', 'findings.html')
NICE = {'vogue':'Vogue-768','bge':'BGE-M3','stella':'Stella','vogue-768':'Vogue-768',
        'arxiv-768':'arXiv-768','bge-m3':'BGE-M3','openai3-1536':'OpenAI3-1536',
        'openai3-3072':'OpenAI3-3072'}

def load_diag():
    """One row per (dataset, nprobe). diag.log sweeps nprobe at alpha=100 and
    then alpha at nprobe=128, so the alpha=100 cell appears in both passes."""
    rows, seen = [], set()
    pat = re.compile(r'\s+(\S+)\s+nlist=\S+\s+np=(\S+)\s+a=(\S+)\s+recall=([\d.]+)\s+ivf=([\d.]+)'
                     r'\s+route_lost=([\d.]+)\s+rank_lost=([\d.]+)\s+cand=(\d+)\s+est=(\d+)')
    for l in open(os.path.join(ROOT, 'results/v43_diag/diag.log')):
        m = pat.match(l)
        if m:
            key = (m.group(1), int(m.group(2)), float(m.group(3)))
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(ds=m.group(1), np=int(m.group(2)), a=float(m.group(3)),
                             recall=float(m.group(4)), ivf=float(m.group(5)),
                             route=float(m.group(6)), rank=float(m.group(7)),
                             cand=int(m.group(8)), est=int(m.group(9))))
    return rows

def load_block():
    d, cur, sect = collections.defaultdict(list), None, None
    for l in open(os.path.join(ROOT, 'results/pending/b1024.log')):
        if '#####' in l and 'BLOCK=' in l:
            cur = int(re.search(r'BLOCK=(\d+)', l).group(1)); continue
        if re.match(r'^\d\d:\d\d:\d\d ---', l):
            sect = l.split('---')[1].strip().split(':')[0]; continue
        m = re.match(r'\s+demo_jhq_(\S+)\s+(\S+)\s+BLOCK=(\d+)\s+np=\S+\s+lat=([\d.]+)', l)
        if m and cur:
            d[(cur, sect, m.group(2), m.group(1))].append(float(m.group(4)))
    A = lambda *k: st.mean(d[k]) if d.get(k) else None
    out = []
    for ds in ('vogue', 'bge', 'stella'):
        for blk in (256, 1024):
            f = A(blk, 'scan split', ds, 'v42_full')
            c = A(blk, 'scan split', ds, 'v42_nocomp')
            n = A(blk, 'scan split', ds, 'v42_neither')
            if f:
                out.append(dict(ds=ds, block=blk, compact=f - c, lookup=c - n, rest=n, total=f))
    return out

def load_ntrain():
    """(dataset, nlist, n_train) -> [(recall, qps, cand)], from two sweeps."""
    g = collections.defaultdict(list)
    pat = re.compile(r'\s+(\S+)\s+M=\S+\s+nlist=(\S+)\s+nt=(\S+)\s+np=\S+\s+recall=([\d.]+)'
                     r'\s+qps=([\d.]+)\s+cand=(\d+)')
    for l in open(os.path.join(ROOT, 'results/front6/final3.log')):
        m = pat.match(l)
        if m and m.group(4):
            g[(m.group(1), int(m.group(2)), int(m.group(3)))].append(
                (float(m.group(4)), float(m.group(5)), int(m.group(6))))
    pat2 = re.compile(r'\s+demo_jhq_v46_diag\s+stella\s+nlist=16384\s+np=\S+\s+nt=(\S+)'
                      r'\s+recall=([\d.]+)\s+qps=([\d.]+)\s+cand=(\d+)')
    for l in open(os.path.join(ROOT, 'results/v46_sigma/v46.log')):
        m = pat2.match(l)
        if m:
            g[('stella', 16384, int(m.group(1)))].append(
                (float(m.group(2)), float(m.group(3)), int(m.group(4))))
    return g

def load_build():
    """Cold-cache train / add / VRAM. Only the first row of each group is a
    real build; the three after it read the cache the first one wrote."""
    seen, pat = {}, re.compile(
        r'\s+(\S+)\s+M=\S+\s+nlist=(\S+)\s+nt=(\S+)\s+np=\S+\s+recall=[\d.]+\s+qps=[\d.]+'
        r'\s+cand=\S+\s+ivf=\S+\s+train=([\d.]+)\s+add=([\d.]+)\s+vram=([\d.]+)')
    for l in open(os.path.join(ROOT, 'results/front6/front6.log')):
        m = pat.match(l)
        if not m:
            continue
        k = (m.group(1), int(m.group(2)), int(m.group(3)))
        if k not in seen:
            seen[k] = (float(m.group(4)) / 1000, float(m.group(5)) / 1000, float(m.group(6)))
    return seen


def load_build_baselines():
    """cuVS build time. Its CSVs put the whole build in train_ms and leave
    add_ms empty, which is why an earlier pass requiring both found nothing."""
    pats = {'vogue-768':'vogue','arxiv-768':'arxiv-768','bge-m3':'bge-m3',
            'stella':'stella-trec24','openai3-1536':'openai3-1536','openai3-3072':'openai3-3072'}
    out = {}
    for ds, pat in pats.items():
        got = {}
        for lab, keys in [('CAGRA fp32', ['cagra']),
                          ('CAGRA int8', ['cagra_int8', 'cagra_int8_hi']),
                          ('IVF-PQ',     ['ivfpq', 'ivf_pq'])]:
            vals = []
            for f in glob.glob(os.path.join(ROOT, 'results/pre_freeze_v22_s2b1', f'*{pat}*.csv')):
                if not re.match(r'(p0|sat)_.*?_(' + '|'.join(keys) + r')\.csv$', os.path.basename(f)):
                    continue
                for r in csv.DictReader(l for l in open(f) if not l.startswith('#')):
                    if r.get('status') == 'FAILED' or not r.get('train_ms'):
                        continue
                    vals.append(float(r['train_ms']) / 1000 + float(r.get('add_ms') or 0) / 1000)
            if vals:
                got[lab] = min(vals)
        out[ds] = got
    return out


def interp(front, r):
    f = sorted((a, b) for a, b, *_ in front)
    if not f or r < f[0][0] or r > f[-1][0]:
        return None
    for i in range(len(f) - 1):
        (r0, q0), (r1, q1) = f[i], f[i + 1]
        if r0 <= r <= r1:
            if r1 == r0:
                return max(q0, q1)
            t = (r - r0) / (r1 - r0)
            return math.exp(math.log(q0) + t * (math.log(q1) - math.log(q0)))
    return None

# ── charts ───────────────────────────────────────────────────────────────────
def bars_loss(rows):
    """Stacked routing / ranking loss against nprobe, one row per dataset."""
    W, H, L, T = 620, 190, 92, 14
    PW, PH = W - L - 78, H - T - 30
    out = []
    for ds in ('vogue', 'bge', 'stella'):
        rs = sorted([r for r in rows if r['ds'] == ds and r['a'] == 100.0], key=lambda x: x['np'])
        if not rs: continue
        top = max(r['route'] + r['rank'] for r in rs) * 1.12
        bw = PH / len(rs) * 0.62
        g = []
        for i, r in enumerate(rs):
            y = T + PH * (i + 0.5) / len(rs) - bw / 2
            wr = r['route'] / top * PW
            wk = r['rank'] / top * PW
            g.append(f'<rect class="b-route" x="{L}" y="{y:.1f}" width="{max(wr,0.6):.1f}" height="{bw:.1f}"/>')
            g.append(f'<rect class="b-rank" x="{L+wr:.1f}" y="{y:.1f}" width="{max(wk,0.6):.1f}" height="{bw:.1f}"/>')
            g.append(f'<text class="blab" x="{L-8}" y="{y+bw/2+3.5:.1f}">nprobe {r["np"]}</text>')
            g.append(f'<text class="bval" x="{L+wr+wk+7:.1f}" y="{y+bw/2+3.5:.1f}">'
                     f'{r["route"]*100:.2f} + {r["rank"]*100:.2f}</text>')
        for f in (0, .25, .5, .75, 1):
            x = L + PW * f
            g.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{T+PH}"/>')
            g.append(f'<text class="tick tx" x="{x:.1f}" y="{T+PH+14}">{top*f*100:.0f}%</text>')
        out.append(f'<figure class="fig"><figcaption><h3>{NICE[ds]}</h3></figcaption>'
                   f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{NICE[ds]} recall loss">'
                   f'{"".join(g)}</svg></figure>')
    return '\n'.join(out)

def bars_cand(rows):
    """Measured candidates over the N*nprobe/nlist estimate."""
    W, H, L, T = 620, 210, 92, 14
    PW, PH = W - L - 66, H - T - 30
    rs = [r for r in rows if r['a'] == 100.0]
    top = max(r['cand'] / r['est'] for r in rs) * 1.1
    g, i = [], 0
    for ds in ('vogue', 'bge', 'stella'):
        for r in sorted([x for x in rs if x['ds'] == ds], key=lambda x: x['np']):
            y = T + PH * (i + 0.5) / len(rs) - (PH / len(rs) * 0.6) / 2
            bw = PH / len(rs) * 0.6
            w = r['cand'] / r['est'] / top * PW
            g.append(f'<rect class="b-cand" x="{L}" y="{y:.1f}" width="{w:.1f}" height="{bw:.1f}"/>')
            g.append(f'<text class="blab" x="{L-8}" y="{y+bw/2+3.3:.1f}">{NICE[ds]} np={r["np"]}</text>')
            g.append(f'<text class="bval" x="{L+w+7:.1f}" y="{y+bw/2+3.3:.1f}">{r["cand"]/r["est"]:.2f}&#215;</text>')
            i += 1
    x1 = L + 1.0 / top * PW
    g.append(f'<line class="ref" x1="{x1:.1f}" x2="{x1:.1f}" y1="{T-4}" y2="{T+PH+2}"/>')
    g.append(f'<text class="reflab" x="{x1+5:.1f}" y="{T+PH+14}">the estimate</text>')
    return (f'<figure class="fig wide"><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="candidates over estimate">{"".join(g)}</svg></figure>')

def bars_block(blk):
    """Scan time split into compaction / lookup / gather, at both block sizes."""
    W, H, L, T = 620, 230, 108, 14
    PW, PH = W - L - 74, H - T - 30
    top = max(b['total'] for b in blk) * 1.06
    g, i, n = [], 0, len(blk)
    for ds in ('vogue', 'bge', 'stella'):
        for b in [x for x in blk if x['ds'] == ds]:
            bw = PH / n * 0.58
            y = T + PH * (i + 0.5) / n - bw / 2
            x = L
            for key, cls in (('compact', 'b-comp'), ('lookup', 'b-look'), ('rest', 'b-rest')):
                w = b[key] / top * PW
                g.append(f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{max(w,0.6):.1f}" height="{bw:.1f}"/>')
                x += w
            g.append(f'<text class="blab" x="{L-8}" y="{y+bw/2+3.3:.1f}">{NICE[ds]} BLOCK={b["block"]}</text>')
            g.append(f'<text class="bval" x="{x+7:.1f}" y="{y+bw/2+3.3:.1f}">{b["total"]:.0f} ms</text>')
            i += 1
    for f in (0, .25, .5, .75, 1):
        xx = L + PW * f
        g.append(f'<line class="grid" x1="{xx:.1f}" x2="{xx:.1f}" y1="{T}" y2="{T+PH}"/>')
        g.append(f'<text class="tick tx" x="{xx:.1f}" y="{T+PH+14}">{top*f:.0f}</text>')
    return (f'<figure class="fig wide"><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="scan decomposition">{"".join(g)}</svg></figure>')

def plot_ntrain(g):
    """Gain from a larger training set against points per centroid, log x."""
    CASES = [('vogue-768', 4096, 100000, 640000, 1000000),
             ('openai3-1536', 4096, 100000, 640000, 999000),
             ('openai3-3072', 4096, 100000, 640000, 999000),
             ('arxiv-768', 8192, 100000, 640000, 2253000),
             ('bge-m3', 16384, 100000, 640000, 10091524),
             ('stella', 16384, 100000, 1300000, 17776615)]
    pts = []
    for ds, nl, a, b, N in CASES:
        A, Bf = g.get((ds, nl, a)), g.get((ds, nl, b))
        if not A or not Bf: continue
        lo = max(min(r for r, *_ in A), min(r for r, *_ in Bf))
        hi = min(max(r for r, *_ in A), max(r for r, *_ in Bf))
        vals = [interp(Bf, lo + (hi - lo) * f) / interp(A, lo + (hi - lo) * f)
                for f in (0.0, 0.5, 1.0)]
        pts.append((ds, a / nl, st.median(vals), min(vals), max(vals)))
    W, H, L, T = 620, 240, 62, 18
    PW, PH = W - L - 118, H - T - 34
    xs = [math.log10(p[1]) for p in pts]
    x0, x1 = min(xs) - 0.12, max(xs) + 0.12
    ys = [p[3] for p in pts] + [p[4] for p in pts] + [1.0]
    y0, y1 = min(ys) * 0.94, max(ys) * 1.05
    X = lambda v: L + (math.log10(v) - x0) / (x1 - x0) * PW
    Y = lambda v: T + (1 - (v - y0) / (y1 - y0)) * PH
    g2 = [f'<rect class="frame" x="{L}" y="{T}" width="{PW}" height="{PH}"/>']
    g2.append(f'<line class="ref" x1="{L}" x2="{L+PW}" y1="{Y(1.0):.1f}" y2="{Y(1.0):.1f}"/>')
    g2.append(f'<text class="reflab" x="{L+4}" y="{Y(1.0)-5:.1f}">no change</text>')
    for v in (0.6, 0.8, 1.0, 1.4, 1.8, 2.2):
        if y0 <= v <= y1:
            g2.append(f'<text class="tick ty" x="{L-7}" y="{Y(v)+3.4:.1f}">{v:.1f}&#215;</text>')
            if v != 1.0:
                g2.append(f'<line class="grid" x1="{L}" x2="{L+PW}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
    for v in (3, 6, 12, 24):
        if x0 <= math.log10(v) <= x1:
            g2.append(f'<line class="grid" y1="{T}" y2="{T+PH}" x1="{X(v):.1f}" x2="{X(v):.1f}"/>')
            g2.append(f'<text class="tick tx" x="{X(v):.1f}" y="{T+PH+14}">{v}</text>')
    for ds, ppc, med, lo_, hi_ in sorted(pts, key=lambda p: p[1]):
        g2.append(f'<line class="rng" x1="{X(ppc):.1f}" x2="{X(ppc):.1f}" '
                  f'y1="{Y(lo_):.1f}" y2="{Y(hi_):.1f}"/>')
        cls = 'p-neg' if med < 1 else 'p-pos'
        g2.append(f'<circle class="{cls}" cx="{X(ppc):.1f}" cy="{Y(med):.1f}" r="4.2"/>')
        g2.append(f'<text class="ptlab" x="{X(ppc)+8:.1f}" y="{Y(med)+3.5:.1f}">{NICE[ds]}</text>')
    g2.append(f'<text class="axl" x="{L+PW//2}" y="{H-4}">training points per centroid at 100K (log)</text>')
    g2.append(f'<text class="axl" transform="translate(14,{T+PH//2}) rotate(-90)" x="0" y="0">QPS ratio</text>')
    return (f'<figure class="fig wide"><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="training gain against points per centroid">{"".join(g2)}</svg></figure>')

def bars_build(build, base):
    """Cold-cache build time: JHQ at both configurations, then the baselines."""
    order = ['vogue-768', 'arxiv-768', 'openai3-1536', 'openai3-3072', 'bge-m3', 'stella']
    W, L, T = 640, 132, 14
    PW = W - L - 104
    rows = []
    for ds in order:
        ks = sorted([k for k in build if k[0] == ds], key=lambda x: x[2])
        if len(ks) != 2:
            continue
        rows.append((ds, [('JHQ, published cfg', build[ks[0]][0], build[ks[0]][1], None),
                          ('JHQ, this work',     build[ks[1]][0], build[ks[1]][1], None)] +
                         [(lab, None, None, t) for lab, t in
                          sorted(base.get(ds, {}).items(), key=lambda x: x[1])]))
    n = sum(len(r[1]) for r in rows)
    PH = n * 17 + len(rows) * 9
    H = PH + T + 30
    top = max(max((v[3] or (v[1] + v[2])) for v in r[1]) for r in rows) * 1.04
    g, y = [], T
    for ds, entries in rows:
        for lab, tr, ad, bt in entries:
            bw = 12
            x = L
            if bt is None:
                for val, cls in ((tr, 'b-train'), (ad, 'b-add')):
                    w = val / top * PW
                    g.append(f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" '
                             f'width="{max(w,0.7):.1f}" height="{bw}"/>')
                    x += w
                tot = tr + ad
            else:
                w = bt / top * PW
                g.append(f'<rect class="b-base" x="{x:.1f}" y="{y:.1f}" '
                         f'width="{max(w,0.7):.1f}" height="{bw}"/>')
                x += w
                tot = bt
            jhq = entries[1][1] + entries[1][2]
            extra = f' &#183; {tot/jhq:.1f}&#215;' if bt is not None else ''
            cls = 'blab lead' if lab == 'JHQ, this work' else 'blab'
            g.append(f'<text class="{cls}" x="{L-8}" y="{y+bw-2.5:.1f}">{lab}</text>')
            g.append(f'<text class="bval" x="{x+7:.1f}" y="{y+bw-2.5:.1f}">{tot:.2f}s{extra}</text>')
            y += 17
        g.append(f'<text class="dslab" x="{L-8}" y="{y+2:.1f}">{NICE[ds]}</text>')
        y += 9
    for f in (0, .25, .5, .75, 1):
        xx = L + PW * f
        g.append(f'<line class="grid" x1="{xx:.1f}" x2="{xx:.1f}" y1="{T-3}" y2="{T+PH-6}"/>')
        g.append(f'<text class="tick tx" x="{xx:.1f}" y="{H-12}">{top*f:.0f}s</text>')
    return (f'<figure class="fig wide"><svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="build cost against the baselines">{"".join(g)}</svg></figure>')


CSS = open(os.path.join(ROOT, 'scripts', '_findings.css')).read()

def main():
    diag, blk, ntr = load_diag(), load_block(), load_ntrain()
    bld, bbase = load_build(), load_build_baselines()
    html = CSS + f'''
<div class="wrap">
<header>
  <p class="eyebrow">JHQ on GPU &middot; 8 September 2026 &middot; RTX 5090</p>
  <h1>Five results the fronts cannot show</h1>
  <p class="lede">Recall&ndash;throughput curves say where the method lands. These say why,
    and each one changed what to work on next.</p>
</header>
<hr>
<section>
  <h2>Recall is lost by routing, not by ranking</h2>
  <p>Split of the missing recall at alpha=100. <b class="k-route">Routing</b> is the share of
    the true top-10 whose IVF list was never opened; <b class="k-rank">ranking</b> is what
    survived routing and was then lost by the primary filter or the refinement.</p>
  <div class="figs">{bars_loss(diag)}</div>
  <p class="pull">The ranking loss is <b>0.13% to 0.51%</b> across three datasets and a
    32&times; range of nprobe. Everything downstream of coarse routing &mdash; the equation-4
    primary code, the ADC scan, the top-ck selection, the residual refinement &mdash; together
    loses at most half a point of what routing hands it. <b>No work on the scan can raise
    recall.</b></p>
</section>
<section>
  <h2>The scan reads up to 2.2&times; what was reported</h2>
  <p>Measured candidates over the <code>N&#215;nprobe/nlist</code> estimate every table in this
    project quoted before <code>JHQ_DIAG</code> existed.</p>
  {bars_cand(diag)}
  <p>Lists are not equal length, and the lists a query probes are not a random sample: a query
    lands near a dense region and the dense region's lists are the long ones. Every efficiency
    comparison that used the estimate flattered JHQ, worst on BGE-M3 &mdash; which is also
    where the gap to a graph index is widest.</p>
</section>
<section>
  <h2>The scan's parts do not scale together</h2>
  <p>Scan latency split by compiling parts out, at the default block size and at the one the
    fronts run. <b class="k-comp">Compaction</b>, <b class="k-look">lookup</b>,
    <b class="k-rest">gather and the rest</b>.</p>
  {bars_block(blk)}
  <p class="pull">The lookup was recorded as the largest single component &mdash; 42&ndash;52%
    at BLOCK=256. At 1024 it is <b>14&ndash;15%</b>, and it is the part that responds most to
    more warps. Compaction gets <b>more expensive in absolute time</b> on the two large sets.
    Every diagnostic taken at the default block size had to be re-read.</p>
</section>
<section>
  <h2>The training set pays below 20 points a centroid and costs above it</h2>
  <p>Matched-recall QPS ratio from raising <code>JHQ_N_TRAIN</code> at fixed nlist. The bar is
    the range over the front; the dot is the median.</p>
  {plot_ntrain(ntr)}
  <p class="pull">Below about ten points a centroid, k-means is degenerate &mdash; most
    centroids never leave initialisation and the partition is wildly uneven &mdash; and
    repairing it cuts Stella's candidates 35% <em>at higher recall</em>. Above twenty the
    quantizer is already fine and more training only sharpens its density-following, so Vogue's
    candidates <b>rise 95%</b>. Same knob, opposite directions.</p>
</section>
<section>
  <h2>Building the bigger index costs five seconds</h2>
  <p>The objection this project raised against itself was that throughput bought with a larger
    index is not free: <code>nlist</code> doubled or quadrupled, and the training set grew up
    to thirteen-fold. Cold caches, so JHQ's <b class="k-train">training</b> and
    <b class="k-add">add</b> are real &mdash; every build timing recorded before this was a
    cache hit reporting 35&nbsp;ms.</p>
  {bars_build(bld, bbase)}
  <p class="pull">The larger index costs <b>five seconds</b> on Stella &mdash; 6.03&nbsp;s to
    11.12&nbsp;s on 17.8M vectors, for 2.3&times; the query throughput. And against the
    baselines it is not close: <b>JHQ builds 5.3&times; to 10.3&times; faster than CAGRA</b>
    and 2.7&times; to 6.1&times; faster than IVF-PQ, on every dataset. Stella's int8 graph
    takes 73&nbsp;s where JHQ takes 11.</p>
  <p>k-means over <code>n_train &#215; nlist &#215; d</code> is a GEMM this card finishes in
    seconds, and <code>add()</code> &mdash; which encodes all N vectors and does not change
    &mdash; dominates JHQ's build in both rows. A graph has to be searched into existence
    instead, which is why it does not amortise the same way.</p>
  <p style="font-size:14px;color:var(--muted)">cuVS reports its whole build in
    <code>train_ms</code> and leaves <code>add_ms</code> empty, so the baseline bars are one
    segment. Memory is compared properly by bytes per vector in the report, not by these runs'
    VRAM figures, which are measured differently on each side.</p>
</section>
<footer>
  Regenerated by <code>scripts/make_findings.py</code> from
  results/v43_diag/, results/pending/b1024.log, results/front6/ and results/v46_sigma/.<br>
  Fronts and the full report: report_adc2026/v47/
</footer>
</div>'''
    open(OUT, 'w').write(html)
    print(f'  wrote {OUT} ({len(html):,} B)')

if __name__ == '__main__':
    main()
