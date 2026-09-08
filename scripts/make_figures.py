#!/usr/bin/env python3
"""Rebuild the v47 report and the six-dataset figure from results/.

Everything the two pages show is derived here, so they can be regenerated
after a new measurement rather than edited by hand. Run from the repo root:

    python3 scripts/make_figures.py

Writes report_adc2026/v47/{frontier.html,fronts_six.html,fronts.json}.

Inputs, all raw logs committed under results/:
  front6/front6.log          six datasets, two configurations, nprobe 8-256
  front6/front6_top.log      the new configuration past 256
  front6/front6_top2.log     the published configuration past 256
  front6/nlist6.log          four nlist per dataset, nprobe 8-1024
  front6/final3.log          denser nprobe, and n_train at fixed nlist
  front6/br4.log             the same at Br=4
  frontier_v39/f_*_Br8.csv   the published fronts
  pre_freeze_v22_s2b1/*.csv  cuVS CAGRA and IVF-PQ

A JHQ front is the Pareto envelope over every nlist and nprobe measured for
that dataset at that Br. Rows where nprobe exceeds nlist are dropped: past
that the probe list repeats and the row is the same measurement at a higher
price.
"""
import csv, glob, json, math, os, re, sys, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, 'report_adc2026', 'v47')

DS = {   # name: (d, M, N, glob pattern for the cuVS csvs, published csv)
 'vogue-768':    (768,  96,  1000000,  'vogue',        'f_vogue-768_M96Br8.csv'),
 'arxiv-768':    (768,  96,  2253000,  'arxiv-768',    'f_arxiv-768_M96Br8.csv'),
 'bge-m3':       (1024, 128, 10091524, 'bge-m3',       'f_bge-m3_M128Br8.csv'),
 'stella':       (1024, 128, 17776615, 'stella-trec24','f_stella-trec24_M128Br8.csv'),
 'openai3-1536': (1536, 192, 999000,   'openai3-1536', 'f_openai3-1536_M192Br8.csv'),
 'openai3-3072': (3072, 384, 999000,   'openai3-3072', 'f_openai3-3072_M384Br8.csv'),
}
NICE = {'vogue-768':'Vogue-768','arxiv-768':'arXiv-768','bge-m3':'BGE-M3',
        'stella':'Stella-TREC24','openai3-1536':'OpenAI3-1536','openai3-3072':'OpenAI3-3072'}
LOGS = [('front6/front6.log',8),('front6/front6_top.log',8),('front6/front6_top2.log',8),
        ('front6/nlist6.log',8),('front6/final3.log',8),('front6/br4.log',None)]
SER = ['JHQ Br=8 (this work)','JHQ Br=8 (published)','JHQ Br=4 (this work)',
       'CAGRA fp32','CAGRA int8','IVF-PQ']
CLS = {'JHQ Br=8 (this work)':'s-jhq','JHQ Br=8 (published)':'s-pub',
       'JHQ Br=4 (this work)':'s-jhq4','CAGRA fp32':'s-c32','CAGRA int8':'s-c8','IVF-PQ':'s-ivf'}
ROW = re.compile(r'\s+(\S+?)(-old)?\s+M=\S+\s+(?:Br=(\S+)\s+)?nlist=(\S+)\s+nt=(\S+)'
                 r'\s+np=(\S+)\s+recall=([\d.]+)\s+qps=([\d.]+)')

def pareto(pts):
    """Highest QPS per recall, then drop anything a higher-recall point beats."""
    best = {}
    for r, q, *_ in pts:
        best[r] = max(best.get(r, 0), q)
    out = []
    for r, q in sorted(best.items(), reverse=True):
        if not out or q > out[-1][1]:
            out.append((r, q))
    return sorted(out)

def interp(front, r):
    f = sorted(front)
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

def cuvs(pattern, keys):
    pts, oom = [], False
    for f in glob.glob(os.path.join(ROOT, 'results/pre_freeze_v22_s2b1', f'*{pattern}*.csv')):
        if not re.match(r'(p0|sat)_.*?_(' + '|'.join(keys) + r')\.csv$', os.path.basename(f)):
            continue
        for r in csv.DictReader(l for l in open(f) if not l.startswith('#')):
            if r.get('status') == 'FAILED':
                if 'out_of_memory' in (r.get('failures') or '') or 'bad_alloc' in (r.get('failures') or ''):
                    oom = True
            elif r.get('recall') and r.get('qps_mean'):
                pts.append((float(r['recall']), float(r['qps_mean'])))
    return pts, oom

def collect():
    raw = collections.defaultdict(list)
    for rel, defbr in LOGS:
        path = os.path.join(ROOT, 'results', rel)
        if not os.path.exists(path):
            print(f'  missing {rel}, skipped', file=sys.stderr); continue
        for line in open(path):
            m = ROW.match(line)
            if not m or not m.group(7) or m.group(1) not in DS:
                continue
            br = int(m.group(3)) if m.group(3) else defbr
            nlist, nprobe = int(m.group(4)), int(m.group(6))
            if nprobe > nlist:
                continue
            raw[(m.group(1), br)].append((float(m.group(7)), float(m.group(8)), nlist))
    data = {}
    for name, (d, M, N, pat, pubf) in DS.items():
        S = {'JHQ Br=8 (this work)': pareto(raw[(name, 8)])}
        S['JHQ Br=8 (published)'] = pareto(
            [(float(r['recall']), float(r['qps_mean'])) for r in
             csv.DictReader(l for l in open(os.path.join(ROOT,'results/frontier_v39',pubf))
                            if not l.startswith('#'))
             if r.get('status') != 'FAILED' and r['recall']])
        b4 = pareto(raw[(name, 4)])
        if b4:
            S['JHQ Br=4 (this work)'] = b4
        oom = []
        for lab, keys in [('CAGRA fp32', ['cagra']),
                          ('CAGRA int8', ['cagra_int8', 'cagra_int8_hi']),
                          ('IVF-PQ',     ['ivfpq', 'ivf_pq'])]:
            pts, failed = cuvs(pat, keys)
            if pts:   S[lab] = pareto(pts)
            elif failed: oom.append(lab)
        contributing = sorted({nl for r, q in S['JHQ Br=8 (this work)']
                               for rr, qq, nl in raw[(name, 8)]
                               if abs(rr - r) < 1e-9 and abs(qq - q) < 1e-6})
        data[name] = {'N': N, 'd': d, 'M': M, 'series': S, 'oom': oom,
                      'nlists': contributing,
                      'bpv': {'JHQ Br=8 (this work)': M + (d * 8 + 7) // 8 + 8,
                              'JHQ Br=8 (published)': M + (d * 8 + 7) // 8 + 8,
                              'JHQ Br=4 (this work)': M + (d * 4 + 7) // 8 + 8,
                              'CAGRA int8': d + 128, 'CAGRA fp32': 4 * d + 128}}
    return data

W, H, L, R, T, B = 430, 268, 54, 14, 12, 36
PW, PH = W - L - R, H - T - B

def fmt_qps(v):
    return f'{v/1e6:.1f}M' if v >= 1e6 else (f'{v/1000:.0f}K' if v >= 1000 else f'{v:.0f}')

def panel(name, d):
    S = d['series']
    xs = [r for s in S.values() for r, _ in s]
    ys = [q for s in S.values() for _, q in s]
    x0, x1 = max(0.62, min(xs) - 0.004), min(1.0, max(xs) + 0.004)
    y0, y1 = min(ys) * 0.72, max(ys) * 1.45
    ly0, ly1 = math.log10(y0), math.log10(y1)
    X = lambda r: L + (r - x0) / (x1 - x0) * PW
    Y = lambda q: T + (1 - (math.log10(q) - ly0) / (ly1 - ly0)) * PH
    g, k = [], math.floor(ly0)
    while k <= math.ceil(ly1):
        for mult in (1, 2, 5):
            v = mult * 10 ** k
            if y0 <= v <= y1:
                g.append(f'<line class="grid" x1="{L}" x2="{L+PW}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
                g.append(f'<text class="tick ty" x="{L-7}" y="{Y(v)+3.4:.1f}">{fmt_qps(v)}</text>')
        k += 1
    step = next(s for s in (0.01, 0.02, 0.025, 0.05, 0.1) if (x1 - x0) / s <= 6)
    t = math.ceil(x0 / step) * step
    while t <= x1 + 1e-9:
        g.append(f'<line class="grid" y1="{T}" y2="{T+PH}" x1="{X(t):.1f}" x2="{X(t):.1f}"/>')
        g.append(f'<text class="tick tx" x="{X(t):.1f}" y="{T+PH+15}">{t:.2f}</text>')
        t += step
    g.append(f'<rect class="frame" x="{L}" y="{T}" width="{PW}" height="{PH}"/>')
    for s in SER:
        if s not in S:
            continue
        pts = sorted(S[s])
        g.append('<polyline class="ln %s" points="%s"/>' %
                 (CLS[s], ' '.join(f'{X(r):.1f},{Y(q):.1f}' for r, q in pts)))
        lead = (s == 'JHQ Br=8 (this work)')
        for r, q in pts:
            g.append(f'<circle class="pt {CLS[s]}" cx="{X(r):.1f}" cy="{Y(q):.1f}" '
                     f'r="{2.7 if lead else 1.9}"/>')
    g.append(f'<text class="axl" x="{L+PW//2}" y="{H-4}">Recall@10</text>')
    g.append(f'<text class="axl" transform="translate(13,{T+PH//2}) rotate(-90)" x="0" y="0">QPS (log)</text>')
    P, E = S['JHQ Br=8 (published)'], S['JHQ Br=8 (this work)']
    lo, hi = max(P[0][0], E[0][0]), min(P[-1][0], E[-1][0])
    # 0.85 of the overlap: the top of the front is near-vertical and a 1e-3
    # build-to-build recall wobble is worth 2-3x there, so it is compared at
    # matched nlist and nprobe in the report instead of interpolated here.
    rr = lo + (hi - lo) * 0.85
    speed = interp(E, rr) / interp(P, rr)
    c8, bpv = S.get('CAGRA int8'), d['bpv']
    eq = (f"<span class='sep'>·</span>ceiling <b>{(E[-1][0]-c8[-1][0])*100:+.1f} pts</b> "
          f"vs int8 at {bpv['CAGRA int8']} B" if c8 else '')
    oom = ''.join(f'<span class="oom">{o} — cannot build</span>' for o in d['oom'])
    return f'''<figure class="panel">
  <figcaption><h3>{NICE[name]}</h3>
  <p class="meta">{d["N"]/1e6:.2f}M &times; {d["d"]}d &nbsp;·&nbsp; M={d["M"]} &nbsp;·&nbsp; \
{bpv["JHQ Br=8 (this work)"]} B/vec &nbsp;·&nbsp; nlist {", ".join(f"{n:,}" for n in d["nlists"])}</p></figcaption>
  <svg viewBox="0 0 {W} {H}" role="img" aria-label="{NICE[name]}">{''.join(g)}</svg>
  <p class="delta"><b>{speed:.2f}&times;</b> vs published at R={rr:.4f}{eq}</p>
  {('<p class="ooms">'+oom+'</p>') if oom else ''}
</figure>'''

def splice(html, panels):
    """Replace the panel grid in an existing page with freshly drawn ones."""
    i = html.index('<div class="grid6">')
    j = html.index('</div>', html.index('</figure>', html.rindex('<figure class="panel">')))
    return html[:i] + '<div class="grid6">' + panels + html[j:]

def main():
    data = collect()
    panels = '\n'.join(panel(n, data[n]) for n in DS)
    os.makedirs(OUT, exist_ok=True)
    json.dump(data, open(os.path.join(OUT, 'fronts.json'), 'w'), indent=1)
    for fname in ('frontier.html', 'fronts_six.html'):
        path = os.path.join(OUT, fname)
        if not os.path.exists(path):
            print(f'  {fname} not present; panels not spliced', file=sys.stderr); continue
        # Read fully, then write. open(path,'w') inside the same expression as
        # open(path).read() truncates before the read runs and leaves the file
        # empty -- it destroyed frontier.html once.
        with open(path) as fh:
            html = fh.read()
        spliced = splice(html, panels)
        with open(path, 'w') as fh:
            fh.write(spliced)
        print(f'  wrote {fname} ({len(spliced):,} B)')
    for n in DS:
        e = data[n]['series']['JHQ Br=8 (this work)']
        print(f'  {n:<15}{len(e):>3} points  R {e[0][0]:.4f}-{e[-1][0]:.4f}  '
              f'nlist {data[n]["nlists"]}')

if __name__ == '__main__':
    main()
