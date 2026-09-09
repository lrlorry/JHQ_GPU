# Figures

Every figure is a script that reads the frozen logs in `../data/` and writes
PDF and PNG into `out/`. Nothing is hand-placed, so re-running after a
re-measurement is `./make_all.sh` and the numbers cannot drift out of step with
the text.

| script | figure | section | reads |
|---|---|---|---|
| `fig_frontier.py` | six-panel recall-QPS frontier | 6.2 | `paper_fronts.log`, `paper_rabitq.log`, `../../report_adc2026/v47/fronts.json` |
| `fig_alpha.py` | ranking loss vs alpha; what the rule is worth | 6.3 | `alpha_ds.log`, `paper_fronts.log` |
| `fig_batch.py` | throughput and the JHQ/RaBitQ ratio against batch | 6.6 | `batch_sweep.log` |
| `fig_ablation.py` | what paid, and what did not | 6.4, 6.5 | ranges from `NEGATIVES.md` and the version logs |
| `fig_hierarchy.py` | what the second level buys | 6.7 | `hierarchy_ablation.log`, `paper_fronts.log` |

`style.py` holds the loaders and the camera-ready settings, so a change there
applies to every figure at once.

## Conventions, and why

- **PDF with Type-42 fonts.** Matplotlib's PDF default is Type-3, which most
  proceedings pipelines reject.
- **Drawn at print size.** `COL = 3.33in`, `WIDE = 6.9in` are the one- and
  two-column widths. Include a figure at scale 1.0 and change the size in the
  script rather than scaling in LaTeX, or the 8pt text stops being 8pt.
- **No titles inside the axes.** The caption is the title; a duplicate costs a
  line of column. Panel letters are `(a)`, `(b)` inside the axes.
- **A marker per series, not colour alone**, so the figures survive greyscale
  printing and the two common colour-vision deficiencies. The hue order is the
  one whose adjacent pairs stay separable under both.
- **Dataset name and shape inside the panel** on the six-panel figure: six
  titles above six panels do not fit.

## Editing

Change the script, run it, and the PDF is replaced in place. `./make_all.sh`
redraws all five. To re-point a figure at new measurements, replace the log in
`../data/` -- the loaders parse the run scripts' own output format, so a
re-run drops in without edits.
