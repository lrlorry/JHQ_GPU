# Figures

Every figure is a script that reads the frozen logs in `../data/` and writes
PDF and PNG into `out/`. Nothing is hand-placed, so re-running after a
re-measurement is `./make_all.sh` and the numbers cannot drift out of step with
the text.

| script | figure | section | reads |
|---|---|---|---|
| `fig_pipeline.py` | the JHQ-GPU pipeline, shaded by what changed | 3.1 | schematic |
| `fig_lut.py` | the Cartesian factorisation, and what it saves per query | 3.3.2 | schematic + arithmetic |
| `fig_rule.py` | how the budget is chosen, and one real calibration | 4.3 | schematic + `alpha_fast.log` |
| `fig_frontier.py` | six-panel recall-QPS frontier | 6.2 | `paper_fronts.log`, `paper_rabitq.log`, `../../report_adc2026/v47/fronts.json` |
| `fig_alpha.py` | ranking loss vs alpha; what the rule is worth | 6.3 | `alpha_ds.log`, `paper_fronts.log` |
| `fig_batch.py` | throughput and the JHQ/RaBitQ ratio against batch | 6.6 | `batch_sweep.log` |
| `fig_ablation.py` | what paid, and what did not | 6.4, 6.5 | ranges from `NEGATIVES.md` and the version logs |
| `fig_calibration.py` | the rule against the sweep; sample size; tolerance | 6.3 | `alpha_sample.log`, `alpha_fast.log` |
| `fig_negatives.py` | the table-free distance, and reuse bounded from above | 6.5 | `v54.log`, `qdup.log`, `qdup_stella.log` |
| `fig_hierarchy.py` | what the second level buys | 6.7 | `hierarchy_ablation.log`, `paper_fronts.log` |
| `fig_cost.py` | resident memory, and build time | 6.7 | `vram.log`, `paper_fronts.log` |

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

## Not yet drawn

- **The CPU reference.** `../data/jhq_cpu_ivf_*.csv` covers six datasets but at
  a protocol that does not match the GPU side -- the residual codebook took 25
  Lloyd iterations against 2000, and the thread count was not pinned. A figure
  from those numbers would compare training budgets. See section 10 of
  `../README.md`.
The three schematics are matplotlib like the rest, so `make_all.sh` produces
every figure the paper needs in one command and a co-author does not have to
choose a second toolchain. If they are later redrawn in TikZ to pick up the
paper\'s own fonts, these stay as the reference for what they should say.
