# Figures

Every figure is a script that reads the frozen logs in `../data/` and writes
PDF and PNG into `out/`. Nothing is hand-placed, so re-running after a
re-measurement is `./make_all.sh` and the numbers cannot drift out of step with
the text.

| script | figure | section | reads |
|---|---|---|---|
| `fig_pipeline.py` | the JHQ-GPU pipeline, shaded by what changed | 3.1 | schematic |
| `fig_lut.py` | the Cartesian factorisation, and what it saves per query | 3.3.2 | schematic + arithmetic |
| `fig_layout.py` | the transpose, and four subspaces in one load | 3.2.3, 3.3.3 | schematic |
| `fig_rule.py` | how the budget is chosen, and one real calibration | 4.3 | schematic + `alpha_fast.log` |
| `fig_frontier.py` | six-panel recall-QPS frontier | 6.2 | `paper_fronts.log`, `paper_rabitq.log`, `../../report_adc2026/v47/fronts.json` |
| `fig_alpha.py` | ranking loss vs alpha; what the rule is worth | 6.3 | `alpha6.log`, `paper_fronts.log` |
| `fig_batch.py` | throughput and the JHQ/RaBitQ ratio against batch | 6.6 | `batch_sweep.log` |
| `fig_ablation.py` | what paid, and what did not | 6.4, 6.5 | ranges from `NEGATIVES.md` and the version logs |
| `fig_calibration.py` | the rule against the sweep; sample size; tolerance | 6.3.2 | `alpha_sample.log`, `alpha6.log`, `alpha_fast.log` |
| `fig_economics.py` | what the calibration costs, and when it is repaid | 6.3.3 | `paper_fronts.log` |
| `fig_negatives.py` | the table-free distance, and reuse as a controlled proxy | 6.5 | `v54.log`, `qdup.log`, `qdup_stella.log` |
| `fig_hierarchy.py` | what the second level buys | 6.4 | `hierarchy_ablation.log`, `paper_fronts.log` |
| `fig_cost.py` | resident memory, and JHQ's own train/encode split | 6.7 | `vram.log`, `paper_fronts.log` |
| `fig_build.py` | **index build time, all five methods** | 6.7 | `results/**/*.csv` (`train_ms`), `paper_fronts.log`, `paper_rabitq.log` |
| `fig_memory.py` | where JHQ\'s memory goes, and why it is above RaBitQ\'s | 6.7 | index parameters + both measured totals |

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
- **No annotation below 7pt.** Camera-ready reduction is already applied by
  drawing at print size; anything smaller than 7pt does not survive a
  photocopy or a projector.
- **Nothing is shaded to mean "cannot".** A curve stops where its sweep
  stopped. The only claims of the form "cannot" in these figures are the two
  allocation failures, and they are stated in words.
- **One hue and one marker per dataset, from `DS_COLOR`/`DS_MARK`**, so a
  dataset never changes colour between figures. The six were validated as a
  categorical palette: lightness band, chroma floor and normal-vision
  separation pass. Two results constrain use rather than the palette --
  openai3-1536 and openai3-3072 separate by only \u0394E 7.2 under protanopia,
  which is legal only because every series also carries a marker, so never
  drop the markers; and bge-m3 is at 2.74:1 against the surface, so a figure
  leaning on it needs a visible label rather than a legend swatch alone.
- **An absent bar is a claim, so say which claim.** `fig_build` separates
  "the build fails on this card" (CAGRA fp32 and IVF-RaBitQ on bge-m3 and
  stella, each with the log that records it) from "we never timed it"
  (IVF-RaBitQ on openai3-3072). Drawing them the same way would assert a
  limitation of a baseline that we have not measured.
- **A sentinel is not a quantity.** `batches_to_repay` is -1 in the log when
  the gain is below one. `fig_economics` draws those above a rule instead of
  plotting them, because "never" has no position on a log axis.

## Editing

Change the script, run it, and the PDF is replaced in place. `./make_all.sh`
redraws all five. To re-point a figure at new measurements, replace the log in
`../data/` -- the loaders parse the run scripts' own output format, so a
re-run drops in without edits.

## Corrections applied to the drawings

Findings that were stated one way in the logs and are drawn another way here,
because the first statement did not survive checking:

- **The word packing is not a coalescing win.** From Pascal on, a global
  access is served in 32-byte sectors, so the byte-per-thread read the
  transpose produces already fetches exactly the sectors it uses. `fig_layout`
  panel (b) attributes the +30-48% to instruction count -- one load replacing
  four -- which is also what v54's table-free result implies from the other
  side. The 128-byte cache-line story in the v52 CMake comment is corrected
  there too.
- **The alpha sweep is keyed on (dataset, nprobe).** `ALPHA_RULE.md` had four
  nprobe=512 rows whose "sweep says" column reused the nprobe=128 sweep. There
  is no nprobe=512 sweep on those datasets, so `fig_calibration` panel (a)
  draws only the four nprobe=128 configurations, and the note is corrected.
- **Ranking loss means ranking loss.** `fig_alpha` plots
  `ivf_recall - recall`, from `alpha6.log`, which is the only run that
  recorded `ivf_recall`. `1 - recall` is mostly routing loss, and alpha cannot
  reach a neighbour whose list was never opened.
- **The batch figure is not iso-recall.** Both systems run at nprobe=128 and
  panel (b) states the recall each reaches, because the ratio is a throughput
  ratio at one operating point.
- **Query duplication is a proxy, not a bound.** It varies how much queries
  have in common while holding the schedule fixed; a cluster-centric rewrite
  would change the schedule too.
- **The memory stack reconciles.** `fig_memory`'s last segment is defined as
  measured minus modelled, so the bar ends at the `cudaMemGetInfo` total
  instead of stopping short of it without saying so.

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
