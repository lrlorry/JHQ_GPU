# Where results live

A directory is named after **what produced it**, because a number is only
interpretable if you can tell which implementation it came from. That was not
true of the old `final/`: every run of every version wrote into it, and once
the search semantics changed there was no way to tell the rows apart.

```
pre_freeze_v22_s2b1/   every JHQ row here has variant=demo_jhq_v22_s2b1
paper_faithful_v1/     runs from the frozen target; empty until one is made
sensitivity/           parameter studies, one directory each
archive/               superseded: v15, v17, gpu_operating_point
```

## The rule for new runs

A run goes in a directory named for the tag or version that produced it —
`paper_faithful_v1/`, `paper_experiment_v1/`, and so on. Never into a directory
that already holds another version's numbers.

`variant` in each CSV names the binary, and the header records the commit and
every `JHQ_*` the run saw, so a row can be traced without relying on the
directory. The directory is for reading at a glance; the header is the record.

## pre_freeze_v22_s2b1 is not JHQ-GPU

Those rows were measured before `paper-faithful-v1` and differ from it in ways
that were measured, not assumed:

- the prefix cascade was on, pruning candidates before their full primary
  distance existed;
- the per-thread retention buffer was in use, whose selected candidate set
  differs from the exact top-alpha*k;
- the primary lookup table was fp16, rounding each Equation 6 term to eleven
  bits before summing.

`jhq_v23_faithful/PAPER_CONFIG.md` says the same and is the authority. Keep
them for comparison; do not report them as JHQ-GPU.

## File naming

| prefix | what it is |
|---|---|
| `p0_<dataset>_<method>.csv` | recall/QPS frontier |
| `fair_<dataset>_<method>_<variant>.csv` | fairness runs at matched budgets |
| `sat_<dataset>_<method>.csv` | saturation sweeps |
| `abl_<name>.csv` | ablations |
| `build_<name>.csv` | index build timings |
| `cpu_<name>.csv` | CPU reference |
| `fig_<name>.png` | figures, regenerated from the CSVs beside them |
| `meta_<name>` | dataset tables, availability notes, anything descriptive |

Files that predated this convention were renamed into it; `git log --follow`
still finds their history.
