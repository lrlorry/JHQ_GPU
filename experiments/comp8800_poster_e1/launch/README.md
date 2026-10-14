# Launch scripts as run on the box

The order the GPU actually ran them, each holding `/root/.gpu_lock` for its whole run:

| script | what | UTC (2026-10-14) |
|---|---|---|
| `smoke_e1.sh`, `smoke2.sh`, `smoke3.sh` | three smoke rounds on a tiny vogue grid (results in `../SMOKE*/`); found the lock re-entry, the cuVS handle segfault, CAGRA's uint32 ids, RaBitQ's search spikes, CPU throttling | 12:30-13:05 |
| `e1a_launch.sh` | E1-A: vogue-768 and arxiv-768, every method | 13:06-13:44 |
| `e1b_launch.sh` | JHQ at more nprobe on the 768-d sets; E1-B: openai3-3072, every method | 13:44-14:34 |
| `e1e_launch.sh` | JHQ's high-recall end on openai3-3072 (alpha up to 256, nprobe 1024) and its low-recall end on the 768-d sets (nprobe 16, 32) | 14:34-14:37 |
| `e1c_launch.sh` | openai3-1536, every method (the optional fourth dataset) | 14:37-15:12 |
| `e1d_launch.sh` | JHQ nprobe 32 on openai3-1536; one cold JHQ build per dataset (empty trained-state cache) | 15:12-15:13 |
| `e1f_launch.sh` | JHQ's high-recall end on openai3-1536 | 15:36-15:38 |

Each was shipped to `/root/autodl-tmp/` and run with `setsid nohup`; all of them call
`../run_e1.sh` from the worktree `/root/autodl-tmp/jhq_pipeline_wt`, which was put at the commit
under test by `git fetch <bundle> main && git checkout --detach FETCH_HEAD`. None of them fetches,
resets or touches `/root/JHQ_GPU`. `calldist.py` summarises per-call latency from a jsonl directory
(used on the smoke rounds).
