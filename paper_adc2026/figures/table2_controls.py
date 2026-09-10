#!/usr/bin/env python3
"""Table 2's control rows, each from the log that measured it.

Why this exists: every range in Table 2 was a literal in fig_ablation.py, typed
from prose summaries written during development rather than derived from a log.
Two of them were wrong, in the same way -- the top of the range is the log's
maximum, and the bottom excludes the cells where the change barely paid or did
not pay at all:

  probe cursor    published +2.5 to +148%   measured -1.4% to +148.5%
  packed loads    published  +30 to  +48%   measured  +6.1% to +48.1%

Both maxima are exact.  Both minima drop the low-probe-depth cells, which are
the cells where the change is worth least -- and in the cursor's case, two where
it is worth slightly less than nothing.  Neither error survives contact with the
log it came from; nobody re-derived them, and audit_numbers.py could not catch
them because it only asks whether a number exists somewhere, not whether it came
from the right place.

The other two rows check out exactly as published, which is worth saying as
plainly as the errors:

  LUT construction share   published 0.2-2.3%      measured 0.20-2.30%
  table-free distance      published -30 to -52%   fig_negatives.py agrees

Sources, all measured before the paper and all in data/ now:

  v51_cursor.log   v51_off / v51_on, 4 datasets x 5 probe depths.  The cursor
                   changes no distance, so recall must not move: it moves by at
                   most 1e-4 across the twenty pairs.
  v52_layout.log   v52_byte / v52_word, 4 datasets x 4 probe depths, one build.
  k_and_stages.log JHQ_STEP_TIMING on v58, six batches a configuration.  The
                   timers bypass the CUDA graph, so the totals are not
                   comparable with paper_fronts.log -- the shares are, because
                   every stage pays the same tax.
  layout6.log      the same layout question re-measured on six datasets at
                   v59, reported by layout_square.py.  It agrees with v52 on
                   the structure that matters: the gain rises with probe depth.
"""
import collections
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style


def ab(logname, off, on):
    """(cell -> (recall, qps)) for two arms of one front6 log."""
    R = collections.defaultdict(dict)
    for ln in open(style.datafile(logname), errors="ignore"):
        m = re.match(r"\s*demo_jhq_(\S+)\s+(\S+)\s+nlist=(\d+)\s+np=(\d+)\s+"
                     r"blk=(\d+)\s+recall=([\d.]+)\s+qps=(\d+)", ln)
        if m:
            R[(m.group(2), int(m.group(4)))][m.group(1)] = (float(m.group(6)),
                                                            int(m.group(7)))
    pairs = [(k, v[off], v[on]) for k, v in sorted(R.items())
             if off in v and on in v]
    return pairs


def report(name, pairs, published):
    if not pairs:
        print("  %-22s no paired cells in the log" % name); return
    g = [100 * (b[1] / a[1] - 1) for _, a, b in pairs]
    dr = max(abs(b[0] - a[0]) for _, a, b in pairs)
    neg = [(k, x) for (k, _, _), x in zip(pairs, g) if x <= 0]
    print("  %-22s %+.1f%% to %+.1f%% over %d cells   (published %s)"
          % (name, min(g), max(g), len(g), published))
    print("      recall moves by at most %.4f; %d cells at or below zero%s"
          % (dr, len(neg),
             ": " + ", ".join("%s np=%d %+.1f%%" % (k[0], k[1], x)
                              for k, x in neg) if neg else ""))


def stage_share(stage):
    v = collections.defaultdict(list)
    for ln in open(style.datafile("k_and_stages.log"), errors="ignore"):
        m = re.match(r"\s*STAGE (\S+) np=(\d+) a=([\d.]+) k=(\d+) %s\s+"
                     r"[\d.]+\s+([\d.]+)%%" % stage, ln)
        if m:
            v[(m.group(1), int(m.group(2)), m.group(3))].append(float(m.group(5)))
    med = [statistics.median(x) for x in v.values()]
    return med, len(v)


def main():
    report("probe cursor", ab("v51_cursor.log", "v51_off", "v51_on"),
           "+2.5 to +148%")
    report("packed 32-bit loads", ab("v52_layout.log", "v52_byte", "v52_word"),
           "+30 to +48%")
    med, n = stage_share("build_byte_lut")
    if med:
        print("  %-22s %.2f%% to %.2f%% of a batch over %d configurations   "
              "(published 0.2-2.3%%)"
              % ("LUT construction", min(med), max(med), n))


if __name__ == "__main__":
    main()
