#!/usr/bin/env python3
"""How far v70's fp16 centroid scoring moves the search from v69's exact one, on one workload.

    python3 compare_v70.py <v69 prefix> <v70 prefix> <v69 dump> <v70 dump>

Same inputs as compare_v69.py (identity runs: G = 0, 1, -1 with checks, b = 8 and 1000, and the G = 1
dump of probes and primary top-ck at b = 8), but v70 is not expected to be identical, so this reports
agreement: queries whose probe list is identical, whose probe set is identical, the mean overlap of
the probe sets, the same for the primary top-ck position sets, and the recall of every (batch, G).
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from compare_v69 import load_dump  # noqa: E402


def main():
    p69, p70, d69, d70 = sys.argv[1:5]
    a, b = load_dump(d69), load_dump(d70)
    assert (a["nused"], a["nprobe"], a["ck"]) == (b["nused"], b["nprobe"], b["ck"])
    n, npb = a["nused"], a["nprobe"]
    same_list = int((a["probes"] == b["probes"]).all(axis=1).sum())
    ov = [len(set(a["probes"][q]) & set(b["probes"][q])) for q in range(n)]
    same_set = sum(o == npb for o in ov)
    tov = [len(set(a["pos"][q]) & set(b["pos"][q])) for q in range(n)]
    same_top = sum(o == a["ck"] for o in tov)
    print(f"queries {n}  nprobe {npb}  ck {a['ck']}")
    print(f"probe list identical {same_list}/{n}, probe set identical {same_set}/{n}, "
          f"mean probe overlap {np.mean(ov) / npb:.5f}, worst query {min(ov)}/{npb}")
    print(f"primary top-ck set identical {same_top}/{n}, mean overlap {np.mean(tov) / a['ck']:.5f}, "
          f"worst {min(tov)}/{a['ck']}")
    j69, j70 = json.load(open(p69 + ".json")), json.load(open(p70 + ".json"))
    rec = {}
    for tag, j in (("v69", j69), ("v70", j70)):
        for r in j["runs"]:
            for v in r["variants"]:
                rec.setdefault((r["alpha"], r["batch"], v["G"]), {})[tag] = v["recall"]
    print("alpha  batch  G   recall v69  recall v70  diff")
    for key in sorted(rec):
        r = rec[key]
        print(f"{key[0]:>5}  {key[1]:>5}  {key[2]:>2}   {r.get('v69', float('nan')):.4f}      "
              f"{r.get('v70', float('nan')):.4f}   {r.get('v70', 0) - r.get('v69', 0):+.4f}")
    print(json.dumps(dict(probe_list_identical=same_list, probe_set_identical=same_set,
                          probe_overlap=float(np.mean(ov) / npb), topck_set_identical=same_top,
                          topck_overlap=float(np.mean(tov) / a["ck"]), nused=n)))


if __name__ == "__main__":
    main()
