#!/usr/bin/env python3
"""End-to-end identity of v69 against v65 on one workload (v69_run.sh step 3).

    python3 compare_v69.py <v65 prefix> <v69 prefix> <v65 dump> <v69 dump>

Prefixes name the harness outputs (<prefix>.json) of the identity runs: variants
G = 0, 1, -1 with checks, b = 8 and 1000. The dumps are the G = 1 probe ids and
exact primary top-ck of every used query at the first batch size
(demo_jhq_batch_pipeline.cu, JHQ_BP_DUMP). Prints the per-query comparison and
the recall of every (batch, G) in both binaries; the last line is
IDENTITY_OK or IDENTITY_FAIL.
"""
import json
import sys

import numpy as np


def load_dump(path):
    raw = np.fromfile(path, dtype=np.int32)
    nused, nprobe, ck, nlist = (int(x) for x in raw[:4])
    o = 4
    sizes = raw[o:o + nlist]; o += nlist
    probes = raw[o:o + nused * nprobe].reshape(nused, nprobe); o += nused * nprobe
    pos = raw[o:o + nused * ck].reshape(nused, ck); o += nused * ck
    val = raw[o:o + nused * ck].view(np.float32).reshape(nused, ck); o += nused * ck
    assert o == raw.size, (path, o, raw.size)
    return dict(nused=nused, nprobe=nprobe, ck=ck, nlist=nlist, sizes=sizes, probes=probes, pos=pos, val=val)


def main():
    p65, p69, d65, d69 = sys.argv[1:5]
    ok = True
    a, b = load_dump(d65), load_dump(d69)
    for key in ("nused", "nprobe", "ck", "nlist"):
        if a[key] != b[key]:
            print(f"header {key}: v65 {a[key]} v69 {b[key]}")
            ok = False
    if ok:
        probe_q = int((a["probes"] == b["probes"]).all(axis=1).sum())
        pos_q = int((a["pos"] == b["pos"]).all(axis=1).sum())
        val_q = int((a["val"].view(np.int32) == b["val"].view(np.int32)).all(axis=1).sum())
        sizes_same = bool((a["sizes"] == b["sizes"]).all())
        print(f"queries {a['nused']}  nprobe {a['nprobe']}  ck {a['ck']}  nlist {a['nlist']}")
        print(f"list sizes identical: {sizes_same}")
        print(f"probe ids identical (same lists, same order): {probe_q}/{a['nused']}")
        print(f"primary top-ck positions identical: {pos_q}/{a['nused']}, distances bit-identical: {val_q}/{a['nused']}")
        ok = sizes_same and probe_q == pos_q == val_q == a["nused"]
    j65, j69 = json.load(open(p65 + ".json")), json.load(open(p69 + ".json"))
    rec = {}
    for tag, j in (("v65", j65), ("v69", j69)):
        for r in j["runs"]:
            for v in r["variants"]:
                rec.setdefault((r["alpha"], r["batch"], v["G"]), {})[tag] = v
    print("alpha  batch  G   recall v65  recall v69   checks vs G=0 (v69)")
    for key in sorted(rec):
        r65, r69 = rec[key].get("v65"), rec[key].get("v69")
        if r65 is None or r69 is None:
            print(key, "missing in one binary"); ok = False; continue
        same = r65["recall"] == r69["recall"]
        ok = ok and same
        chk = {k: v for k, v in r69.items() if k not in ("G", "recall", "used_mib", "group_scratch_mib",
                                                         "lat_med_ms", "lat_p95_ms", "qps")}
        print(f"{key[0]:>5}  {key[1]:>5}  {key[2]:>2}   {r65['recall']:.4f}      {r69['recall']:.4f}   "
              f"{'same' if same else 'DIFFERENT'}  {json.dumps(chk)}")
    print("IDENTITY_OK" if ok else "IDENTITY_FAIL")


if __name__ == "__main__":
    main()
