import json, sys, glob
import numpy as np
for f in sorted(glob.glob(sys.argv[1] + "/*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r["type"] == "failure":
            print(f.split("/")[-1], "FAILURE", r.get("label"), r.get("error", "")[:150]); continue
        if r["type"] != "point":
            continue
        for P in ("A", "B"):
            if P not in r: continue
            c = np.array([x for rr in r[P]["call_ms"] for x in rr])
            print(f"{f.split('/')[-1]:22s} {r['label']:24s} {json.dumps(r['search'])[:44]:44s} {P} R={r['recall']:.4f} "
                  f"p50={np.percentile(c,50):7.3f} p95={np.percentile(c,95):7.3f} p99={np.percentile(c,99):7.3f} "
                  f"max={c.max():7.3f} n={len(c)} qps_sum={r[P]['qps_median']:9.0f} qps_p50={1000*1000/np.percentile(c,50):9.0f} "
                  f"det={r.get('deterministic')} setdiff={r.get('rows_set_diff_'+P.lower())} orddiff={r.get('rows_order_diff_'+P.lower())}")
