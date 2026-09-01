#!/usr/bin/env python3
"""One driver for every final measurement: JHQ, cuVS IVF-PQ, CAGRA (fp32 and int8).

Written after a round of sweeps in which several rows that looked like data were
not: a launch that ran out of registers printed recall 0.0000 beside 285796 QPS,
a shell substitution called a binary whose name had lost its suffix and left the
row blank while its neighbours filled in, and a hand-copied CSV duplicated one
dataset's numbers onto another. So:

  * every row carries the commit, GPU, driver, CUDA and cuVS versions, and the
    full parameter set it was produced with;
  * every point is run REPS times after a warm-up and reports mean and stdev,
    not one number;
  * rows that cannot be parsed, or that carry the signature of a failed launch
    (recall 0 with an impossible QPS), are recorded as failures with their
    stderr, never silently dropped and never averaged into anything;
  * nothing is transcribed by hand -- the CSV is the only output.

Usage:
  bench_all.py --suite front --dataset vogue-768 --out results/final/front.csv
"""
import argparse, csv, json, os, re, subprocess, statistics, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    # name: (base, query, groundtruth, d, N)
    "vogue-768":      ("/root/data/vogue-768_base.fvecs",
                       "/root/data/vogue-768_query.fvecs",
                       "/root/data/vogue-768_groundtruth.ivecs", 768, 932328),
    "arxiv-768":      ("/root/autodl-tmp/arxiv-abstracts-768/base.fvecs",
                       "/root/autodl-tmp/arxiv-abstracts-768/query.fvecs",
                       "/root/autodl-tmp/arxiv-abstracts-768/groundtruth.ivecs", 768, 2253000),
    "openai3-1536":   ("/root/autodl-tmp/openai3-1536/base.fvecs",
                       "/root/autodl-tmp/openai3-1536/query.fvecs",
                       "/root/autodl-tmp/openai3-1536/groundtruth.ivecs", 1536, 999000),
    "openai3-3072":   ("/root/autodl-tmp/openai3-3072/base.fvecs",
                       "/root/autodl-tmp/openai3-3072/query.fvecs",
                       "/root/autodl-tmp/openai3-3072/groundtruth.ivecs", 3072, 999000),
    "bge-m3":         ("/root/autodl-tmp/bge-m3/base.fvecs",
                       "/root/autodl-tmp/bge-m3/query.fvecs",
                       "/root/autodl-tmp/bge-m3/groundtruth.ivecs", 1024, 10091524),
    "stella-trec24":  ("/root/autodl-tmp/stella-trec24/base.fvecs",
                       "/root/autodl-tmp/stella-trec24/query.fvecs",
                       "/root/autodl-tmp/stella-trec24/groundtruth.ivecs", 1024, 17776615),
}

RE_RECALL = re.compile(r"^Recall@\d+\s*:\s*([\d.]+)", re.M)
RE_QPS    = re.compile(r"^QPS\s*:\s*([\d.]+)", re.M)
RE_VRAM   = re.compile(r"^VRAM used\s*:\s*([\d.]+) MiB", re.M)
RE_TRAIN  = re.compile(r"^\s*train:\s*([\d.]+) ms", re.M)
RE_ADD    = re.compile(r"^\s*add:\s*([\d.]+) ms", re.M)


def env_metadata():
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    meta = {
        "commit": sh(f"git -C {REPO} rev-parse --short HEAD"),
        "dirty": "yes" if sh(f"git -C {REPO} status --porcelain") else "no",
        "gpu": sh("nvidia-smi --query-gpu=name --format=csv,noheader").split("\n")[0],
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader").split("\n")[0],
        "nvcc": (sh("nvcc --version | tail -1") or "").split(",")[-1].strip(),
        "host_cores": sh("nproc"),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import cuvs
        meta["cuvs"] = cuvs.__version__
    except Exception:
        meta["cuvs"] = "n/a"
    return meta


def looks_like_failed_launch(recall, qps):
    """A launch that fails for resources leaves every distance at its initial
    value: recall 0 next to a QPS no configuration reaches."""
    return recall is not None and recall == 0.0 and qps is not None and qps > 150000


def run_jhq(binary, ds, params, env, reps):
    """Run the JHQ demo `reps` times and return one aggregated record."""
    base, query, gt, d, N = DATASETS[ds]
    argv = [os.path.join(REPO, "build", binary), base, query, gt,
            str(params["M"]), str(params["B"]), str(params["Br"]),
            str(params["alpha"]), str(params["k"]), str(params["nlist"]),
            str(params["nprobe"]), str(params["ivf_iters"]),
            str(params["batch"]), "", str(params["kmeans_iters"])]
    e = dict(os.environ); e.update({k: str(v) for k, v in env.items()})
    qpss, recalls, vram, train_ms, add_ms, errs = [], [], None, None, None, []
    for _ in range(reps):
        p = subprocess.run(argv, capture_output=True, text=True, env=e, timeout=3600)
        out = p.stdout + p.stderr
        mr, mq = RE_RECALL.search(out), RE_QPS.search(out)
        if not (mr and mq):
            errs.append((p.returncode, out.strip().splitlines()[-3:]))
            continue
        r, q = float(mr.group(1)), float(mq.group(1))
        if looks_like_failed_launch(r, q):
            errs.append((p.returncode, ["failed-launch signature: recall 0 with %.0f QPS" % q]))
            continue
        recalls.append(r); qpss.append(q)
        mv = RE_VRAM.search(out)
        if mv: vram = float(mv.group(1))
        mt, ma = RE_TRAIN.search(out), RE_ADD.search(out)
        if mt: train_ms = float(mt.group(1))
        if ma: add_ms = float(ma.group(1))
    return aggregate("JHQ", binary, ds, params, env, qpss, recalls,
                     vram, train_ms, add_ms, errs)


def aggregate(method, variant, ds, params, env, qpss, recalls,
              vram, train_ms, add_ms, errs):
    ok = len(qpss) > 0
    return {
        "method": method, "variant": variant, "dataset": ds,
        "d": DATASETS[ds][3], "N": DATASETS[ds][4],
        "params": json.dumps(params, sort_keys=True),
        "env": json.dumps(env, sort_keys=True),
        "n_runs": len(qpss),
        "recall": f"{statistics.mean(recalls):.4f}" if ok else "",
        "recall_spread": f"{(max(recalls)-min(recalls)):.5f}" if ok else "",
        "qps_mean": f"{statistics.mean(qpss):.0f}" if ok else "",
        "qps_std": f"{statistics.stdev(qpss):.0f}" if len(qpss) > 1 else "0",
        "qps_rel_std_pct": f"{100*statistics.stdev(qpss)/statistics.mean(qpss):.2f}" if len(qpss) > 1 else "0",
        "vram_mib": f"{vram:.0f}" if vram else "",
        "train_ms": f"{train_ms:.0f}" if train_ms else "",
        "add_ms": f"{add_ms:.0f}" if add_ms else "",
        "status": "ok" if ok else "FAILED",
        "failures": json.dumps(errs[:2]) if errs else "",
    }


FIELDS = ["method", "variant", "dataset", "d", "N", "params", "env", "n_runs",
          "recall", "recall_spread", "qps_mean", "qps_std", "qps_rel_std_pct",
          "vram_mib", "train_ms", "add_ms", "status", "failures"]


def write_rows(path, rows, meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write("# " + json.dumps(meta, sort_keys=True) + "\n")
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}: {len(rows)} rows, "
          f"{sum(1 for r in rows if r['status']!='ok')} failed", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--binary", default="demo_jhq_v22_s2b1")
    ap.add_argument("--nprobe", default="1,4,8,16,32,64,128,256")
    ap.add_argument("--M", type=int, default=96)
    ap.add_argument("--Br", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--prefix", default="1/2")
    a = ap.parse_args()

    meta = env_metadata()
    print("env: " + json.dumps(meta, sort_keys=True), file=sys.stderr)
    num, den = a.prefix.split("/")
    env = {"JHQ_BLOCK": a.block, "JHQ_PFX_NUM": num, "JHQ_PFX_DEN": den,
           "JHQ_TILE_M_RT": a.M}
    rows = []
    for np_ in [int(x) for x in a.nprobe.split(",")]:
        params = dict(M=a.M, B=8, Br=a.Br, alpha=a.alpha, k=10, nlist=a.nlist,
                      nprobe=np_, ivf_iters=8, batch=a.batch, kmeans_iters=5,
                      prefix=a.prefix)
        r = run_jhq(a.binary, a.dataset, params, env, a.reps)
        rows.append(r)
        print(f"  nprobe={np_:<5} {r['status']:<7} R={r['recall']} "
              f"QPS={r['qps_mean']}±{r['qps_std']} ({r['qps_rel_std_pct']}%)",
              file=sys.stderr, flush=True)
    write_rows(a.out, rows, meta)


if __name__ == "__main__":
    main()
