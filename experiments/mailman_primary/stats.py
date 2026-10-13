#!/usr/bin/env python3
"""Structure of the real JHQ primary codes, for the two mechanisms of experiments/mailman_primary.

    python3 stats.py <pp_pilot dir> <lf_pilot dir> <out.json> [--lists N]

Inputs (read only): <pp_pilot>/primary.u8 [N][M] (MSB of a byte = the subspace's first dimension,
bit 1 = +c), list_offsets.i32, and the probes of the 1000 sampled queries (rb_q*.bin in <lf_pilot>),
which weight every list by how often the scan reads it (probe count x list length).

A. Mailman (groups of r consecutive codes of one list; per dimension the r bits form a pattern):
   bit balance per (list, dimension); pattern entropy and the share of all-0 / all-1 patterns;
   distinct patterns a group uses over its d dimensions; and, for a warp that takes 32 consecutive
   dimensions of one group, the distinct patterns and the largest multiplicity among the 32 lanes
   (what a shared histogram's atomics would serialise on).
B. Lossless block dictionary (blocks of n consecutive codes of one list, chunks of g dimensions):
   U = distinct g-bit strings in a (block, chunk); theoretical bits n*ceil(log2 U) + U*g against n*g,
   a GPU format (index width rounded up to 4/8/16 bits, dictionary in whole bytes, 8 bytes of
   metadata per (block, chunk)); share of (block, chunk) where the dictionary is smaller; the same
   weighted by scan accesses; and the local-table work U*g/4 + n against the LUT's n*g/4.
"""
import glob
import json
import os
import struct
import sys
import time
from collections import Counter

import numpy as np


def load_probe_counts(lf_dir, nlist):
    cnt = np.zeros(nlist, np.int64)
    files = glob.glob(os.path.join(lf_dir, "rb_q*.bin"))
    for f in files:
        with open(f, "rb") as fh:
            hdr = struct.unpack("5i", fh.read(20))
            nprobe = hdr[2]
            probes = np.frombuffer(fh.read(4 * nprobe), np.int32)
        np.add.at(cnt, probes, 1)
    return cnt, len(files)


def main():
    pp, lf, out = sys.argv[1:4]
    max_lists = int(sys.argv[sys.argv.index("--lists") + 1]) if "--lists" in sys.argv else None
    meta = json.load(open(os.path.join(pp, "meta.json")))
    N, d, M, nlist = meta["N"], meta["d"], meta["M"], meta["nlist"]
    offs = np.fromfile(os.path.join(pp, "list_offsets.i32"), np.int32)
    codes = np.memmap(os.path.join(pp, "primary.u8"), np.uint8, "r", shape=(N, M))
    pcount, nq = load_probe_counts(lf, nlist)
    sizes = np.diff(offs)
    weight = pcount * sizes                       # codes the scan reads from each list over the sample
    order = np.argsort(-weight)
    lists = order if max_lists is None else order[:max_lists]
    t0 = time.time()

    res = {"tag": os.path.basename(os.path.normpath(pp)), "N": N, "d": d, "nlist": nlist, "queries": nq,
           "lists_analysed": int(len(lists)), "share_of_scan_reads": float(weight[lists].sum() / weight.sum()),
           "list_size": {"mean": float(sizes.mean()), "median": float(np.median(sizes)), "min": int(sizes.min()),
                         "max": int(sizes.max())}}

    # ── A. Mailman ──────────────────────────────────────────────────────────
    bal_dev = []          # |p1 - 1/2| per (list, dim), weighted later
    bal_w = []
    A = {}
    for r in (4, 8):
        ent_hist = np.zeros(1 << r, np.float64)          # pattern frequencies (scan weighted)
        used = []; used_w = []
        warp_distinct = []; warp_maxmult = []; warp_w = []
        for li in lists:
            a, b = offs[li], offs[li + 1]
            n = b - a
            if n == 0:
                continue
            bits = np.unpackbits(np.asarray(codes[a:b]), axis=1)      # [n][d], dimension order
            w = float(pcount[li])
            if r == 4:
                p1 = bits.mean(axis=0)
                bal_dev.append(np.abs(p1 - 0.5)); bal_w.append(np.full(d, w * n))
            ng = (n + r - 1) // r
            pad = np.zeros((ng * r - n, d), np.uint8)
            g = np.concatenate([bits, pad]).reshape(ng, r, d)          # [group][i][dim]
            pat = np.zeros((ng, d), np.int32)
            for i in range(r):
                pat |= g[:, i, :].astype(np.int32) << i
            # frequencies over (group, dim), weighted by how often the list is scanned
            np.add.at(ent_hist, pat.ravel(), w)
            # distinct patterns a group uses over its d dims (row-wise bincount via offsets)
            K = 1 << r
            cg = np.bincount((pat + (np.arange(ng)[:, None] * K)).ravel(), minlength=ng * K).reshape(ng, K)
            used.extend((cg > 0).sum(axis=1).tolist()); used_w.extend([w] * ng)
            # a warp over 32 consecutive dims of one group
            segs = pat.reshape(-1, 32)
            ns = segs.shape[0]
            cs = np.bincount((segs + (np.arange(ns)[:, None] * K)).ravel(), minlength=ns * K).reshape(ns, K)
            warp_distinct.extend((cs > 0).sum(axis=1).tolist()); warp_maxmult.extend(cs.max(axis=1).tolist())
            warp_w.extend([w] * ns)
        pfreq = ent_hist / ent_hist.sum()
        nz = pfreq[pfreq > 0]
        used = np.array(used); used_w = np.array(used_w)
        wd = np.array(warp_distinct); wm = np.array(warp_maxmult); ww = np.array(warp_w)
        # random-bit reference for a warp of 32 lanes into 2^r buckets
        rng = np.random.default_rng(0)
        sim = rng.integers(0, 1 << r, size=(20000, 32))
        sim_max = np.array([np.bincount(s, minlength=1 << r).max() for s in sim])
        A[f"r{r}"] = {
            "pattern_entropy_bits": float(-(nz * np.log2(nz)).sum()), "max_entropy_bits": r,
            "share_all_zero": float(pfreq[0]), "share_all_one": float(pfreq[(1 << r) - 1]),
            "top8_patterns": [[int(p), float(pfreq[p])] for p in np.argsort(-pfreq)[:8]],
            "distinct_patterns_per_group": {"mean": float(np.average(used, weights=used_w)),
                                            "of": 1 << r},
            "warp32": {"distinct_mean": float(np.average(wd, weights=ww)),
                       "max_multiplicity_mean": float(np.average(wm, weights=ww)),
                       "max_multiplicity_p95": float(np.percentile(wm, 95)),
                       "random_bits_max_multiplicity_mean": float(sim_max.mean())},
        }
    bd = np.concatenate(bal_dev); bw = np.concatenate(bal_w)
    res["bit_balance"] = {"mean_abs_dev_from_half": float(np.average(bd, weights=bw)),
                          "p90": float(np.percentile(bd, 90)), "share_dev_gt_0.25": float(np.average(bd > 0.25, weights=bw))}
    res["mailman"] = A

    # ── B. block dictionary ─────────────────────────────────────────────────
    B = {}
    for n in (64, 128, 256):
        for gbits in (8, 16, 32):
            raw = 0.0; theo = 0.0; impl = 0.0; worth = 0.0; tot = 0.0; worth_w = 0.0; tot_w = 0.0
            work_raw = 0.0; work_dict = 0.0
            Us = []
            for li in lists:
                a, b = offs[li], offs[li + 1]
                L = b - a
                if L == 0:
                    continue
                w = float(pcount[li])
                bits = np.unpackbits(np.asarray(codes[a:b]), axis=1)
                nchunk = d // gbits
                # chunk values as integers (gbits <= 32)
                pw = (1 << np.arange(gbits - 1, -1, -1, dtype=np.uint64))
                vals = (bits.reshape(L, nchunk, gbits).astype(np.uint64) * pw).sum(axis=2)   # [L][nchunk]
                for s0 in range(0, L, n):
                    blk = np.sort(vals[s0:s0 + n], axis=0)                  # [nb][nchunk]
                    nb = blk.shape[0]
                    U = 1 + (np.diff(blk, axis=0) != 0).sum(axis=0) if nb > 1 else np.ones(nchunk, np.int64)
                    Us.extend(U.tolist())
                    rb = nb * gbits
                    tb = nb * np.maximum(1, np.ceil(np.log2(U))) + U * gbits
                    iw = np.select([U <= 16, U <= 256, U <= 65536], [4, 8, 16], 32)
                    ib = nb * iw + U * ((gbits + 7) // 8) * 8 + 64
                    good = ib < rb
                    raw += rb * nchunk * w; theo += tb.sum() * w; impl += np.minimum(ib, rb).sum() * w
                    tot += nchunk; tot_w += nchunk * w
                    worth += good.sum(); worth_w += good.sum() * w
                    work_raw += nb * gbits / 4 * nchunk * w
                    work_dict += np.where(good, U * gbits / 4 + nb, nb * gbits / 4).sum() * w
            Us = np.array(Us)
            B[f"n{n}_g{gbits}"] = {"U_mean": float(Us.mean()), "U_median": float(np.median(Us)),
                                   "U_p10": float(np.percentile(Us, 10)), "U_max_possible": int(min(n, 2 ** gbits)),
                                   "theoretical_bits_ratio": theo / raw, "implementable_bytes_ratio_with_fallback": impl / raw,
                                   "share_blocks_worth_dictionary": worth / tot,
                                   "share_blocks_worth_dictionary_scan_weighted": worth_w / tot_w,
                                   "work_ratio_scan_weighted": work_dict / work_raw}
    res["dictionary"] = B
    res["seconds"] = time.time() - t0
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("dictionary",)}, indent=1)[:3000])
    for k, v in B.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
