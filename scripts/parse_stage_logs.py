#!/usr/bin/env python3
"""Turn the stage sweep logs into CSVs so the figures never read a log directly.

These runs predate the harness and their output is a printed table, not a CSV.
Parsing them here keeps the same rule the rest of the results follow: every
number a figure draws comes from a file, and nothing is retyped by hand.

Writes, under results/pre_freeze_v22_s2b1/:
  abl_occupancy_cascade.csv   BLOCK and prefix separated (stage11)
  abl_alpha.csv               ck = alpha*k against recall and QPS (stage9)
  abl_klocal.csv              K_LOCAL against BLOCK, the shared-memory wall (stage18 B)
  abl_bytes.csv               M and Br against the recall ceiling (stage18 C)
  abl_nprobe_ceiling.csv      probing every list (stage18 A)
  build_phases.csv            train phases before and after (stage8/14/19)
"""
import csv, os, re, sys

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "results", "final")
L = os.path.join(D, "logs")


def write(name, header, rows):
    p = os.path.join(D, name)
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def lines(f):
    p = os.path.join(L, f)
    return open(p, errors="replace").read().splitlines() if os.path.exists(p) else []


# ── stage11: v16 baseline, then v21 at each BLOCK with the cascade off and on ──
rows = []
for ln in lines("stage11.log"):
    m = re.match(r"^(v16 \(BLOCK fixed 256\)|v21 k8)\s+(\d+)\s+(\S+)\s+([\d.]+)\s+(\d+)\s*$", ln)
    if m:
        variant, block, prefix, recall, qps = m.groups()
        rows.append(["v16" if variant.startswith("v16") else "v21",
                     block, prefix, recall, qps])
write("abl_occupancy_cascade.csv", ["variant", "block", "prefix", "recall", "qps"], rows)

# ── stage9: alpha sweep, both variants ──
rows = []
for ln in lines("stage9.log"):
    m = re.match(r"^(v16|v21 1/2 k8)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s*$", ln)
    if m:
        variant, alpha, ck, recall, qps = m.groups()
        rows.append(["v16" if variant == "v16" else "v21", alpha, ck, recall, qps])
write("abl_alpha.csv", ["variant", "alpha", "ck", "recall", "qps"], rows)

# ── stage18: three sections, split on their banners ──
sec, a_rows, b_rows, c_rows = None, [], [], []
for ln in lines("stage18.log"):
    if ln.startswith("=== A."): sec = "A"; continue
    if ln.startswith("=== B."): sec = "B"; continue
    if ln.startswith("=== C."): sec = "C"; continue
    if sec == "A":
        m = re.match(r"^v22 kl4\s+(\d+)\s+([\d.]+)\s+(\d+)\s*$", ln)
        if m: a_rows.append(list(m.groups()))
    elif sec == "B":
        m = re.match(r"^v22\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)?\s*(\d+)?\s*$", ln)
        if m:
            block, kl, nprobe, recall, qps = m.groups()
            # a blank recall is the launch failing for shared memory, which is
            # the point of this table: K_LOCAL and BLOCK cannot both be large
            b_rows.append([block, kl, nprobe, recall or "", qps or "",
                           "ok" if recall else "shared-memory limit"])
    elif sec == "C":
        m = re.match(r"^v22 kl4\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s*$", ln)
        if m:
            M, Br, _, nprobe, recall, qps = m.groups()
            # the printed "bytes" column counted M*Br/8; the residual level is
            # per dimension, so the payload is M + d*Br/8 + 4 at d=768
            payload = int(M) + 768 * int(Br) // 8 + 4
            c_rows.append([M, Br, payload, nprobe, recall, qps])
write("abl_nprobe_ceiling.csv", ["nprobe", "recall", "qps"], a_rows)
write("abl_klocal.csv", ["block", "k_local", "nprobe", "recall", "qps", "status"], b_rows)
write("abl_bytes.csv", ["M", "Br", "bytes_per_vec", "nprobe", "recall", "qps"], c_rows)

# ── build phases: stage8 is the original, stage19 the parallel residual loop ──
def phases(f):
    out = {}
    for ln in lines(f):
        m = re.match(r"^\s*\[train\]\s+(.+?)\s+([\d.]+) ms\s*$", ln)
        if m: out[m.group(1).strip()] = float(m.group(2))
        m = re.match(r"^\s*train:\s+([\d.]+) ms\s*$", ln)
        if m: out["total"] = float(m.group(1))
    return out

before, after = phases("stage8.log") or phases("stage14.log"), phases("stage19.log")
rows = [[k, f"{before.get(k, '')}", f"{after.get(k, '')}"]
        for k in dict.fromkeys(list(before) + list(after))]
write("build_phases.csv", ["phase", "before_ms", "after_ms"], rows)
