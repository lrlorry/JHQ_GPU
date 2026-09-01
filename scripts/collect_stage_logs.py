#!/usr/bin/env python3
"""Fold the stage*.log runs into one table.

The sweeps were run as a chain of numbered scripts, several of which had to be
redone (a build that silently produced no binary, a register limit that made a
launch fail without a message, a shell substitution that called a binary whose
name was missing its suffix). Rows that came back blank, or that report recall
0.0000 beside a QPS in the hundreds of thousands, are failures rather than
measurements and are dropped here rather than being averaged into anything.

Usage: collect_stage_logs.py stage*.log > results/v21_sweeps.csv
"""
import csv, re, sys

# "<variant> <n> <n> <recall> <qps>"-ish rows, with a leading label that may
# contain spaces (e.g. "v21fast pfx 1/2", "v21 casc 1/4 k8 blk1024").
ROW = re.compile(r"^(?P<label>\S.*?)\s{2,}(?P<rest>[-\d./]+(?:\s+[-\d./]+)*)\s*$")
NUM = re.compile(r"^-?\d+(?:\.\d+)?$")


def looks_bogus(recall, qps):
    """A launch that fails for resources leaves every distance at its initial
    value: recall 0.0000 next to an impossibly high QPS."""
    return recall is not None and recall == 0.0 and qps is not None and qps > 150000


def main(paths):
    w = csv.writer(sys.stdout)
    w.writerow(["source", "label", "fields", "recall", "qps"])
    kept = dropped = 0
    for p in paths:
        section = ""
        for line in open(p, errors="replace"):
            line = line.rstrip("\n")
            if line.startswith("==="):
                section = line.strip("= ").strip()
                continue
            m = ROW.match(line)
            if not m:
                continue
            parts = m.group("rest").split()
            nums = [float(x) for x in parts if NUM.match(x)]
            if len(nums) < 2:
                dropped += 1
                continue
            recall, qps = nums[-2], nums[-1]
            if not (0.0 <= recall <= 1.0):
                dropped += 1
                continue
            if looks_bogus(recall, qps):
                dropped += 1
                continue
            w.writerow([f"{p}:{section}", m.group("label"), " ".join(parts[:-2]),
                        f"{recall:.4f}", f"{qps:.0f}"])
            kept += 1
    print(f"# kept {kept}, dropped {dropped} blank/failed rows", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:] or sys.exit(__doc__))
