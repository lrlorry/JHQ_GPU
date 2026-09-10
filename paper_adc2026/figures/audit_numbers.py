#!/usr/bin/env python3
"""Every number in the body, against a source that can produce it.

Five errors of one kind surfaced while writing this paper, and not one of them
was found by re-reading the text:

    52-54x       for a recall span whose endpoint measured 89.7x
    43-48x       for one that measured 79.8x
    3.8-13.4x    where the per-dataset build ratios are 3.9-7.4x
    1.6-2.9x     where CAGRA-int8's margins run 1.3-2.9x
    Figure 9     drawn from batch_sweep.log under a caption describing
                 batch_matched.log

All five came from writing a range or a claim from memory instead of from the
script that computes it. So this does not re-read the text. It extracts every
numeric literal from the body, then asks whether some file under data/ or some
generator under figures/ can produce it, and prints the ones nothing can.

An unmatched number is not necessarily wrong -- derived quantities, ratios and
rounded values will not appear verbatim in a log -- but every unmatched number
is one a human has to justify out loud, which is the check that was missing.
"""
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(HERE, os.pardir, "tex", "ADC")
DATA = os.path.join(HERE, os.pardir, "data")

# Numbers that are definitional, not measured: grid points, parameters, sizes
# fixed by the setup, section and reference numbers.
KNOWN = {
    "8", "10", "16", "32", "64", "96", "100", "128", "192", "200", "256", "384",
    "512", "1024", "2048", "4096", "8192", "32768", "159744",
    "1", "2", "3", "4", "5", "6", "7", "9", "11", "12", "14", "15", "20", "24",
    "26", "48", "768", "1536", "3072", "5090", "0.90", "0.93", "0.95", "0.97",
    "0.98", "0.99", "1.0", "0.001",
    # The machine and the corpora: nothing here is a result, and none of it
    # can come from a run log.
    "32607", "170", "754", "2.25", "10.1", "17.8",
    # S*k slots at S=32 and at the 500-query pool, with k=10; the text now
    # gives the arithmetic, so these are checkable without a log.
    "320", "5000",
}


def body_numbers():
    out = {}
    for f in sorted(glob.glob(os.path.join(TEX, "*.tex"))):
        if os.path.basename(f) == "samplepaper.tex":
            continue
        for i, ln in enumerate(open(f), 1):
            if ln.lstrip().startswith("%"):
                continue
            txt = re.sub(r"%.*", "", ln)
            # A brace list is a set of grid values, and my first pass glued
            # {32,128,512} into the number 32128512. ORCIDs are not
            # measurements either.
            txt = re.sub(r"\\orcidID\{[^}]*\}", " ", txt)
            # LaTeX writes a thousands separator as 1{,}024; the number
            # pattern below cannot span the braces, so it used to yield the
            # tail "024" as a number of its own.
            txt = txt.replace("{,}", ",")
            txt = re.sub(r"\\?\{\s*\$?\d[\d,.$\s]*\\?\}", " ", txt)
            # ...and a bare comma list, as in nprobe}=8,32,128,512$
            txt = re.sub(r"(?:\{=\}|=)\s*\d+(?:\s*,\s*\d+)+", " ", txt)
            for m in re.finditer(r"(?<![\w.])(\d+(?:[,{}]\d+)*(?:\.\d+)?)", txt):
                v = m.group(1).replace("{,}", "").replace(",", "")
                if v in KNOWN or len(v) < 2:
                    continue
                out.setdefault(v, []).append("%s:%d" % (os.path.basename(f), i))
    return out


def corpus():
    """Everything a number could legitimately have come from."""
    text = []
    for f in glob.glob(os.path.join(DATA, "*")):
        if os.path.isfile(f):
            try:
                text.append(open(f, errors="ignore").read())
            except Exception:
                pass
    for f in glob.glob(os.path.join(HERE, "*.py")):
        text.append(open(f, errors="ignore").read())
    # and whatever the generators print, since ranges are computed not stored
    for g in ("table_frontier.py", "build_crossover.py", "budget_arms.py",
              "cpu_gpu_envelope.py", "fig_issue.py", "fig_lutgroups.py",
              "fig_build.py", "fig_negatives.py", "fig_memory.py"):
        p = os.path.join(HERE, g)
        if not os.path.exists(p):
            continue
        try:
            r = subprocess.run([sys.executable, p], capture_output=True,
                               text=True, timeout=300, cwd=HERE)
            text.append(r.stdout)
        except Exception:
            pass
    return "\n".join(text)


# Every number in the corpus, parsed once; rounds_to() scans this instead of
# the raw text.
_CORPUS_VALS = None


def rounds_to(v, hay):
    """True if some number in the corpus rounds to v at v's own precision."""
    global _CORPUS_VALS
    if _CORPUS_VALS is None:
        _CORPUS_VALS = set()
        for m in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?", hay):
            try:
                _CORPUS_VALS.add(float(m.group(0)))
            except ValueError:
                pass
    dp = len(v.split(".")[1]) if "." in v else 0
    try:
        target = float(v)
    except ValueError:
        return False
    # A 2-digit integer quoted as "77" must not be satisfied by 77.4 from an
    # unrelated log unless it really rounds there; that is the whole point.
    return any(round(x, dp) == target for x in _CORPUS_VALS)


def main():
    nums, hay = body_numbers(), corpus()
    missing = []
    for v, where in sorted(nums.items(), key=lambda kv: -len(kv[0])):
        # match the digits with or without a thousands separator, and allow the
        # log to carry more precision than the text quotes
        # A bare substring search made this check far weaker than it read:
        # "9.8" matched the "19.8" inside an unrelated timing, which is how
        # three numbers in Section 6.2 passed while no generator produced
        # them.  Both ends are anchored -- no digit or dot may precede the
        # match, and only more precision may follow it.
        L, R = r"(?<![\d.])", r"(?![\d.])"
        pats = [L + re.escape(v) + R]
        if "." in v:
            pats.append(L + re.escape(v) + r"\d+" + R)
        if "." not in v:
            pats.append(L + re.escape(v) + r"\.\d+" + R)
        if len(v) > 3 and "." not in v:
            pats.append(L + re.escape(v[:-3]) + "," + re.escape(v[-3:]) + R)
        if any(re.search(p, hay) for p in pats):
            continue
        # The body rounds; the log does not.  "1.1" is a legitimate quote of
        # 1.09 and "77" of 76.7, and prefix matching cannot see either -- it
        # only accepts a log that carries *more* digits of the same value.
        # Rounding is checked numerically instead, at the precision the body
        # chose to quote.
        if rounds_to(v, hay):
            continue
        missing.append((v, where))
    print("body carries %d distinct measured numbers; %d have no source"
          % (len(nums), len(missing)))
    if missing:
        print("\nunmatched -- each needs a spoken justification:")
        for v, where in missing:
            print("  %-12s %s" % (v, ", ".join(sorted(set(where))[:3])))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
