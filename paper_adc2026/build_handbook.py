#!/usr/bin/env python3
"""Build one PDF a co-author can draft from: the skeleton, with the figures.

    python3 build_handbook.py     ->  handbook.pdf

Two parts, because a co-author needs two different things and only one of them
is in the skeleton.

Part I is the mother skeleton, converted, with each figure placed at the
section it belongs to. Part II is a figure reference: every figure at full
width followed by its script's docstring verbatim. Those docstrings are where
the measured numbers, the caveats and the corrections actually live -- what a
bar means, which rows are excluded and why, which floor applies -- and they
are currently readable only by opening the .py. Captions cannot be written
from the figures alone, so the reference is the half of this document that
does work the skeleton does not.

No pandoc on this machine; the converter below handles the subset the
skeleton uses (headings, pipe tables, blockquotes, bullets, inline emphasis
and code) and nothing else. It is deliberately small: if the skeleton grows a
construct it does not know, the right fix is to add it here rather than to
hand-edit generated TeX.
"""
import os, re, subprocess, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
OUT  = os.path.join(FIGS, "out")
SKEL = os.path.join(HERE, "ADC_sections_3_4_6_mother_skeleton_v3.md")

SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def esc(t):
    return "".join(SPECIAL.get(c, c) for c in t)


def inline(t):
    """Emphasis and code, after escaping. Order matters: code last, so its
    contents are not re-processed."""
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)          # links -> text
    parts, out = re.split(r"(`[^`]+`)", t), []
    for p in parts:
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(r"\texttt{%s}" % esc(p[1:-1]))
        else:
            p = esc(p)
            # non-greedy, because bold spans can contain a literal * --
            # "**ck = alpha * k**" was coming through as raw markdown
            p = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", p, flags=re.S)
            p = re.sub(r"(?<![\*\\])\*([^*]+?)\*(?!\*)", r"\\emph{\1}", p)
            out.append(p)
    return "".join(out)


def table(rows):
    """A pipe table, minus its |---| rule row. First row is the header."""
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    n = max(len(c) for c in cells)
    cells = [c + [""] * (n - len(c)) for c in cells]
    body = [c for c in cells if not all(re.fullmatch(r":?-{2,}:?", x) or x == ""
                                        for x in c)]
    if not body:
        return ""
    w = "@{}p{%.3f\\linewidth}@{}" % (0.97 / n)
    o = [r"\begingroup\small\setlength{\tabcolsep}{4pt}",
         r"\begin{tabularx}{\linewidth}{@{}%s@{}}" % ("Y" * n), r"\toprule"]
    o.append(" & ".join(inline(x) for x in body[0]) + r" \\")
    o.append(r"\midrule")
    for row in body[1:]:
        o.append(" & ".join(inline(x) for x in row) + r" \\")
    o += [r"\bottomrule", r"\end{tabularx}\endgroup", ""]
    return "\n".join(o)


def md2tex(md, figures_by_section=None):
    out, i, lines = [], 0, md.split("\n")
    bullets = quoting = False

    def close():
        nonlocal bullets, quoting
        if bullets:
            out.append(r"\end{itemize}"); bullets = False
        if quoting:
            out.append(r"\end{quote}"); quoting = False

    while i < len(lines):
        ln = lines[i]
        if re.match(r"^\|", ln):                       # table
            close()
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i]); i += 1
            out.append(table(block)); continue
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            close()
            lvl, txt = len(m.group(1)), m.group(2)
            cmd = ["section", "subsection", "subsubsection", "paragraph"][lvl - 1]
            out.append("\\%s{%s}" % (cmd, inline(txt)))
            if figures_by_section:                     # place figures by number
                num = re.match(r"^(\d+(?:\.\d+)*)", txt.strip())
                if num:
                    for f in figures_by_section.get(num.group(1), []):
                        out.append(f)
            i += 1; continue
        if ln.startswith(">"):
            if not quoting:
                close(); out.append(r"\begin{quote}\small"); quoting = True
            out.append(inline(ln.lstrip("> ").rstrip())); i += 1; continue
        m = re.match(r"^[-*]\s+(.*)$", ln)
        if m:
            if not bullets:
                close(); out.append(r"\begin{itemize}[leftmargin=1.1em,itemsep=1pt]")
                bullets = True
            out.append(r"\item " + inline(m.group(1))); i += 1; continue
        if not ln.strip():
            close(); out.append(""); i += 1; continue
        close(); out.append(inline(ln)); i += 1
    close()
    return "\n".join(out)


def docstring(path):
    src = open(path).read()
    m = re.search(r'^"""(.*?)"""', src, re.S | re.M)
    return m.group(1).strip() if m else ""


# ── figure catalogue, from figures/README.md ──────────────────────────────
cat = []
for ln in open(os.path.join(FIGS, "README.md")):
    m = re.match(r"^\|\s*`(fig_\w+)\.py`\s*\|\s*(.*?)\s*\|\s*([\d.]+(?:,\s*[\d.]+)*)\s*\|",
                 ln)
    if m:
        cat.append(dict(script=m.group(1), what=m.group(2),
                        secs=[s.strip() for s in m.group(3).split(",")]))
assert cat, "no figures parsed from figures/README.md"

by_sec = {}
for f in cat:
    pdf = os.path.join(OUT, f["script"].replace("fig_", "fig_") + ".pdf")
    if not os.path.exists(pdf):
        continue
    blk = ("\n" + r"\begin{figure}[htbp]\centering" + "\n"
           r"\includegraphics[width=\linewidth]{%s}" % pdf + "\n"
           r"\caption{\textbf{%s.} %s \emph{Full caption material and every"
           r" caveat: Part~II, \texttt{%s.py}.}}" % (
               f["script"].replace("_", r"\_"), inline(f["what"]),
               f["script"].replace("_", r"\_")) + "\n"
           r"\end{figure}" + "\n")
    by_sec.setdefault(f["secs"][0], []).append(blk)

# ── document ──────────────────────────────────────────────────────────────
skel = md2tex(open(SKEL).read(), by_sec)

ref = []
for f in cat:
    pdf = os.path.join(OUT, f["script"] + ".pdf")
    if not os.path.exists(pdf):
        continue
    ds = docstring(os.path.join(FIGS, f["script"] + ".py"))
    ref.append(r"\clearpage\subsection{\texttt{%s}\quad\normalfont\small %s}"
               % (f["script"].replace("_", r"\_"), inline(f["what"])))
    ref.append(r"\noindent\includegraphics[width=\linewidth]{%s}" % pdf)
    ref.append(r"\vspace{4pt}")
    ref.append(r"{\footnotesize\begin{verbatim}")
    ref.append(ds.replace("\\end{verbatim}", ""))
    ref.append(r"\end{verbatim}}")

TEX = r"""\documentclass[10pt,a4paper]{article}
\usepackage{fontspec}
\setmainfont{Times New Roman}[Scale=1.0]
\setmonofont{Menlo}[Scale=0.72]
\usepackage[margin=22mm]{geometry}
\usepackage{graphicx,booktabs,tabularx,enumitem,microtype}
\usepackage[hidelinks]{hyperref}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\usepackage{titlesec}
%% The skeleton numbers its own sections (3.1, 4.2, 6.3) and LaTeX numbering
%% on top of that produced "1.3.1 3.1 Design Overview". The document's numbers
%% are the paper's, so LaTeX contributes none.
\setcounter{secnumdepth}{-1}
\setcounter{tocdepth}{2}
\titleformat{\section}{\large\bfseries}{}{0em}{}
\titlespacing*{\section}{0pt}{14pt}{4pt}
\titleformat{\subsection}{\normalsize\bfseries}{}{0em}{}
\usepackage{fancyhdr}\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\footnotesize JHQ-GPU / ADC 2026 --- drafting handbook}
\fancyhead[R]{\footnotesize\thepage}\renewcommand{\headrulewidth}{0.2pt}
\begin{document}
\begin{center}
{\LARGE\bfseries JHQ-GPU: drafting handbook}\\[3pt]
{\large Sections 3, 4 and 6 --- skeleton, and every figure with its evidence}\\[8pt]
{\small branch \texttt{fix/recall-eval-v15} \quad---\quad generated %s \quad---\quad %d figures}
\end{center}
\vspace{4pt}
\begin{quote}\small
\textbf{How to use this.} Part~I is the drafting skeleton with each figure
placed at its section, so the prose and the evidence sit together. Part~II is
the figure reference: every figure at full width followed by its script's
docstring, which is where the measured numbers, the excluded rows and the
noise floors are written down. \textbf{Write captions from Part~II, not from
Part~I} --- the one-line descriptions in Part~I are labels, not claims.

Numbers move. \texttt{paper\_adc2026/README.md} is the evidence map and is
authoritative; the ``Superseded measurements'' section of the skeleton lists
what has already been withdrawn. Regenerate with
\texttt{python3 build\_handbook.py} after \texttt{figures/make\_all.sh}.
\end{quote}
\tableofcontents
\clearpage
\part*{Part I --- the skeleton, with the figures}
\addcontentsline{toc}{section}{Part I --- the skeleton, with the figures}
%s
\clearpage
\part*{Part II --- figure reference}
\addcontentsline{toc}{section}{Part II --- figure reference}
Each figure at full width, then its script's docstring verbatim: what it
measures, which runs are in it, which are excluded and why, and which noise
floor applies. This is caption material.
%s
\end{document}
""" % (__import__("datetime").date.today().isoformat(), len(by_sec) and
       sum(len(v) for v in by_sec.values()) or len(cat),
       skel, "\n".join(ref))

tex = os.path.join(HERE, "handbook.tex")
open(tex, "w").write(TEX)
for pas in (1, 2):
    r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                        "-output-directory", HERE, tex],
                       capture_output=True, text=True)
    if r.returncode != 0:
        err = [l for l in r.stdout.split("\n") if l.startswith("!")][:6]
        print("xelatex failed on pass %d:" % pas); print("\n".join(err) or r.stdout[-1500:])
        sys.exit(1)
print("wrote handbook.pdf  (%d figures placed, %d in the reference)"
      % (sum(len(v) for v in by_sec.values()), len(cat)))
