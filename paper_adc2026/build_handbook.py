#!/usr/bin/env python3
"""Build one PDF a co-author can draft from: the skeleton, with the figures.

    python3 build_handbook.py          ->  handbook.pdf     (English)
    python3 build_handbook.py --zh     ->  handbook_zh.pdf  (Chinese)

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

The Chinese edition is not a machine translation of the English one. It reads
its own skeleton (`..._v3_zh.md`) and its own figure notes
(`figures/NOTES_zh.md`), because the two documents want different things from
Part II: the English one reproduces each script's docstring verbatim, which is
the authoritative record and should not be retyped in another language, while
the Chinese one carries a distilled note per figure -- what it measures, the
numbers, and the thing a caption must not get wrong. The paper itself is
written in English, so both editions keep every identifier, path and number in
its original form.

No pandoc on this machine; the converter below handles the subset the
skeleton uses (headings, pipe tables, blockquotes, bullets, inline emphasis
and code) and nothing else. It is deliberately small: if the skeleton grows a
construct it does not know, the right fix is to add it here rather than to
hand-edit generated TeX.
"""
import os, re, subprocess, sys, glob

ZH   = "--zh" in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
OUT  = os.path.join(FIGS, "out")
SKEL = os.path.join(HERE, "ADC_sections_3_4_6_mother_skeleton_v3%s.md"
                    % ("_zh" if ZH else ""))
NAME = "handbook_zh" if ZH else "handbook"

T = dict(
  title      = ("JHQ-GPU: drafting handbook", "JHQ-GPU:写作手册"),
  sub        = ("Sections 3, 4 and 6 --- skeleton, and every figure with its evidence",
                "第 3、4、6 节 —— 骨架,以及每张图连同它的证据"),
  meta       = ("branch \\texttt{fix/recall-eval-v15} \\quad---\\quad generated %s \\quad---\\quad %d figures",
                "分支 \\texttt{fix/recall-eval-v15} \\quad—\\quad 生成于 %s \\quad—\\quad %d 张图"),
  p1         = ("Part I --- the skeleton, with the figures",
                "第一部分 —— 骨架,图在各自的章节"),
  p2         = ("Part II --- figure reference", "第二部分 —— 图鉴"),
  p2lead     = ("Each figure at full width, then its script's docstring verbatim: what it "
                "measures, which runs are in it, which are excluded and why, and which noise "
                "floor applies. This is caption material.",
                "每张图整页宽,底下是它的中文说明:测的是什么、关键数字、"
                "以及写图注时不能说错的地方。英文原始 docstring 见 "
                "\\texttt{handbook.pdf} 的第二部分。"),
  how        = (r"""\textbf{How to use this.} Part~I is the drafting skeleton with each figure
placed at its section, so the prose and the evidence sit together. Part~II is
the figure reference: every figure at full width followed by its script's
docstring, which is where the measured numbers, the excluded rows and the
noise floors are written down. \textbf{Write captions from Part~II, not from
Part~I} --- the one-line descriptions in Part~I are labels, not claims.

Numbers move. \texttt{paper\_adc2026/README.md} is the evidence map and is
authoritative; the ``Superseded measurements'' section of the skeleton lists
what has already been withdrawn. Regenerate with
\texttt{python3 build\_handbook.py} after \texttt{figures/make\_all.sh}.""",
                r"""\textbf{怎么用。} 第一部分是写作骨架,每张图放在它所属的章节,正文和证据在同一页。
第二部分是图鉴:每张图整页宽,底下跟着它的中文说明 ——
测量的数字、被排除的运行、以及适用的噪声地板都在那里。
\textbf{图注从第二部分写,不要从第一部分写}:第一部分里那些一行描述是标签,不是主张。

\textbf{论文正文用英文写。} 本中文版是对照参考;数字、路径、参数名一律保持原样。
两版如有出入,以 \texttt{handbook.pdf} 和 \texttt{paper\_adc2026/README.md}(证据表)为准。
骨架里那节「已被推翻的测量」列出了已经撤回的说法。

重新生成:先跑 \texttt{figures/make\_all.sh},再跑
\texttt{python3 build\_handbook.py --zh}。"""),
  cap        = ("Full caption material and every caveat: Part~II, \\texttt{%s.py}.",
                "完整说明与全部注意事项见第二部分 \\texttt{%s.py}。"),
)
def t(k):
    return T[k][1 if ZH else 0]

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
           r"\caption{\textbf{%s.} %s \emph{%s}}" % (
               f["script"].replace("_", r"\_"), inline(f["what"]),
               t("cap") % f["script"].replace("_", r"\_")) + "\n"
           r"\end{figure}" + "\n")
    by_sec.setdefault(f["secs"][0], []).append(blk)

# ── document ──────────────────────────────────────────────────────────────
skel = md2tex(open(SKEL).read(), by_sec)

zh_notes = {}
if ZH:
    cur = None
    for ln in open(os.path.join(FIGS, "NOTES_zh.md")):
        m = re.match(r"^##\s+(fig_\w+)\s*$", ln.strip())
        if m:
            cur = m.group(1); zh_notes[cur] = []
        elif cur and not ln.startswith("# "):
            zh_notes[cur].append(ln.rstrip())
    zh_notes = {k: "\n".join(v).strip() for k, v in zh_notes.items()}
    missing = [f["script"] for f in cat if f["script"] not in zh_notes]
    assert not missing, "NOTES_zh.md is missing: %s" % missing

ref = []
for f in cat:
    pdf = os.path.join(OUT, f["script"] + ".pdf")
    if not os.path.exists(pdf):
        continue
    ref.append(r"\clearpage\subsection{\texttt{%s}\quad\normalfont\small %s}"
               % (f["script"].replace("_", r"\_"), inline(f["what"])))
    ref.append(r"\noindent\includegraphics[width=\linewidth]{%s}" % pdf)
    ref.append(r"\vspace{4pt}")
    if ZH:
        ref.append(md2tex(zh_notes[f["script"]]))
    else:
        # verbatim, so the authoritative record is reproduced and not retyped
        ds = docstring(os.path.join(FIGS, f["script"] + ".py"))
        ref.append(r"{\footnotesize\begin{verbatim}")
        ref.append(ds.replace("\\end{verbatim}", ""))
        ref.append(r"\end{verbatim}}")

CJK = r"""\usepackage[UTF8,fontset=none]{ctex}
\setCJKmainfont{Songti SC}
\setCJKsansfont{PingFang SC}
\setCJKmonofont{PingFang SC}
\XeTeXlinebreaklocale "zh"\XeTeXlinebreakskip=0pt plus 1pt
""" if ZH else ""

TEX = r"""\documentclass[10pt,a4paper]{article}
\usepackage{fontspec}
\setmainfont{Times New Roman}[Scale=1.0]
\setmonofont{Menlo}[Scale=0.72]
__CJK__
\usepackage[margin=22mm]{geometry}
\usepackage{graphicx,booktabs,tabularx,enumitem,microtype}
\usepackage[hidelinks]{hyperref}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\usepackage{titlesec}
% The skeleton numbers its own sections (3.1, 4.2, 6.3) and LaTeX numbering on
% top of that produced "1.3.1 3.1 Design Overview". The document's numbers are
% the paper's, so LaTeX contributes none.
\setcounter{secnumdepth}{-1}
\setcounter{tocdepth}{2}
\titleformat{\section}{\large\bfseries}{}{0em}{}
\titlespacing*{\section}{0pt}{14pt}{4pt}
\titleformat{\subsection}{\normalsize\bfseries}{}{0em}{}
\usepackage{fancyhdr}\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\footnotesize __RUNHEAD__}
\fancyhead[R]{\footnotesize\thepage}\renewcommand{\headrulewidth}{0.2pt}
\begin{document}
\begin{center}
{\LARGE\bfseries __TITLE__}\\[3pt]
{\large __SUB__}\\[8pt]
{\small __META__}
\end{center}
\vspace{4pt}
\begin{quote}\small
__HOW__
\end{quote}
\tableofcontents
\clearpage
\part*{__P1__}
\addcontentsline{toc}{section}{__P1__}
__SKEL__
\clearpage
\part*{__P2__}
\addcontentsline{toc}{section}{__P2__}
__P2LEAD__
__REF__
\end{document}
"""

for k, v in [("__CJK__", CJK),
             ("__RUNHEAD__", "JHQ-GPU / ADC 2026 --- " +
              ("写作手册" if ZH else "drafting handbook")),
             ("__TITLE__", t("title")), ("__SUB__", t("sub")),
             ("__META__", t("meta") % (
                 __import__("datetime").date.today().isoformat(), len(cat))),
             ("__HOW__", t("how")),
             ("__P1__", t("p1")), ("__P2__", t("p2")),
             ("__P2LEAD__", t("p2lead")),
             ("__SKEL__", skel), ("__REF__", "\n".join(ref))]:
    TEX = TEX.replace(k, v)

tex = os.path.join(HERE, NAME + ".tex")
open(tex, "w").write(TEX)
for pas in (1, 2):
    r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                        "-output-directory", HERE, tex],
                       capture_output=True, text=True)
    if r.returncode != 0:
        err = [l for l in r.stdout.split("\n") if l.startswith("!")][:6]
        print("xelatex failed on pass %d:" % pas); print("\n".join(err) or r.stdout[-1500:])
        sys.exit(1)
print("wrote %s.pdf  (%d figures placed, %d in the reference)"
      % (NAME, sum(len(v) for v in by_sec.values()), len(cat)))
