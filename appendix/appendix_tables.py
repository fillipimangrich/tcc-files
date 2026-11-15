#!/usr/bin/env python3
"""Generate Appendix C of the thesis from the lookup tables of Padding.cpp.

The script parses the interlacing orders, the padding matrices P and the
composite matrices T stored in Padding.cpp, verifies them, and writes the
LaTeX source of the appendix.

Usage:
    python3 appendix_tables.py <Padding.cpp> [appendix_c.tex]
"""

import os
import re
import sys

import numpy as np

DEFAULT_TEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "appendix_c.tex")

BLOCK_SIZES = (4, 8, 16, 32)

# Layout of the matrices.
MAX_COLS = 11            # maximum number of columns in one block
MATRIX_FONT = r"\tiny"   # 8 pt in the class of the thesis (memoir, 12 pt)
COL_TEMPLATE = "-0.00000"  # widest entry, used to set the width of the columns
COL_SEP = "2.2pt"          # half of the space between two columns

# Colors of the figure (the same as in the other figures of the thesis).
COLOR_VALID = "1F6FB4"
COLOR_PADDING = "E67E22"

NUMBER = r"-?\d+\.\d+"


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def parse_rows(body):
    """Split the initializer of a 2D array into rows of number tokens.

    The tokens are kept as strings, exactly as they are stored in the file.
    A flat initializer (without inner braces) is returned as a single row.
    """
    inner = re.findall(r"\{([^{}]*)\}", body)
    if not inner:
        inner = [body]
    return [re.findall(NUMBER, row) for row in inner]


def parse_tables(cpp_text):
    """Return {(n, k): {"order": [...], "P": [[str]], "T": [[str]]}}."""
    tables = {}

    def entry(n, k):
        return tables.setdefault((n, k), {})

    order_re = re.compile(
        r"PaddingManager::n(\d+)_v(\d+)_order\[(\d+)\]\s*=\s*\{([^}]*)\}\s*;")
    for n, k, size, body in order_re.findall(cpp_text):
        n, k, size = int(n), int(k), int(size)
        order = [int(v) for v in re.findall(r"\d+", body)]
        entry(n, k)["order"] = order
        entry(n, k)["order_declared"] = size

    # The name must be followed directly by the dimensions, which excludes the
    # "_without_order" and "_composite" tables.
    padding_re = re.compile(
        r"PaddingManager::n(\d+)_v(\d+)\[(\d+)\]\[(\d+)\]\s*=\s*\{(.*?)\}\s*;", re.S)
    for n, k, rows, cols, body in padding_re.findall(cpp_text):
        n, k = int(n), int(k)
        entry(n, k)["P"] = parse_rows(body)
        entry(n, k)["P_declared"] = (int(rows), int(cols))

    composite_re = re.compile(
        r"static\s+const\s+double\s+n(\d+)_v(\d+)_composite\[(\d+)\]\[(\d+)\]\s*=\s*\{(.*?)\}\s*;",
        re.S)
    for n, k, rows, cols, body in composite_re.findall(cpp_text):
        n, k = int(n), int(k)
        entry(n, k)["T"] = parse_rows(body)
        entry(n, k)["T_declared"] = (int(rows), int(cols))

    return tables


def covered_pairs():
    """Pairs (N, K) documented in the appendix: N/2 < K < N."""
    return [(n, k) for n in BLOCK_SIZES for k in range(n // 2 + 1, n)]


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def dct_matrix(n):
    """Orthonormal N-point DCT-II matrix (row = coefficient, column = sample)."""
    i = np.arange(n).reshape(-1, 1)
    j = np.arange(n).reshape(1, -1)
    c = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * j + 1) * i / (2 * n))
    c[0, :] = np.sqrt(1.0 / n)
    return c


def shape_of(rows):
    lengths = {len(r) for r in rows}
    if len(lengths) != 1:
        raise ValueError("rows of different lengths: %s" % sorted(lengths))
    return (len(rows), lengths.pop())


def verify(tables):
    """Check the data of every covered pair. Return a dictionary of results.

    Raises AssertionError if any check fails, so that a wrong appendix is
    never written.
    """
    pairs = covered_pairs()
    max_dev_t = 0.0
    max_dev_p = 0.0
    max_dev_tp = 0.0
    p_exceptions = []
    for n, k in pairs:
        name = "n%d_v%d" % (n, k)
        assert (n, k) in tables, "missing tables of %s" % name
        tab = tables[(n, k)]
        for key in ("order", "P", "T"):
            assert key in tab, "missing %s of %s" % (key, name)

        # Interlacing order: N-K distinct positions in [0, N-1].
        order = tab["order"]
        assert tab["order_declared"] == n - k, "declared size of %s_order" % name
        assert len(order) == n - k, "length of %s_order" % name
        assert len(set(order)) == n - k, "repeated positions in %s_order" % name
        assert all(0 <= p <= n - 1 for p in order), "position out of range in %s_order" % name
        assert order == sorted(order), "%s_order is not increasing" % name

        # Shapes.
        assert tab["P_declared"] == (n - k, k), "declared shape of %s" % name
        assert shape_of(tab["P"]) == (n - k, k), "shape of %s" % name
        assert tab["T_declared"] == (k, k), "declared shape of %s_composite" % name
        assert shape_of(tab["T"]) == (k, k), "shape of %s_composite" % name

        # Stored precision (stated in the introduction of the appendix).
        # The rule of P is: seven characters, that is, five decimal places in
        # the positive entries and four in the negative ones. The entries that
        # do not follow it are collected and reported in the text.
        for row in tab["P"]:
            for tok in row:
                integer, decimals = tok.lstrip("-").split(".")
                assert len(integer) == 1, "%s: unexpected entry %s" % (name, tok)
                assert float(tok) != 0.0, "%s: zero entry %s" % (name, tok)
                assert len(decimals) in (4, 5), "%s: unexpected precision of %s" % (name, tok)
                if len(tok) != 7:
                    assert len(tok) == 8, "%s: unexpected entry %s" % (name, tok)
                    p_exceptions.append((n, k, tok))
        for row in tab["T"]:
            for tok in row:
                integer, decimals = tok.lstrip("-").split(".")
                assert len(integer) == 1 and len(decimals) == 5, \
                    "%s_composite: entry %s does not have 5 decimal places" % (name, tok)
                assert len(tok) <= len(COL_TEMPLATE), "%s_composite: entry %s is too wide" % (name, tok)

        # Reference matrices for the orthonormal DCT and the stored order.
        valid = [p for p in range(n) if p not in order]
        c = dct_matrix(n)
        c00 = c[:k][:, valid]
        c01 = c[:k][:, order]
        c10 = c[k:][:, valid]
        c11 = c[k:][:, order]
        t_ref = np.linalg.inv(c00.T)
        p_ref = -np.linalg.solve(c11, c10)

        t = np.array(tab["T"], dtype=float)
        p = np.array(tab["P"], dtype=float)

        # T = (C00^T)^-1 to within 1e-5.
        dev_t = float(np.max(np.abs(t - t_ref)))
        assert dev_t <= 1e-5, "%s_composite differs from (C00^T)^-1 by %.2e" % (name, dev_t)
        max_dev_t = max(max_dev_t, dev_t)

        # P = -C11^-1 C10 to within the stored precision (four decimal places
        # in the negative entries, which are truncated).
        dev_p = float(np.max(np.abs(p - p_ref)))
        assert dev_p <= 1e-4, "%s differs from -C11^-1 C10 by %.2e" % (name, dev_p)
        max_dev_p = max(max_dev_p, dev_p)

        # Consistency between the two tables: T = C00 + C01 P.
        dev_tp = float(np.max(np.abs(t - (c00 + c01 @ p))))
        assert dev_tp <= 1e-3, "%s: T differs from C00 + C01 P by %.2e" % (name, dev_tp)
        max_dev_tp = max(max_dev_tp, dev_tp)

    return {
        "pairs": len(pairs),
        "max_dev_t": max_dev_t,
        "max_dev_p": max_dev_p,
        "max_dev_tp": max_dev_tp,
        "p_exceptions": p_exceptions,
    }


# --------------------------------------------------------------------------
# LaTeX generation
# --------------------------------------------------------------------------

def split_columns(ncols, max_cols=MAX_COLS):
    """Split ncols columns into the fewest balanced blocks of at most max_cols."""
    nblocks = -(-ncols // max_cols)
    base, extra = divmod(ncols, nblocks)
    blocks = []
    start = 0
    for b in range(nblocks):
        width = base + (1 if b < extra else 0)
        blocks.append((start, start + width))
        start += width
    return blocks


def matrix_blocks(symbol, n, k, rows):
    """LaTeX lines of one matrix, split into blocks of columns."""
    nrows, ncols = len(rows), len(rows[0])
    blocks = split_columns(ncols)
    lines = []
    for first, last in blocks:
        label = r"%s for $\symN = %d$ and $\symK = %d$" % (symbol, n, k)
        if len(blocks) > 1:
            label += ", columns %d to %d" % (first, last - 1)
        lines.append(r"\noindent %s:" % label)
        lines.append(r"\apcblock{%d}" % (last - first))
        lines.append("  {%s}" % " & ".join(str(j) for j in range(first, last)))
        lines.append("  {%s}" % r" \\ ".join(str(i) for i in range(nrows)))
        lines.append("  {%")
        for i, row in enumerate(rows):
            end = r" \\" if i < nrows - 1 else ""
            lines.append("    " + " & ".join(row[first:last]) + end)
        lines.append("  }")
        lines.append("")
    return lines


def orders_table(tables):
    """Table of the orders, in two panels: N = 4, 8, 16 and N = 32."""

    def panel(sizes):
        rows = [
            r"\begin{tabular}[t]{ccl}",
            r"\hline",
            r"$\symN$ & $\symK$ & \textbf{Padding positions} \\",
            r"\hline",
        ]
        for n in sizes:
            if n != sizes[0]:
                rows.append(r"\hline")
            for k in range(n // 2 + 1, n):
                order = ", ".join(str(p) for p in tables[(n, k)]["order"])
                rows.append(r"%d & %d & %s \\" % (n, k, order))
        rows += [r"\hline", r"\end{tabular}"]
        return rows

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption[Interlacing orders stored in the reference software]"
        r"{Interlacing orders stored in the reference software: positions of the "
        r"$\symN-\symK$ padding samples in the completed vector, counted from 0.}",
        r"\label{tab:ap_orders}",
        r"\footnotesize",
    ]
    lines += panel(BLOCK_SIZES[:-1])
    lines.append(r"\hspace{1.5em}")
    lines += panel(BLOCK_SIZES[-1:])
    lines += [
        r"\fonte{the author.}",
        r"\end{table}",
    ]
    return lines


def orders_figure(tables):
    """TikZ figure with one strip of N cells for each pair."""
    cell = 3.6     # width and height of a cell (mm)
    pitch = 4.6    # vertical distance between strips (mm)
    label_w = 13.0  # room for the label "K = ..." (mm)
    gap = 7.0      # horizontal distance between groups (mm)

    lines = [
        r"\begin{figure}[!ht]",
        r"\centering",
        r"\caption[Interlacing orders of the completed vector]"
        r"{Interlacing orders of \cref{tab:ap_orders}: each row is a completed vector "
        r"of $\symN$ positions, with the valid samples in blue and the padding samples in orange.}",
        r"\label{fig:ap_orders}",
        r"\begin{tikzpicture}[x=1mm, y=1mm,",
        r"    cell/.style={draw, line width=0.3pt, minimum size=%.1fmm, inner sep=0pt, outer sep=0pt}," % cell,
        r"    valid/.style={cell, fill=apcvalid},",
        r"    padding/.style={cell, fill=apcpadding},",
        r"    idx/.style={font=\tiny, inner sep=0pt, anchor=base},",
        r"    lbl/.style={font=\footnotesize, inner sep=0pt, anchor=east}]",
    ]

    def group(n, x0, y0):
        """Draw the strips of block size n with the top-left corner at (x0, y0)."""
        out = []
        xs = x0 + label_w
        out.append(r"  \node[font=\footnotesize, anchor=base west, inner sep=0pt] at (%.2f, %.2f) {$\symN = %d$};"
                   % (x0, y0 + 5.0, n))
        for p in range(n):
            out.append(r"  \node[idx] at (%.2f, %.2f) {%d};" % (xs + (p + 0.5) * cell, y0 + 1.2, p))
        y = y0
        for k in range(n // 2 + 1, n):
            order = tables[(n, k)]["order"]
            yc = y - cell / 2
            out.append(r"  \node[lbl] at (%.2f, %.2f) {$\symK = %d$};" % (xs - 1.5, yc, k))
            for p in range(n):
                style = "padding" if p in order else "valid"
                out.append(r"  \node[%s] at (%.2f, %.2f) {};" % (style, xs + (p + 0.5) * cell, yc))
            y -= pitch
        return out, y

    # Upper part: N = 4, 8 and 16, side by side. Lower part: N = 32.
    x = 0.0
    bottom = 0.0
    for n in BLOCK_SIZES[:-1]:
        out, y_end = group(n, x, 0.0)
        lines += out
        bottom = min(bottom, y_end)
        x += label_w + n * cell + gap
    out, y_end = group(BLOCK_SIZES[-1], 0.0, bottom - 9.0)
    lines += out

    # Legend.
    yl = y_end - 3.0
    xl = label_w
    lines.append(r"  \node[valid] at (%.2f, %.2f) {};" % (xl + cell / 2, yl))
    lines.append(r"  \node[font=\footnotesize, anchor=west, inner sep=0pt] at (%.2f, %.2f) {valid sample};"
                 % (xl + cell + 1.5, yl))
    lines.append(r"  \node[padding] at (%.2f, %.2f) {};" % (xl + 32.0 + cell / 2, yl))
    lines.append(r"  \node[font=\footnotesize, anchor=west, inner sep=0pt] at (%.2f, %.2f) {padding sample};"
                 % (xl + 32.0 + cell + 1.5, yl))

    lines += [
        r"\end{tikzpicture}",
        r"\fonte{the author.}",
        r"\end{figure}",
    ]
    return lines


def block_macro():
    """Definition of the macro that typesets one block of columns.

    Arguments of the macro: number of columns, indices of the columns,
    indices of the rows, entries. All the columns have the width of the
    widest entry, so the indices stay aligned with the columns. The box is
    raised by its depth so that a block at the end of a page does not go
    below the text area.
    """
    return [
        r"\newlength{\apccolwidth}",
        r"\settowidth{\apccolwidth}{%s$%s$}" % (MATRIX_FONT, COL_TEMPLATE),
        r"\newcolumntype{V}{w{r}{\apccolwidth}}",
        r"\newcommand{\apcblock}[4]{%",
        r"  \[",
        r"  \raisebox{\depth}{\renewcommand{\baselinestretch}{1}%s\setlength{\arraycolsep}{%s}$" % (MATRIX_FONT, COL_SEP),
        r"  \begin{array}{@{}r@{\hspace{0.5em}}c@{}}",
        r"    & \begin{array}{@{}*{#1}{V}@{}} #2 \end{array} \\[0.4ex]",
        r"    \begin{array}{@{}r@{}} #3 \end{array} &",
        r"    \begin{bmatrix*}[V] #4 \end{bmatrix*}",
        r"  \end{array}$}",
        r"  \]}",
    ]


def precision_sentences(p_exceptions):
    """Sentences of the introduction about the stored precision of P."""
    rule = (r"Each of its entries is stored with seven characters, so the positive entries have five decimal places "
            r"and the negative entries have four.")
    return [rule]


def generate(tables, p_exceptions):
    pairs = covered_pairs()
    out = []
    out += [
        r"\chapter{Interlacing Orders, Padding Matrices, and Composite Matrices}",
        r"\label[appendix]{ap.tables}",
        "",
        r"\definecolor{apcvalid}{HTML}{%s}" % COLOR_VALID,
        r"\definecolor{apcpadding}{HTML}{%s}" % COLOR_PADDING,
        r"\setcounter{MaxMatrixCols}{32}",
        *block_macro(),
        "",
        r"This appendix lists the interlacing orders, the padding matrices $\matP$, and the composite matrices $\matT$ "
        r"stored in the file \texttt{Padding.cpp} of the modified \gls{jplm} (\cref{sec:method.implementation_details}).",
        r"They cover the %d pairs $(\symN, \symK)$ with $\symN \in \{4, 8, 16, 32\}$ and $\symN/2 < \symK < \symN$."
        % len(pairs),
        r"The interlacing order of a pair gives the positions of the $\symN-\symK$ padding samples in the completed vector, "
        r"counted from 0.",
        r"The padding matrix $\matP$ has $\symN-\symK$ rows and $\symK$ columns.",
        *precision_sentences(p_exceptions),
        r"The composite matrix $\matT$ has $\symK$ rows and $\symK$ columns, and its entries are stored with five decimal places.",
        r"It is given for the orthonormal \gls{dct}, and the encoder multiplies it by $\sqrt{\symN}$, since the \gls{dct} "
        r"of the \gls{jplm} is not normalized.",
        "",
    ]

    # Section 1: interlacing orders.
    out += [
        r"\section{Interlacing Orders}",
        "",
        r"\Cref{tab:ap_orders} lists the interlacing order of each pair $(\symN, \symK)$.",
        r"\Cref{fig:ap_orders} shows the same orders graphically.",
        r"\Cref{sec:method.order_search} describes how the orders were obtained.",
        "",
    ]
    out += orders_table(tables)
    out.append("")
    out += orders_figure(tables)
    out += ["", r"\FloatBarrier", ""]

    # Section 2: padding matrices.
    out += [
        r"\section{Padding Matrices}",
        "",
        r"This section lists the padding matrices $\matP$ defined in \cref{sec:method.math}.",
        r"Each row of $\matP$ gives one padding sample, in the order of the positions of \cref{tab:ap_orders}.",
        r"Each column multiplies one valid sample, in the order in which the valid samples appear in the vector.",
        r"The entries are printed exactly as they are stored.",
        r"The indices of the rows and of the columns are printed outside the brackets and are counted from 0.",
        r"A matrix with more than %d columns is divided into blocks of consecutive columns, "
        r"which are printed one below the other." % MAX_COLS,
        "",
        r"\medskip",
        "",
    ]
    for n, k in pairs:
        out += matrix_blocks(r"Matrix $\matP$", n, k, tables[(n, k)]["P"])

    # Section 3: composite matrices.
    out += [
        r"\section{Composite Matrices}",
        "",
        r"This section lists the composite matrices $\matT$ defined in \cref{sec:method.composite}.",
        r"Each row of $\matT$ gives one coefficient, in the order of increasing frequency.",
        r"Each column multiplies one valid sample, as in the padding matrices.",
        r"The matrices are printed in the same way as the padding matrices.",
        "",
        r"\medskip",
        "",
    ]
    for n, k in pairs:
        out += matrix_blocks(r"Matrix $\matT$", n, k, tables[(n, k)]["T"])

    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cpp_path = sys.argv[1]
    tex_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TEX

    with open(cpp_path, encoding="utf-8") as f:
        tables = parse_tables(f.read())

    res = verify(tables)
    print(
        "checks OK for %d pairs: orders (N-K distinct positions in [0, N-1]), shapes of P ((N-K) x K) and T (K x K), "
        "stored precision; max |T - (C00^T)^-1| = %.1e (limit 1e-5); max |P - (-C11^-1 C10)| = %.1e; "
        "max |T - (C00 + C01 P)| = %.1e; entries of P outside the 7-character rule: %s"
        % (res["pairs"], res["max_dev_t"], res["max_dev_p"], res["max_dev_tp"],
           ", ".join("n%d_v%d %s" % e for e in res["p_exceptions"]) or "none")
    )

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(generate(tables, res["p_exceptions"]))
    print("written: %s" % tex_path)


if __name__ == "__main__":
    main()
