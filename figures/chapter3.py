import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

OUT = "."
plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, BLACK, GRAY, COEF = "#1f6fb4", "#e67e22", "#222222", "#8a8a8a", "#6b7280"
N, K = 8, 5
X = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
INTER = [1, 3, 5]
CONCAT = [5, 6, 7]


def dct_matrix(n):
    c = np.array([[np.cos(np.pi * (2 * j + 1) * k / (2 * n)) for j in range(n)] for k in range(n)])
    c *= np.sqrt(2.0 / n)
    c[0] /= np.sqrt(2.0)
    return c


C = dct_matrix(N)


def padding_matrix(pad):
    valid = [i for i in range(N) if i not in pad]
    c10, c11 = C[K:][:, valid], C[K:][:, pad]
    return -np.linalg.inv(c11) @ c10, valid


def completed(x, pad):
    P, valid = padding_matrix(pad)
    full = np.zeros(N)
    full[valid] = x
    full[pad] = P @ x
    return full


def curve(full, t):
    y = C @ full
    out = np.zeros_like(t)
    for k in range(K):
        a = np.sqrt((1.0 if k == 0 else 2.0) / N)
        out += y[k] * a * np.cos(np.pi * (2 * t + 1) * k / (2 * N))
    return out


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def tint(color, a):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(1 - a * (1 - rgb))


SHADE = {("v", "L"): tint(BLUE, 0.22), ("p", "L"): tint(ORANGE, 0.30),
         ("v", "H"): tint(BLUE, 0.62), ("p", "H"): tint(ORANGE, 0.80)}


def fig_matrix():
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.75))
    for ax, pad, title in zip(axes, (CONCAT, INTER),
                              ("(a) Padding at the end", "(b) Padding interlaced")):
        kinds = ["p" if j in pad else "v" for j in range(N)]
        names, iv, ip = [], 0, 0
        for kind in kinds:
            if kind == "v":
                names.append("$x_%d$" % iv); iv += 1
            else:
                names.append("$p_%d$" % ip); ip += 1
        for j in range(N):
            col = BLUE if kinds[j] == "v" else ORANGE
            ax.add_patch(Rectangle((j, -1.75), 1, 1, facecolor=col, edgecolor="white", lw=1.2))
            ax.text(j + 0.5, -1.25, names[j], ha="center", va="center", color="white", fontsize=8)
            ax.text(j + 0.5, -2.05, str(j), ha="center", va="bottom", fontsize=6.5, color=GRAY)
        for i in range(N):
            for j in range(N):
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=SHADE[(kinds[j], "L" if i < K else "H")],
                                       edgecolor="white", lw=1.2))
        ax.plot([0, N], [K, K], color=BLACK, lw=0.9, ls="--")
        if pad == CONCAT:
            for (cx, cy, lab) in ((K / 2, K / 2, "$\\mathbf{C}_{00}$"), (K + (N - K) / 2, K / 2, "$\\mathbf{C}_{01}$"),
                                  (K / 2, K + (N - K) / 2, "$\\mathbf{C}_{10}$"),
                                  (K + (N - K) / 2, K + (N - K) / 2, "$\\mathbf{C}_{11}$")):
                ax.text(cx, cy, lab, ha="center", va="center", fontsize=11, color=BLACK)
        ax.text(N + 0.55, N / 2, "=", ha="center", va="center", fontsize=11)
        for i in range(N):
            if i < K:
                ax.add_patch(Rectangle((N + 1.1, i), 1, 1, facecolor=COEF, edgecolor="white", lw=1.2))
                ax.text(N + 1.6, i + 0.5, "$y_%d$" % i, ha="center", va="center", color="white", fontsize=8)
            else:
                ax.add_patch(Rectangle((N + 1.1, i), 1, 1, facecolor="white", edgecolor=GRAY, lw=0.6, hatch="////"))
                ax.text(N + 1.6, i + 0.5, "0", ha="center", va="center", fontsize=8,
                        bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))
        ax.text(N + 2.3, K / 2, "$\\mathbf{y}_\\mathrm{L}$\nkept", ha="left", va="center", fontsize=8)
        ax.text(N + 2.3, K + (N - K) / 2, "$\\mathbf{y}_\\mathrm{H}$\nforced\nto zero", ha="left", va="center",
                fontsize=8)
        if pad == CONCAT:
            ax.text(-0.4, K / 2, "low-frequency\nrows of $\\mathbf{C}$", rotation=90, ha="right", va="center",
                    fontsize=7.5, linespacing=1.0, multialignment="center")
            ax.text(-0.4, K + (N - K) / 2, "high-freq.\nrows of $\\mathbf{C}$", rotation=90, ha="right", va="center",
                    fontsize=7.5, linespacing=1.0, multialignment="center")
            ax.text(-0.4, -1.25, "input", ha="right", va="center", fontsize=7.5)
        ax.set_title(title, fontsize=9, pad=6, loc="left")
        ax.set_xlim(-2.2, N + 4.9)
        ax.set_ylim(N + 0.15, -2.6)
        ax.set_aspect("equal")
        ax.axis("off")
    yl, yh = "$\\mathbf{y}_\\mathrm{L}$", "$\\mathbf{y}_\\mathrm{H}$"
    handles = [Patch(facecolor=SHADE[("v", "L")], label="$\\mathbf{C}_{00}$: multiplies the valid samples, gives " + yl),
               Patch(facecolor=SHADE[("p", "L")], label="$\\mathbf{C}_{01}$: multiplies the padding, gives " + yl),
               Patch(facecolor=SHADE[("v", "H")], label="$\\mathbf{C}_{10}$: multiplies the valid samples, gives " + yh),
               Patch(facecolor=SHADE[("p", "H")], label="$\\mathbf{C}_{11}$: multiplies the padding, gives " + yh)]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=7.5, handlelength=1.2,
               columnspacing=1.6, bbox_to_anchor=(0.5, -0.16))
    fig.subplots_adjust(wspace=0.0, left=0.02, right=0.98, top=0.9, bottom=0.04)
    save(fig, "fig_matrix_blocks")


def draw_block(ax, x0, y0, state, s=1.0):
    rows, cols = len(state), len(state[0])
    for i in range(rows):
        for j in range(cols):
            st = state[i][j]
            if st == "z":
                ax.add_patch(Rectangle((x0 + j * s, y0 + i * s), s, s, facecolor="white", edgecolor=GRAY, lw=0.4,
                                       hatch="/////"))
            else:
                col = {"v": BLUE, "p": ORANGE, "c": COEF}[st]
                ax.add_patch(Rectangle((x0 + j * s, y0 + i * s), s, s, facecolor=col, edgecolor="white", lw=0.7))
    return x0 + cols * s, y0 + rows * s


def grid(rows, cols, fn):
    return [[fn(i, j) for j in range(cols)] for i in range(rows)]


def fig_block_shape():
    P = set(INTER)
    fig, ax = plt.subplots(figsize=(6.3, 3.75))
    gap = 5.8
    stages_a = [
        (grid(K, K, lambda i, j: "v"), "$5 \\times 5$\nsamples", None),
        (grid(K, N, lambda i, j: "p" if j in P else "v"), "$5 \\times 8$", "pad\nalong $u$"),
        (grid(K, N, lambda i, j: "c" if j < K else "z"), "$5 \\times 8$", "DCT\nalong $u$"),
        (grid(N, K, lambda i, j: "p" if i in P else "c"), "$8 \\times 5$", "discard\nzeros, pad\nalong $v$"),
        (grid(N, K, lambda i, j: "c" if i < K else "z"), "$8 \\times 5$", "DCT\nalong $v$"),
    ]
    stages_b = [
        (grid(K, K, lambda i, j: "v"), "$5 \\times 5$\nsamples", None),
        (grid(N, N, lambda i, j: "p" if (i in P or j in P) else "v"), "$8 \\times 8$", "pad the\nwhole\nblock"),
        (grid(N, N, lambda i, j: "c" if j < K else "z"), "$8 \\times 8$", "DCT\nalong $u$"),
        (grid(N, N, lambda i, j: "c" if (i < K and j < K) else "z"), "$8 \\times 8$", "DCT\nalong $v$"),
    ]
    ends = []
    for stages, y0, title in ((stages_a, 0.0, "(a) Padding before the DCT of each dimension (this work)"),
                              (stages_b, 14.6, "(b) Padding of the whole block first")):
        x = 0.0
        ax.text(0, y0 - 2.9, title, fontsize=8.5, ha="left", va="bottom", weight="bold")
        for idx, (state, size, step) in enumerate(stages):
            if step is not None:
                ax.add_patch(FancyArrowPatch((x + 0.7, y0 + 2.5), (x + gap - 0.7, y0 + 2.5), arrowstyle="-|>",
                                             mutation_scale=8, lw=0.8, color=BLACK))
                ax.text(x + gap / 2, y0 + 1.9, step, ha="center", va="bottom", fontsize=6.5, linespacing=1.0)
                x += gap
            x1, y1 = draw_block(ax, x, y0, state)
            ax.text((x + x1) / 2, y1 + 0.35, size, ha="center", va="top", fontsize=7.5, linespacing=1.0)
            if idx == len(stages) - 1:
                ax.add_patch(Rectangle((x, y0), K, K, fill=False, edgecolor=BLACK, lw=1.6))
                ends.append((x + K, y0 + K / 2, x1))
            x = x1
    xr = max(e[2] for e in ends) + 1.2
    for (xe, ye, x1) in ends:
        ax.plot([x1 + 0.3, xr], [ye, ye], color=BLACK, lw=0.7)
    ax.plot([xr, xr], [ends[0][1], ends[1][1]], color=BLACK, lw=0.7)
    ax.text(xr + 0.4, (ends[0][1] + ends[1][1]) / 2, "the same\n25 coefficients", ha="left", va="center", fontsize=8)
    handles = [Patch(facecolor=BLUE, label="valid sample"), Patch(facecolor=ORANGE, label="padding value"),
               Patch(facecolor=COEF, label="transformed value"),
               Patch(facecolor="white", edgecolor=GRAY, hatch="/////", label="zero coefficient")]
    ax.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8, handlelength=1.2,
              bbox_to_anchor=(0.5, -0.08), columnspacing=1.6)
    ax.set_xlim(-0.5, xr + 8.5)
    ax.set_ylim(24.8, -4.4)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "fig_two_orders")


def solve(n, k, x, pad):
    c = dct_matrix(n)
    valid = [i for i in range(n) if i not in pad]
    p = -np.linalg.inv(c[k:][:, pad]) @ c[k:][:, valid] @ x
    full = np.zeros(n)
    full[valid] = x
    full[pad] = p
    y = c @ full

    def cur(t):
        out = np.zeros_like(t)
        for q in range(k):
            out += y[q] * np.sqrt((1.0 if q == 0 else 2.0) / n) * np.cos(np.pi * (2 * t + 1) * q / (2 * n))
        return out
    return valid, full, cur


def thousands(v, _pos=None):
    return ("%d" % round(v)).replace("-", "−") if abs(v) < 1000 else \
        ("{:,.0f}".format(v)).replace(",", " ").replace("-", "−")


def sci(v):
    if abs(v) < 1000:
        return ("$%.1f$" % v).replace("-", "{-}")
    e = int(np.floor(np.log10(abs(v))))
    return "$%s%.1f \\times 10^{%d}$" % ("-" if v < 0 else "", abs(v) / 10 ** e, e)


def fig_growth():
    cases = [(8, 5, [5, 6, 7], "(a)", "$N = 8$, $K = 5$, concatenated", False),
             (16, 9, list(range(9, 16)), "(b)", "$N = 16$, $K = 9$, concatenated", True),
             (16, 9, [1, 3, 5, 7, 9, 11, 13], "(c)", "$N = 16$, $K = 9$, interlaced", False)]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.6))
    LIN = 30.0
    for ax, (n, k, pad, tag, title, log) in zip(axes, cases):
        x = 2.0 * np.arange(1, k + 1)
        valid, full, cur = solve(n, k, x, pad)
        t = np.linspace(-0.5, n - 0.5, 1200)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e6e6e6", lw=0.5)
        ax.axhline(0, color=GRAY, lw=0.5)
        ax.plot(t, cur(t), color=GRAY, lw=1.1, zorder=2)
        ax.scatter(valid, x, s=20, color=BLUE, zorder=3, edgecolor="white", linewidth=0.6)
        ax.scatter(pad, full[pad], s=22, marker="s", color=ORANGE, zorder=3, edgecolor="white", linewidth=0.6)
        j = int(np.argmax(np.abs(full[pad])))
        ext, xe = full[pad][j], pad[j]
        if log:
            ax.set_yscale("symlog", linthresh=LIN, linscale=1.1)
            ax.set_ylim(-3e5, LIN)
            ax.set_yticks([20, 0, -20, -1e2, -1e3, -1e4, -1e5])
            ax.set_yticklabels(["$20$", "$0$", "$-20$", "$-10^{2}$", "$-10^{3}$", "$-10^{4}$", "$-10^{5}$"])
            ax.yaxis.set_minor_locator(matplotlib.ticker.SymmetricalLogLocator(base=10, linthresh=LIN,
                                                                                subs=range(2, 10)))
            ax.tick_params(axis="y", which="minor", length=1.5, width=0.4)
            ax.axhspan(-LIN, LIN, color="#f2f2f2", lw=0, zorder=0)
            ax.set_ylabel("sample value (symmetric log axis)", fontsize=8, labelpad=1)
            ax.annotate(sci(ext), xy=(xe, ext), xytext=(xe - 4.0, -1.9e5), fontsize=7.5, ha="right", va="center",
                        arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.5, shrinkA=1, shrinkB=3))
        else:
            lo, hi = min(full.min(), cur(t).min()), max(full.max(), cur(t).max())
            span = hi - lo
            ax.set_ylim(lo - 0.08 * span, hi + 0.10 * span)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, integer=True))
            if ext < 0:
                ax.annotate(sci(ext), xy=(xe, ext), xytext=(-5, 4), textcoords="offset points", fontsize=7.5,
                            ha="right", va="bottom")
        ax.text(0.0, 1.04, tag, transform=ax.transAxes, fontsize=9, weight="bold", ha="left", va="bottom")
        ax.text(0.17, 1.045, title, transform=ax.transAxes, fontsize=7.5, ha="left", va="bottom")
        ax.set_xticks(range(0, n, 1 if n <= 8 else 3))
        ax.set_xlabel("position $n$", fontsize=8)
        ax.tick_params(labelsize=7.5, length=2.5, width=0.6)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(0.6)
        ax.set_xlim(-0.7, n - 0.3)
    axes[0].set_ylabel("sample value", fontsize=8)
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, markersize=4.5, label="valid sample"),
               Line2D([], [], marker="s", ls="", color=ORANGE, markersize=4.5, label="padding value"),
               Line2D([], [], color=GRAY, lw=1.1, label="curve of the $K$ lowest-frequency cosines")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01), columnspacing=1.8, handletextpad=0.4)
    fig.subplots_adjust(wspace=0.46, bottom=0.27, top=0.88, left=0.075, right=0.985)
    save(fig, "fig_padding_growth")


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("output", help="output folder")
    OUT = ap.parse_args().output
    os.makedirs(OUT, exist_ok=True)
    fig_matrix()
    fig_block_shape()
    fig_growth()


if __name__ == "__main__":
    main()
