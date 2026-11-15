import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm", "pdf.fonttype": 42})
RED, ORANGE, BLACK, BLUE = "#c0392b", "#e67e22", "#222222", "#1f6fb4"
B = 32
ORDER_32_18 = [2, 4, 6, 8, 10, 12, 14, 17, 19, 21, 23, 25, 27, 29]


def read_pgx(path):
    raw = open(path, "rb").read()
    header, data = raw.split(b"\n", 1)
    parts = header.decode().replace("+", " + ").split()
    w, h = int(parts[-2]), int(parts[-1])
    return np.frombuffer(data[:2 * w * h], dtype=">u2").reshape(h, w).astype(float)


def to_rgb(ycc):
    y, cb, cr = ycc[..., 0], ycc[..., 1] - 512, ycc[..., 2] - 512
    rgb = np.stack([y + 1.5748 * cr, y - 0.1873 * cb - 0.4681 * cr, y + 1.8556 * cb], axis=-1) / 1023
    return np.clip(rgb, 0, 1)


def dct_matrix(n):
    k, t = np.arange(n)[:, None], np.arange(n)[None, :]
    m = np.sqrt(2 / n) * np.cos(np.pi * (2 * t + 1) * k / (2 * n))
    m[0] /= np.sqrt(2)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="folder with the light fields in PGX (one subfolder per light field)")
    ap.add_argument("output", help="output folder")
    ap.add_argument("--column", type=int, default=16, help="block column of the boundary block")
    args = ap.parse_args()
    pgx = os.path.join(args.dataset, "Bikes")

    ycc = np.stack([read_pgx(f"{pgx}/{c}/005_005.pgx") for c in range(3)], axis=-1)
    view = to_rgb(ycc)
    H, W = view.shape[:2]
    Y0 = (H // B) * B
    KV = H - Y0
    x0 = args.column * B
    blk = ycc[Y0:H, x0:x0 + B]

    rep = np.concatenate([blk, np.repeat(blk[-1:], B - KV, axis=0)], axis=0)

    assert len(ORDER_32_18) == B - KV
    valid = [i for i in range(B) if i not in ORDER_32_18]
    C = dct_matrix(B)
    P = -np.linalg.solve(C[KV:][:, ORDER_32_18], C[KV:][:, valid])
    prop = np.empty((B, B, 3))
    prop[valid] = blk
    prop[ORDER_32_18] = np.einsum("ij,jkc->ikc", P, blk)

    coef_t = dct_matrix(KV) @ blk[..., 0] @ C.T
    coef_r = C @ rep[..., 0] @ C.T
    coef_p = C @ prop[..., 0] @ C.T
    coef_p[KV:] = 0.0
    VM = max(np.abs(c).max() for c in (coef_t, coef_r, coef_p))
    NORM = SymLogNorm(linthresh=10, linscale=0.6, vmin=-VM, vmax=VM, base=10)
    CMAP = "RdBu_r"

    FW, FH = 6.3, 4.25
    fig = plt.figure(figsize=(FW, FH))

    def ax_in(x, y, w, h):
        return fig.add_axes([x / FW, y / FH, w / FW, h / FH])

    p = 0.027
    xb, xc = 2.5, 4.75
    gap = 0.55
    y_c = 0.35
    y_b = y_c + B * p + gap
    y_a = y_b + B * p + gap

    vw = 2.2
    vh = vw * H / W
    yv = (y_c + y_a + KV * p) / 2 - vh / 2 + 0.15
    axv = ax_in(0.0, yv, vw, vh)
    axv.imshow(view, interpolation="lanczos")
    for x in range(0, W, B):
        axv.axvline(x - 0.5, color="white", lw=0.3, alpha=0.7)
    for y in range(0, H, B):
        axv.axhline(y - 0.5, color="white", lw=0.3, alpha=0.7)
    axv.add_patch(Rectangle((-0.5, Y0 - 0.5), W, KV, facecolor=ORANGE, alpha=0.6, edgecolor="none"))
    axv.add_patch(Rectangle((x0 - 0.5, Y0 - 0.5), B, KV, fill=False, edgecolor=RED, lw=1.8))
    axv.set_xlim(-0.5, W - 0.5)
    axv.set_ylim(H - 0.5, -0.5)
    axv.axis("off")
    fig.text(vw / 2 / FW, (yv - 0.08) / FH, "a view of Bikes, in blocks of $32 \\times 32$ pixels:\n"
             "$434 = 13 \\times 32 + 18$ rows", ha="center", va="top", fontsize=8)

    def panel(x, y, nrow, ncol):
        ax = ax_in(x, y, ncol * p, nrow * p)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
        return ax

    a1 = panel(xb, y_a, KV, B)
    a1.imshow(to_rgb(blk), interpolation="nearest")
    a2 = panel(xc, y_a, KV, B)
    a2.imshow(coef_t, cmap=CMAP, norm=NORM, interpolation="nearest")

    b1 = panel(xb, y_b, B, B)
    b1.imshow(to_rgb(rep), interpolation="nearest")
    b1.axhline(KV - 0.5, color=RED, lw=1.0, ls="--")
    b2 = panel(xc, y_b, B, B)
    b2.imshow(coef_r, cmap=CMAP, norm=NORM, interpolation="nearest")

    c1 = panel(xb, y_c, B, B)
    c1.imshow(to_rgb(prop), interpolation="nearest")
    for r in ORDER_32_18:
        c1.plot([B - 0.5 + 0.4, B - 0.5 + 2.2], [r, r], color=RED, lw=1.0, clip_on=False, solid_capstyle="butt")
    c2 = panel(xc, y_c, B, B)
    im = c2.imshow(coef_p, cmap=CMAP, norm=NORM, interpolation="nearest")
    c2.add_patch(Rectangle((-0.5, KV - 0.5), B, B - KV, facecolor="none", edgecolor="#8a8a8a", hatch="////",
                           lw=0.0))
    c2.axhline(KV - 0.5, color=BLACK, lw=1.0, ls="--")
    c2.text(B / 2 - 0.5, (KV + B) / 2 - 0.5, "448 zeros,\ndiscarded", fontsize=7, ha="center", va="center",
            linespacing=1.0, bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"))

    titles = (("(a) Truncation", y_a + KV * p), ("(b) Padding by repetition", y_b + B * p),
              ("(c) Proposed: computed and interlaced padding", y_c + B * p))
    for txt, ytop in titles:
        fig.text((xb - 0.15) / FW, (ytop + 0.08) / FH, txt, fontsize=9, weight="bold")

    a1.set_xlabel("$18 \\times 32$ samples", fontsize=8, labelpad=2)
    a2.set_xlabel("576 coefficients", fontsize=8, labelpad=2)
    b1.set_xlabel("$32 \\times 32$ samples", fontsize=8, labelpad=2)
    b2.set_xlabel("1024 coefficients", fontsize=8, labelpad=2)
    c1.set_xlabel("$32 \\times 32$ samples", fontsize=8, labelpad=2)
    c2.set_xlabel("576 coefficients", fontsize=8, labelpad=2)
    b1.text(B + 0.8, (KV + B) / 2 - 0.5, "14\nrepeated\nrows", fontsize=7, color=RED, va="center", ha="left")
    c1.text(B + 4.2, B / 2, "14\ninterlaced\nrows", fontsize=7, color=RED, va="center", ha="left")

    for y, nrow, n in ((y_a, KV, KV), (y_b, B, B), (y_c, B, B)):
        ym = (y + nrow * p / 2) / FH
        xa, xz = (xb + B * p + 0.8) / FW, (xc - 0.08) / FW
        fig.add_artist(FancyArrowPatch((xa, ym), (xz, ym), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=9, lw=0.9, color=BLACK))
        fig.text((xa + xz) / 2, ym + 0.012, f"{n}-point\nDCT", ha="center", va="bottom", fontsize=7.5,
                 linespacing=1.0)

    for ax in (a1, b1, c1):
        fig.add_artist(ConnectionPatch(xyA=(x0 + B - 0.5, Y0 + KV / 2), coordsA=axv.transData, xyB=(-0.5, KV / 2),
                                       coordsB=ax.transData, color=RED, lw=0.6, alpha=0.7))

    cax = ax_in(xc + B * p + 0.12, y_c, 0.09, y_a + KV * p - y_c)
    cb = fig.colorbar(im, cax=cax)
    ticks = [-1e4, -1e3, -1e2, -1e1, 0, 1e1, 1e2, 1e3, 1e4]
    cb.set_ticks(ticks)
    cb.set_ticklabels(["$-10^{%d}$" % np.log10(-t) if t < 0 else ("$10^{%d}$" % np.log10(t) if t > 0 else "$0$")
                       for t in ticks])
    cb.ax.tick_params(labelsize=7, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)
    a2.set_title("DCT coefficients\nof the luma (Y)", fontsize=8, pad=4)

    os.makedirs(args.output, exist_ok=True)
    fig.savefig(os.path.join(args.output, "fig_block_policies.pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
