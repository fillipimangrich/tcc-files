import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
BLUE, RED, GRAY, BLACK = "#1f6fb4", "#c0392b", "#7f7f7f", "#222222"
NAMES = {"Bikes": "Bikes", "Danger_de_Mort": "Danger de Mort", "Fountain_Vincent2": "Fountain & Vincent 2",
         "Stone_Pillars_Outside": "Stone Pillars Outside", "sideboard": "Sideboard", "greek": "Greek",
         "tarot": "Tarot Cards"}


def alpha(k, n):
    return np.sqrt(1 / n) if k == 0 else np.sqrt(2 / n)


def basis(k, n, t):
    return alpha(k, n) * np.cos(np.pi * (2 * np.asarray(t) + 1) * k / (2 * n))


def dct_matrix(n):
    return np.array([basis(k, n, np.arange(n)) for k in range(n)])


def read_pgx(path):
    raw = open(path, "rb").read()
    header, data = raw.split(b"\n", 1)
    parts = header.decode().replace("+", " + ").split()
    w, h = int(parts[-2]), int(parts[-1])
    return np.frombuffer(data[:2 * w * h], dtype=">u2").reshape(h, w).astype(float)


def thumb_rgb(pgx, a, b, f=8):
    ycc = np.stack([read_pgx(f"{pgx}/{c}/{a:03d}_{b:03d}.pgx") for c in range(3)], axis=-1)
    h, w = (ycc.shape[0] // f) * f, (ycc.shape[1] // f) * f
    ycc = ycc[:h, :w].reshape(h // f, f, w // f, f, 3).mean(axis=(1, 3))
    y, cb, cr = ycc[..., 0], ycc[..., 1] - 512, ycc[..., 2] - 512
    rgb = np.stack([y + 1.5748 * cr, y - 0.1873 * cb - 0.4681 * cr, y + 1.8556 * cb], axis=-1) / 1023
    return np.clip(rgb, 0, 1)


def dct_basis(out):
    n = 8
    t = np.linspace(-0.4, n - 0.6, 300)
    fig, axs = plt.subplots(2, 4, figsize=(6.3, 2.9), sharex=True, sharey=True)
    for k, ax in enumerate(axs.flat):
        ax.plot(t, basis(k, n, t), color=GRAY, lw=0.8)
        ax.stem(range(n), basis(k, n, np.arange(n)), linefmt=BLUE, markerfmt="o", basefmt=" ")
        ax.axhline(0, color=GRAY, lw=0.4)
        ax.set_title(f"$k = {k}$" + (" (DC)" if k == 0 else ""), fontsize=9)
        ax.set_xticks(range(0, n, 2))
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    for ax in axs[1]:
        ax.set_xlabel("sample $n$", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_dct_basis.pdf"))
    plt.close(fig)


def padding_curve(out):
    n, u = 8, np.array([2., 4, 6, 8, 10])
    k_valid = len(u)
    c = dct_matrix(n)
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 4.4), gridspec_kw={"height_ratios": [1.5, 1]})
    for col, (title, pad, color) in enumerate([("(a) Concatenated: positions 5, 6, 7", [5, 6, 7], RED),
                                               ("(b) Interlaced: positions 1, 3, 5", [1, 3, 5], BLUE)]):
        valid = [i for i in range(n) if i not in pad]
        v = -np.linalg.solve(c[k_valid:][:, pad], c[k_valid:][:, valid] @ u)
        x = np.zeros(n)
        x[valid], x[pad] = u, v
        coef = c @ x
        ax = axs[0, col]
        tt = np.linspace(0, n - 1, 300)
        curve = sum(coef[k] * basis(k, n, tt) for k in range(k_valid))
        ax.plot(tt, curve, color=GRAY, lw=1.0, zorder=1, label="sum of the 5 lowest frequencies")
        ax.scatter(valid, u, s=28, color=BLACK, zorder=3, label="original sample")
        ax.scatter(pad, v, s=34, color=color, marker="s", zorder=3, label="padding")
        for i, val in zip(pad, v):
            ax.annotate(f"{val:.2f}", (i, val), textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=7.5, color=color)
        ax.axhline(0, color=GRAY, lw=0.4)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("sample position", fontsize=8)
        ax.set_xticks(range(n))
        ax.set_ylim(-14, 14)
        ax = axs[1, col]
        ax.bar(range(n), coef, color=[color if k > 0 else GRAY for k in range(n)], width=0.6)
        for k in range(k_valid, n):
            ax.text(k, 0.6, "0", ha="center", fontsize=7.5, color=BLACK)
        ax.axhline(0, color=GRAY, lw=0.4)
        ax.axvspan(k_valid - 0.5, n - 0.5, color=GRAY, alpha=0.12, lw=0)
        ax.set_xticks(range(n))
        ax.set_xlabel("DCT coefficient $k$", fontsize=8)
        ax.set_ylim(-17, 17)
    axs[0, 0].set_ylabel("value")
    axs[1, 0].set_ylabel("coefficient")
    axs[0, 1].legend(fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_padding_curve.pdf"))
    plt.close(fig)


def rate_gap(out, metrics):
    psnr = defaultdict(dict)
    for r in csv.DictReader(open(metrics)):
        psnr[(r["lightfield"], r["configuration"])][float(r["target_rate"])] = float(r["psnr_y"])
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    styles = ["o-", "s-", "^-", "D-", "o--", "s--", "^:"]
    for (lf, name), st in zip(NAMES.items(), styles):
        rates = sorted(psnr[(lf, "anchor")])
        diff = [psnr[(lf, "proposed")][r] - psnr[(lf, "anchor")][r] for r in rates]
        ax.plot(range(1, 6), diff, st, ms=4, lw=1, label=name)
    ax.axhline(0, color=GRAY, lw=0.6)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(["1 (lowest)", "2", "3", "4", "5 (highest)"])
    ax.set_xlabel("target rate of the CTC")
    ax.set_ylabel(r"$\Delta$PSNR$_\mathrm{Y}$ (dB)")
    ax.legend(fontsize=7, ncol=2, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_rate_gap.pdf"))
    plt.close(fig)


def boundary_blocks(out):
    W, H, B = 625, 434, 32
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    orange = "#f0b27a"
    for x0 in range(0, W, B):
        for y0 in range(0, H, B):
            w, h = min(B, W - x0), min(B, H - y0)
            border = w < B or h < B
            ax.add_patch(Rectangle((x0, y0), w, h, facecolor=orange if border else "white",
                                   edgecolor=GRAY, lw=0.4))
    ax.set_xlim(-5, W + 70)
    ax.set_ylim(H + 40, -40)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.annotate("", xy=(0, -18), xytext=(W, -18), arrowprops=dict(arrowstyle="<->", color=BLACK, lw=0.7))
    ax.text(W / 2, -24, r"$625 = 19 \times 32 + 17$ pixels", ha="center", va="bottom", fontsize=8.5)
    ax.annotate("", xy=(W + 14, 0), xytext=(W + 14, H), arrowprops=dict(arrowstyle="<->", color=BLACK, lw=0.7))
    ax.text(W + 20, H / 2, "$434 = 13 \\times 32 + 18$\npixels", ha="left", va="center", fontsize=8.5)
    ax.text(W - 8, H + 26, "boundary blocks: 17 columns and/or 18 rows", ha="right", va="center", fontsize=8,
            color="#a04000")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_boundary_blocks.pdf"))
    plt.close(fig)


def border_policies(out):
    n, u = 8, np.array([2., 4, 6, 8, 10])
    k_valid = len(u)
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 4.2), gridspec_kw={"height_ratios": [1.3, 1]})
    for col, title in enumerate(["(a) Truncation: 5-point DCT", "(b) Padding by repetition: 8-point DCT"]):
        ax = axs[0, col]
        ax.axvspan(k_valid - 0.5, n - 0.5, color=GRAY, alpha=0.12, lw=0)
        ax.scatter(range(k_valid), u, s=28, color=BLACK, zorder=3, label="original sample")
        if col == 0:
            coef = dct_matrix(k_valid) @ u
            ax.text(6, 2, "outside the\nlight field", ha="center", fontsize=7.5, color=GRAY)
        else:
            x = np.r_[u, [u[-1]] * (n - k_valid)]
            ax.scatter(range(k_valid, n), x[k_valid:], s=34, color=RED, marker="s", zorder=3, label="repeated sample")
            ax.plot(range(n), x, color=GRAY, lw=0.8, zorder=1)
            coef = dct_matrix(n) @ x
            ax.legend(fontsize=7, loc="lower right", frameon=False)
        if col == 0:
            ax.plot(range(k_valid), u, color=GRAY, lw=0.8, zorder=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 12)
        ax.set_xticks(range(n))
        ax.set_xlabel("sample position", fontsize=8)
        ax = axs[1, col]
        ax.bar(range(len(coef)), coef, color=[GRAY] + [RED if col else BLUE] * (len(coef) - 1), width=0.6)
        ax.axhline(0, color=GRAY, lw=0.4)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-10, 23)
        ax.set_xticks(range(n))
        ax.set_xlabel("DCT coefficient $k$", fontsize=8)
    axs[0, 0].set_ylabel("value")
    axs[1, 0].set_ylabel("coefficient")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_border_policies.pdf"))
    plt.close(fig)


def padded_views(out, dataset):
    pgx = os.path.join(dataset, "Bikes")
    nv, kv, gap = 16, 11, 3
    thumbs = {(a, b): thumb_rgb(pgx, a, b) for a in range(kv) for b in range(kv)}
    th, tw = thumbs[(0, 0)].shape[:2]
    mosaic = np.ones((nv * (th + gap), nv * (tw + gap), 3))
    for r in range(nv):
        for c in range(nv):
            img = thumbs[(min(c, kv - 1), min(r, kv - 1))]
            y0, x0 = r * (th + gap), c * (tw + gap)
            mosaic[y0:y0 + th, x0:x0 + tw] = img
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    ax.imshow(mosaic, interpolation="lanczos")
    for r in range(nv):
        for c in range(nv):
            if r >= kv or c >= kv:
                ax.add_patch(Rectangle((c * (tw + gap) - 1.5, r * (th + gap) - 1.5), tw + 2, th + 2, fill=False,
                                       edgecolor=RED, lw=0.9))
    ax.add_patch(Rectangle((-gap, -gap), kv * (tw + gap) + gap / 2, kv * (th + gap) + gap / 2, fill=False,
                           edgecolor=BLACK, lw=1.4, linestyle="--"))
    ax.text(kv * (tw + gap) / 2, -12, "121 views of the light field ($11 \\times 11$)", ha="center", va="bottom",
            fontsize=8.5)
    ax.text(nv * (tw + gap) + 8, kv * (th + gap) + 2.5 * (th + gap),
            "135 new views:\ncopies of the\nlast row and\ncolumn of views", ha="left", va="center", fontsize=8.5,
            color=RED)
    ax.set_xlim(-gap - 2, nv * (tw + gap) + 150)
    ax.set_ylim(nv * (th + gap) + 2, -40)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_padded_views.jpg"), dpi=250, pil_kwargs={"quality": 92})
    plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="folder with the light fields in PGX (one subfolder per light field)")
    ap.add_argument("output", help="output folder")
    ap.add_argument("--metrics", default=os.path.join(here, "..", "data", "metrics.csv"))
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    dct_basis(args.output)
    padding_curve(args.output)
    rate_gap(args.output, args.metrics)
    boundary_blocks(args.output)
    border_policies(args.output)
    padded_views(args.output, args.dataset)


if __name__ == "__main__":
    main()
