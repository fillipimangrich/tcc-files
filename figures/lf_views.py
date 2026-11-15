import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle

import style
from style import BLACK, ORANGE

style.setup(10)


def read_pgx(path):
    raw = open(path, "rb").read()
    header, data = raw.split(b"\n", 1)
    parts = header.decode().replace("+", " + ").split()
    w, h = int(parts[-2]), int(parts[-1])
    return np.frombuffer(data[:2 * w * h], dtype=">u2").reshape(h, w).astype(float)


def view_rgb(pgx, a, b, f):
    ycc = np.stack([read_pgx(f"{pgx}/{c}/{a:03d}_{b:03d}.pgx") for c in range(3)], axis=-1)
    h, w = (ycc.shape[0] // f) * f, (ycc.shape[1] // f) * f
    ycc = ycc[:h, :w].reshape(h // f, f, w // f, f, 3).mean(axis=(1, 3))
    y, cb, cr = ycc[..., 0], ycc[..., 1] - 512, ycc[..., 2] - 512
    rgb = np.stack([y + 1.5748 * cr, y - 0.1873 * cb - 0.4681 * cr, y + 1.8556 * cb], axis=-1) / 1023
    return np.clip(rgb, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="folder with the light fields in PGX (one subfolder per light field)")
    ap.add_argument("output", help="output folder")
    args = ap.parse_args()
    pgx = os.path.join(args.dataset, "Bikes")

    kv, gap = 11, 3
    thumbs = {(s, t): view_rgb(pgx, s, t, 8) for s in range(kv) for t in range(kv)}
    th, tw = thumbs[(0, 0)].shape[:2]
    mosaic = np.ones((kv * (th + gap) - gap, kv * (tw + gap) - gap, 3))
    for t in range(kv):
        for s in range(kv):
            y0, x0 = t * (th + gap), s * (tw + gap)
            mosaic[y0:y0 + th, x0:x0 + tw] = thumbs[(s, t)]

    ts, ss = 5, 5
    big = view_rgb(pgx, ss, ts, 2)

    fig = plt.figure(figsize=(6.3, 3.35))
    axm = fig.add_axes([0.06, 0.04, 0.47, 0.82])
    axv = fig.add_axes([0.62, 0.20, 0.36, 0.52])

    axm.imshow(mosaic, interpolation="lanczos")
    x0, y0 = ss * (tw + gap), ts * (th + gap)
    axm.add_patch(Rectangle((x0 - 1.5, y0 - 1.5), tw + 2, th + 2, fill=False, edgecolor=ORANGE, lw=1.4))
    axm.set_xlim(-2, mosaic.shape[1] + 1)
    axm.set_ylim(mosaic.shape[0] + 1, -2)
    axm.axis("off")
    W, H = mosaic.shape[1], mosaic.shape[0]
    axm.annotate("", xy=(W * 0.62, -16), xytext=(W * 0.38, -16), xycoords="data",
                 arrowprops=dict(arrowstyle="->", lw=1.0, color=BLACK), annotation_clip=False)
    axm.text(W * 0.5, -24, "$s$", ha="center", va="bottom", fontsize=11, clip_on=False)
    axm.annotate("", xy=(-16, H * 0.62), xytext=(-16, H * 0.38), xycoords="data",
                 arrowprops=dict(arrowstyle="->", lw=1.0, color=BLACK), annotation_clip=False)
    axm.text(-24, H * 0.5, "$t$", ha="right", va="center", fontsize=11, clip_on=False)
    axm.text(W * 0.5, H + 12, "$11 \\times 11$ views", ha="center", va="top", fontsize=9.5, clip_on=False)

    axv.imshow(big, interpolation="lanczos")
    axv.set_xticks([])
    axv.set_yticks([])
    for sp in axv.spines.values():
        sp.set_edgecolor(ORANGE)
        sp.set_linewidth(1.4)
    bh, bw = big.shape[:2]
    axv.annotate("", xy=(bw * 0.62, -10), xytext=(bw * 0.38, -10), xycoords="data",
                 arrowprops=dict(arrowstyle="->", lw=1.0, color=BLACK), annotation_clip=False)
    axv.text(bw * 0.5, -15, "$u$", ha="center", va="bottom", fontsize=11, clip_on=False)
    axv.annotate("", xy=(-10, bh * 0.62), xytext=(-10, bh * 0.38), xycoords="data",
                 arrowprops=dict(arrowstyle="->", lw=1.0, color=BLACK), annotation_clip=False)
    axv.text(-15, bh * 0.5, "$v$", ha="right", va="center", fontsize=11, clip_on=False)
    axv.text(bw * 0.5, bh + 12, "one view ($t = s = 5$): $625 \\times 434$ pixels", ha="center", va="top",
             fontsize=9.5, clip_on=False)

    for (xa, ya), (xb, yb) in [((x0 + tw, y0), (0, 0)), ((x0 + tw, y0 + th), (0, bh))]:
        fig.add_artist(ConnectionPatch(xyA=(xa, ya), coordsA=axm.transData, xyB=(xb, yb), coordsB=axv.transData,
                                       color=ORANGE, lw=0.8, alpha=0.8))

    os.makedirs(args.output, exist_ok=True)
    fig.savefig(os.path.join(args.output, "fig_lf_views.jpg"), dpi=250, bbox_inches="tight", pad_inches=0.04,
                pil_kwargs={"quality": 92})
    plt.close(fig)


if __name__ == "__main__":
    main()
