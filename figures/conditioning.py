import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct

plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
BLUE, RED, GRAY = "#1f6fb4", "#c0392b", "#7f7f7f"


def dct_matrix(n):
    return dct(np.eye(n), norm="ortho", axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("padding_cpp", help="Padding.cpp of the patched reference software (source of the orders)")
    ap.add_argument("output", help="output folder")
    args = ap.parse_args()
    src = re.sub(r"//.*", "", open(args.padding_cpp).read())
    order = {(int(m[1]), int(m[2])): [int(v) for v in m[3].split(",")]
             for m in re.finditer(r"n(\d+)_v(\d+)_order\[\d+\]\s*=\s*\{([^}]*)\}", src)}

    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    for n, mk in [(8, "o"), (16, "s"), (32, "D")]:
        ks = [k for k in range(n // 2 + 1, n) if (n, k) in order]
        m = [n - k for k in ks]
        c = dct_matrix(n)
        unordered = [np.linalg.cond(c[k:][:, list(range(k, n))]) for k in ks]
        ordered = [np.linalg.cond(c[k:][:, order[(n, k)]]) for k in ks]
        ax.semilogy(m, unordered, marker=mk, ms=4, color=RED, lw=1, label=f"concatenated, $N={n}$")
        ax.semilogy(m, ordered, marker=mk, ms=4, color=BLUE, lw=1, ls="--", label=f"interlaced, $N={n}$")
    ax.axhline(1 / np.finfo(float).eps, color=GRAY, ls=":", lw=0.8)
    ax.text(8.5, 2e16 * 1.6, "double precision limit ($\\approx 10^{16}$)", color=GRAY, fontsize=7.5)
    ax.set_xlabel("number of padding samples ($N-K$)")
    ax.set_ylabel("condition number of $C_{11}$")
    ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout()
    os.makedirs(args.output, exist_ok=True)
    fig.savefig(os.path.join(args.output, "fig_conditioning.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
