import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import style

VIEWS = {"Bikes": "007_007", "Danger_de_Mort": "007_007", "Fountain_Vincent2": "007_007",
         "Stone_Pillars_Outside": "007_007", "sideboard": "004_004", "greek": "004_004", "tarot": "008_008"}


def read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    parts = data.split(maxsplit=4)
    w, h, mx = int(parts[1]), int(parts[2]), int(parts[3])
    dt = ">u2" if mx > 255 else "u1"
    return np.frombuffer(parts[4], dtype=dt, count=w * h * 3).reshape(h, w, 3).astype(float) / mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="folder with the light fields in PPM (one subfolder per light field)")
    ap.add_argument("output", help="output folder")
    args = ap.parse_args()
    style.setup()
    fig, axs = plt.subplots(3, 3, figsize=(6.3, 6.6))
    axs = axs.ravel()
    for i, (lf, v) in enumerate(VIEWS.items()):
        img = read_ppm(f"{args.dataset}/{lf}/{v}.ppm")
        axs[i].imshow(np.clip(img, 0, 1))
        axs[i].set_axis_off()
        axs[i].set_title(f"({chr(97 + i)}) {lf.replace('_', ' ')}", fontsize=8.5)
    for j in range(len(VIEWS), 9):
        axs[j].set_axis_off()
    fig.tight_layout()
    os.makedirs(args.output, exist_ok=True)
    fig.savefig(os.path.join(args.output, "fig_dataset_views.pdf"), dpi=200)


if __name__ == "__main__":
    main()
