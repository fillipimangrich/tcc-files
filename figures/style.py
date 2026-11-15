"""Common style of the figures of the thesis: one palette, one set of fonts and line widths.

Meaning of the colors:
    BLUE    valid (original) samples and what uses only them
    ORANGE  padding values and the proposed technique
    COEF    transform coefficients
    GRAY    auxiliary curves and other alternatives
    BLACK   text and the reference (anchor)
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, COEF, GRAY, BLACK = "#1f6fb4", "#e67e22", "#6b7280", "#8a8a8a", "#222222"
GRID = "#e6e6e6"
WIDTH = 6.3  # width of the text block, in inches


def setup(font_size=9):
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": font_size,
        "axes.titlesize": font_size, "axes.labelsize": font_size - 0.5,
        "xtick.labelsize": font_size - 1, "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.1, "lines.markersize": 4.5, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "pdf.fonttype": 42,
    })


def tint(color, amount):
    """Mix a color with white: amount = 1 keeps the color, 0 gives white."""
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(1 - amount * (1 - rgb))


def shade(color, amount):
    """Mix a color with black: amount = 1 keeps the color, 0 gives black."""
    return tuple(amount * np.array(matplotlib.colors.to_rgb(color)))


def panel_title(ax, tag, text):
    """Title of a panel in the form '(a) text', aligned to the left."""
    ax.set_title(f"({tag}) {text}", loc="left", pad=6)


def samples(ax, positions, values, **kw):
    return ax.scatter(positions, values, s=22, color=BLUE, zorder=3, edgecolor="white", linewidth=0.6, **kw)


def padding(ax, positions, values, **kw):
    return ax.scatter(positions, values, s=24, marker="s", color=ORANGE, zorder=3, edgecolor="white",
                      linewidth=0.6, **kw)


# diverging map for signed values: blue (negative), white (zero), orange (positive)
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_orange", [shade(BLUE, 0.45), BLUE, tint(BLUE, 0.35), "white", tint(ORANGE, 0.4), ORANGE, shade(ORANGE, 0.5)])
