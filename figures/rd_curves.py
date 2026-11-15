import argparse
import csv
import json
import os
import sys
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import style
from style import BLACK, BLUE, GRAY, ORANGE

warnings.filterwarnings("ignore")

ANCHOR = "anchor"
CODECS = {
    "anchor": ("Anchor (CTC blocks)", BLACK, "o"),
    "truncation": ("Truncation", BLUE, "s"),
    "repetition": ("Padding by repetition", GRAY, "v"),
    "proposed": ("Proposed", ORANGE, "D"),
}
DISPLAY = {"Bikes": "Bikes", "Danger_de_Mort": "Danger de Mort", "Fountain_Vincent2": "Fountain & Vincent 2",
           "Stone_Pillars_Outside": "Stone Pillars Outside", "greek": "Greek", "sideboard": "Sideboard",
           "tarot": "Tarot Cards"}
LIGHTFIELDS = ["Bikes", "Danger_de_Mort", "Fountain_Vincent2", "Stone_Pillars_Outside", "greek", "sideboard", "tarot"]
LABELS = [("metrics", "psnr_y", r"\mathrm{PSNR}_\mathrm{Y}"),
          ("weighted-metrics", "psnr_yuv_weighted", r"\mathrm{PSNR}_\mathrm{YCbCr}"),
          ("bd-adjusted-metrics", "float_ssim_db", r"\mathrm{SSIM}_\mathrm{dB}"),
          ("bd-adjusted-metrics", "vmaf_db", r"\mathrm{VMAF}_\mathrm{dB}")]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("ctc_tools", help="folder of jpeg-pleno-ctc-tools (lfc_toolkit)")
    ap.add_argument("output", help="output folder")
    ap.add_argument("--metrics", default=os.path.join(here, "..", "data", "metrics.csv"))
    args = ap.parse_args()

    sys.path.insert(0, args.ctc_tools)
    from lfc_toolkit.src.quality.rd_curve import RDCurve
    from lfc_toolkit.src.quality.rd_plot import RDCurveMatplotlibView, RDPlotMatplotlib

    quality = json.load(open(f"{args.ctc_tools}/lfc_toolkit/data/quality.json"))["quality"]
    for group, metric, label in LABELS:
        quality[group][metric]["label"] = label
    lf_cfg = f"{args.ctc_tools}/lfc_toolkit/data/ctc_configurations/lightfields"
    out = args.output
    rd_plots = [
        {"path": out, "anchor": ANCHOR, "interpolation": 100, "bpp_logscale": True,
         "figure_size": [6, 4], "metrics": ["psnr_y", "psnr_yuv_weighted", "float_ssim_db", "vmaf_db"]},
        {"path": out, "anchor": ANCHOR, "bpp_logscale": True,
         "figure_size": [6, 4], "metrics": ["float_ssim", "vmaf"]},
    ]

    class CurveView(RDCurveMatplotlibView):
        def show(self, *a, **kw):
            super().show(*a, **kw)
            import matplotlib.pyplot as plt
            line = plt.gca().get_lines()[-1]
            line.set_label(line.get_label().replace("BDBR", "BD-Rate"))

    class Config(dict):
        def __init__(self):
            super().__init__(quality=quality, rd_plots=rd_plots)
            self.lightfield_names = LIGHTFIELDS
            self.cfg = {lf: json.load(open(f"{lf_cfg}/{lf}.json")) for lf in LIGHTFIELDS}

        def get_lightfield_configuration(self, name):
            return self.cfg[name]

    class LF:
        def __init__(self, name):
            self.name = name

    os.makedirs(out, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "pdf.fonttype": 42})
    cfg = Config()
    pts = defaultdict(list)
    for r in csv.DictReader(open(args.metrics)):
        if r["configuration"] in CODECS:
            pts[(r["lightfield"], r["configuration"])].append(r)
    units = {**quality["metrics"], **quality["bd-adjusted-metrics"], **quality["weighted-metrics"]}
    for rdp in rd_plots:
        for lf in LIGHTFIELDS:
            for metric in rdp["metrics"]:
                curves = []
                for v, (title, color, marker) in CODECS.items():
                    rows = sorted(pts[(lf, v)], key=lambda r: float(r["bpp"]))
                    curves.append(CurveView(cfg, RDCurve(
                        rates=[float(r["bpp"]) for r in rows], rate_unit="bpp",
                        distortions=[float(r[metric]) for r in rows], distortion_name=metric,
                        distortion_unit=units[metric].get("unit", ""), codec_name=v, title=title), color, marker))
                targets = sorted({float(r["target_rate"]) for r in pts[(lf, ANCHOR)]})
                for c in curves:
                    c.db_title = None
                RDPlotMatplotlib(cfg, rdp, LF(lf)).plot(
                    rd_curves=curves, anchor=curves[0], title=DISPLAY[lf], bpp_logscale=True, target_bpps=targets,
                    interpolation=rdp.get("interpolation"), filename=f"{out}/rd_{lf}_{metric}.pdf")


if __name__ == "__main__":
    main()
