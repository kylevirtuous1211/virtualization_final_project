"""Figures for the progress report, read from the capacity experiments' metrics.json and summary.json.

Run in the StreamSplat image, which has matplotlib:
  docker run --rm --network none -u $(id -u):$(id -g) -v "$PWD:$PWD" -w "$PWD" streamsplat:5d686e9 \
    python docs/reports/progress_2026-11/make_figures.py
OUTPUTS overrides the source root (default: <repo>/results/capacity, which results/collect.py fills from outputs/).
"""
import json, os
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("OUTPUTS", HERE.parents[2] / "results" / "capacity"))
FIGS = HERE / "figures"

# dataviz reference palette, light mode; slots 1-3 validate under all pairs.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, CONTROL = "#0b0b0b", "#52514e", "#e4e3df", "#b5b3ad"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42,
})
WIDTH = 3.33  # one USENIX column, inches


def load(experiment, name="metrics.json"):
    return json.loads((OUT / experiment / name).read_text())


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def latency():
    g15, g16 = load("G15")["points"], load("G16")["points"]
    series = [  # (label, colour, marker, [(N, p95 of each repeat)])
        ("Serial", ORANGE, "s", [(int(n), [r["lat_p95_max"] for r in runs.values()]) for n, runs in g15["serial"].items()]),  # G15
        ("Opt", BLUE, "o",
         [(int(n), [r["lat_p95_max"] for r in runs.values()]) for source in (g15["fast"], g16["fast"]) for n, runs in source.items()]),  # G15, G16
        ("Opt+DepthBF16", AQUA, "^", [(int(n), [r["lat_p95_max"] for r in runs.values()]) for n, runs in g16["depthbf16"].items()]),  # G16
    ]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.2))
    ax.axhline(1.0, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.text(0.6, 1.12, "SLO: p95 $\\leq$ 1 s", color=INK2, fontsize=7)
    for label, colour, marker, points in series:
        by_n = {}
        for n, values in points:
            by_n.setdefault(n, []).extend(values)
        ns = sorted(by_n)
        ax.plot(ns, [mean(by_n[n]) for n in ns], color=colour, linewidth=1.2, label=label)
        ax.scatter([n for n in ns for _ in by_n[n]], [v for n in ns for v in by_n[n]], color=colour, marker=marker,
                   s=12, linewidths=0.5, edgecolors="white", zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 20)
    ax.set_yticks([0.1, 1, 10], ["0.1", "1", "10"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0, 18)
    ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    ax.set_xlabel("Users, one keyframe per user per second")
    ax.set_ylabel("Worst user's p95 latency (s)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, which="major")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7, handlelength=1.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "latency.pdf")
    return series


def breakdown():
    def user(path):
        return json.loads((OUT / path / "summary.json").read_text())["users"][0]
    serial = user("G15/step0/serial")  # G15, one run
    configs = [("Serial", serial["svc_p50"], serial["svc_depth_p50"])]
    for arm, label in (("fast", "Opt"), ("depthbf16", "Opt+DepthBF16")):  # G16, two runs each
        runs = [user(f"G16/step0/{arm}_r{rep}") for rep in (1, 2)]
        configs.append((label, mean(r["svc_p50"] for r in runs), mean(r["svc_depth_p50"] for r in runs)))
    fig, ax = plt.subplots(figsize=(WIDTH, 1.35))
    for row, (label, total, depth) in enumerate(reversed(configs)):
        ax.barh(row, depth * 1e3, height=0.55, color=INK2, edgecolor="white", linewidth=1,
                label="Depth (upload included)" if row == 0 else None)
        ax.barh(row, (total - depth) * 1e3, left=depth * 1e3, height=0.55, color=CONTROL, edgecolor="white", linewidth=1,
                label="Rest of the update" if row == 0 else None)
        ax.text(depth * 1e3 / 2, row, f"{depth * 1e3:.1f}", va="center", ha="center", color="white", fontsize=7)
        ax.text(total * 1e3 + 4, row, f"{total * 1e3:.1f} ms", va="center", color=INK, fontsize=7)
    ax.set_yticks(range(len(configs)), [c[0] for c in reversed(configs)])
    ax.set_xlabel("Median service time per update (ms)")
    ax.set_xlim(0, 240)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", fontsize=7, handlelength=1.0, bbox_to_anchor=(1.0, 0.98), ncol=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "breakdown.pdf")
    return configs


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    for label, _, _, points in latency():
        print("latency", label, sorted({n for n, _ in points}))
    for label, total, depth in breakdown():
        print("breakdown", label, round(total * 1e3, 1), round(depth * 1e3, 1))
