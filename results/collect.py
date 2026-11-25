"""Copy the numbers the progress report quotes from the untracked run outputs into the tracked results/,
and build results/tables.md, the report's four tables computed from those files.

Usage: python3 results/collect.py
It reads outputs/ and runs/ under the repository root. A source that does not exist leaves the tracked
copy as it is (for example the AnySplat runs G1-G3, which reproduce.sh does not rerun), so after a partial
rerun `git diff results/` shows exactly what changed against the published numbers.
"""
import json, statistics, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SOURCES = [  # globs relative to the repository root; outputs/<x> is copied to results/<x>, runs/<x> to results/runs/<x>
    "runs/G1/k*/summary.json",
    "runs/G2/ts/n*/r*/summary.json", "runs/G2/mps/n*/r*/summary.json",
    "outputs/capacity/G3/metrics.json",
    "outputs/streamsplat-repro/G4/metrics.json",
    "outputs/capacity/G5/metrics.json", "outputs/capacity/G6/metrics.json", "outputs/capacity/G7/metrics.json",
    "outputs/streamsplat-repro/G8/metrics.json", "outputs/streamsplat-repro/G8/windows.jsonl",
    "outputs/capacity/G9/metrics.json", "outputs/capacity/G10/run/metrics.json",
    "outputs/capacity/G11/metrics.json", "outputs/capacity/G11/probe.log",
    "outputs/capacity/G12/metrics.json", "outputs/capacity/G12/davis/metrics.json", "outputs/capacity/G12/davis/windows.jsonl",
    "outputs/capacity/G13/metrics.json", "outputs/capacity/G14/metrics.json",
    "outputs/capacity/G15/metrics.json", "outputs/capacity/G15/step0/serial/summary.json",
    "outputs/capacity/G16/metrics.json", "outputs/capacity/G16/step0/*_r*/summary.json",
    "outputs/capacity/G16/davis/metrics.json", "outputs/capacity/G16/davis/windows.jsonl",
    "outputs/capacity/G16/davis/repro.json", "outputs/capacity/G16/davis/depth_against_G8.json",
]
DROP_KEYS = {"outputs/capacity/G16/davis/depth_against_G8.json": "per_map"}  # per-image detail, 0.3 MB


def destination(source):
    relative = Path(source)
    return RESULTS / (relative.relative_to("outputs") if relative.parts[0] == "outputs" else relative)


def copy_sources():
    for pattern in SOURCES:
        matches = sorted(REPO.glob(pattern))
        if not matches:
            print(f"kept the tracked copy: nothing matches {pattern}")
            continue
        for source in matches:
            relative = str(source.relative_to(REPO))
            target = destination(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".log":  # logs quote absolute paths of the machine that ran them
                target.write_text(source.read_text().replace(str(REPO), "<repo>"))
            elif relative in DROP_KEYS:
                data = json.loads(source.read_text())
                data.pop(DROP_KEYS[relative], None)
                target.write_text(json.dumps(data, indent=1) + "\n")
            else:
                target.write_bytes(source.read_bytes())


def load(path):
    return json.loads((RESULTS / path).read_text())


def span(values, digits, joiner="–"):
    low, high = f"{min(values):.{digits}f}", f"{max(values):.{digits}f}"
    return low if low == high else f"{low}{joiner}{high}"


def mean(values):
    return statistics.mean(list(values))


def g2_points():
    """G2 has no metrics.json: its runs are AnySplat summaries; throughput is updates over the 60 s window."""
    points = {}
    for summary_path in sorted((RESULTS / "runs/G2").glob("*/n*/r*/summary.json")):
        summary = json.loads(summary_path.read_text())
        arm, users = summary_path.parts[-4], int(summary_path.parts[-3][1:])
        passed = summary["all_meet_slo"] and not summary["foreign_procs"]
        points.setdefault(arm, {}).setdefault(users, []).append(
            {"pass": passed, "updates_per_s": sum(user["updates"] for user in summary["users"]) / 60})
    return points


def sweep_points(metrics):
    return {arm: {int(users): list(reps.values()) for users, reps in by_users.items()}
            for arm, by_users in metrics["points"].items()}


def capacity(by_users):
    both = [users for users, reps in by_users.items() if len(reps) == 2 and all(rep["pass"] for rep in reps)]
    return max(both, default=0)


def saturated(by_users):
    return [rep["updates_per_s"] for reps in by_users.values() for rep in reps if not rep["pass"]]


def table_sharing():
    rows = [("AnySplat, k=8", "1/s", g2_points()),
            ("AnySplat, k=2", "2/s", sweep_points(load("capacity/G3/metrics.json"))),
            ("StreamSplat", "1/s", sweep_points(load("capacity/G6/metrics.json"))),
            ("StreamSplat", "3/s", sweep_points(load("capacity/G7/metrics.json")))]
    lines = ["## Table 1: users served under time-slicing (TS) and MPS",
             "", "| Model | Rate | Users TS | Users MPS | Saturated updates/s TS | Saturated updates/s MPS |", "|---|---|---|---|---|---|"]
    for model, rate, points in rows:
        lines.append(f"| {model} | {rate} | {capacity(points['ts'])} | {capacity(points['mps'])} | "
                     f"{span(saturated(points['ts']), 2)} | {span(saturated(points['mps']), 2)} |")
    batched = load("capacity/G9/metrics.json")["step0_updates_per_s"]
    lines += ["", "| Batching server, batch size b | 1 | 2 | 4 | 8 |", "|---|---|---|---|---|",
              "| Updates/s, back to back | " + " | ".join(f"{batched[f'b{b}']:.2f}" for b in (1, 2, 4, 8)) + " |"]
    return lines


def table_ablation():
    steps = [("Feature cache", "G11", "nocache", "cache"), ("bf16 model", "G12", "tf32", "bf16"),
             ("No host waits", "G13", "sync", "nosync"), ("CUDA graphs", "G14", "eager", "graphs")]
    lines = ["## Table 2: each optimization against its matched control",
             "", "| Optimization | Updates/s b=1 | Speedup b=1 | Speedup b=4 | Memory (MiB) | Keyframe PSNR (dB) |",
             "|---|---|---|---|---|---|"]
    for label, entry, control, treated in steps:
        points = load(f"capacity/{entry}/metrics.json")["points"]
        rate = {arm: {b: mean(rep["updates_per_s"] for rep in points[arm][b].values()) for b in ("1", "4")}
                for arm in (control, treated)}
        memory = [points[treated]["1"][rep]["gpu_mem_mib_max"] - points[control]["1"][rep]["gpu_mem_mib_max"]
                  for rep in points[treated]["1"]]
        psnr = (f"{load('capacity/G12/davis/metrics.json')['against_G8']['psnr_given']['change_mean']:+.2f}"
                if entry == "G12" else "not evaluated on DAVIS")
        lines.append(f"| {label} | {rate[control]['1']:.2f}→{rate[treated]['1']:.2f} | "
                     f"{rate[treated]['1'] / rate[control]['1']:.2f} | {rate[treated]['4'] / rate[control]['4']:.2f} | "
                     f"{span_signed(memory)} | {psnr} |")
    g16 = load("capacity/G16/metrics.json")["derived"]
    depth_psnr = load("capacity/G16/davis/metrics.json")["against_G12"]["psnr_given"]["change_mean"]
    lines.append(f"| bf16 depth | {g16['step0']['fast']['mean_updates_per_s']:.2f}→{g16['step0']['depthbf16']['mean_updates_per_s']:.2f} | "
                 f"{g16['throughput_ratio']:.2f} | – | {span_signed(g16['gpu_mem_change_mib'].values())} | {depth_psnr:+.3f} |")
    return lines


def span_signed(values):
    values = sorted(round(value) for value in values)
    low, high = f"{values[0]:+d}", f"{values[-1]:+d}"
    return low if low == high else f"{low} to {high}"


def table_capacity():
    g15, g16 = load("capacity/G15/metrics.json"), load("capacity/G16/metrics.json")
    sweeps = {"Serial": [g15["points"]["serial"]], "Opt": [g15["points"]["fast"], g16["points"]["fast"]],
              "Opt+DepthBF16": [g16["points"]["depthbf16"]]}
    back_to_back = {"Serial": [g15["step0_updates_per_s"]["serial"]],
                    "Opt": [g15["step0_updates_per_s"]["fast"], g16["derived"]["step0"]["fast"]["mean_updates_per_s"]],
                    "Opt+DepthBF16": [g16["derived"]["step0"]["depthbf16"]["mean_updates_per_s"]]}
    lines = ["## Table 3: capacity at one keyframe per user per second",
             "", "| Configuration | Back to back (updates/s) | Capacity (users) | p95 at capacity (s) | Memory total (MiB) | Memory per user (MiB) |",
             "|---|---|---|---|---|---|"]
    for name, sweep_list in sweeps.items():
        capacities = {capacity(sweep_points({"points": {"arm": sweep}})["arm"]) for sweep in sweep_list}
        users = min(capacities)
        reps = [rep for sweep in sweep_list for rep in sweep[str(users)].values()]
        p95 = sorted(rep["lat_p95_max"] for rep in reps)
        total = max(rep["gpu_mem_mib_max"] for rep in reps)
        lines.append(f"| {name} | {span(back_to_back[name], 2)} | {'/'.join(map(str, sorted(capacities)))} | "
                     f"{', '.join(f'{value:.3f}' for value in p95)} | {total:,} | {total / users:.0f} |")
    return lines


def table_quality():
    full = load("streamsplat-repro/G8/metrics.json")["val"]
    rows = [("Full precision", full, None),
            ("bf16 model", load("capacity/G12/davis/metrics.json")["val"], load("capacity/G12/davis/metrics.json")["against_G8"]),
            ("bf16 model and depth", load("capacity/G16/davis/metrics.json")["val"], load("capacity/G16/davis/metrics.json")["against_G8"])]
    lines = ["## Table 4: quality on the DAVIS 2017 validation windows (the authors' metric)",
             "", "| Precision | Keyframes (dB) | Middle (dB) | Keyframes vs full, mean (dB) | Windows losing > 0.5 dB |",
             "|---|---|---|---|---|",
             "| Paper (StreamSplat, reported) | 37.83 | 23.66 | | |"]
    for name, val, paired in rows:
        change = "–" if paired is None else f"{paired['psnr_given']['change_mean']:+.2f}"
        share = "–" if paired is None else f"{paired['psnr_given']['share_dropping_over_0.5_dB']:.1%}"
        lines.append(f"| {name} | {val['psnr_given_mean']:.2f} | {val['psnr_novel_mean']:.2f} | {change} | {share} |")
    return lines


def main():
    copy_sources()
    sections = [table_sharing(), table_ablation(), table_capacity(), table_quality()]
    header = ["# Tables of the progress report, computed from results/", "",
              "Generated by results/collect.py; do not edit by hand.",
              "Definitions follow the table captions in docs/reports/progress_2026-11.", ""]
    (RESULTS / "tables.md").write_text("\n".join(header + [line for section in sections for line in section + [""]]))
    print((RESULTS / "tables.md").read_text())


if __name__ == "__main__":
    sys.exit(main())
