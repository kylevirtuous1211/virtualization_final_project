#!/usr/bin/env bash
# G3 (LOG.md): at k = 2 frames per update, does MPS serve more AnySplat users than time-slicing?
# Exploratory lane, like G1 and G2. Plan `misty-painting-summit`, approved 2026-11-13.
#
# Setup: every user is its own process with its own AnySplat (bench/capacity.py, unchanged since G2);
# k = 2, one update per user every 0.5 s, 60 s per point, 3 warm-up updates, evenly staggered phases;
# arms ts (driver time-slicing) and mps (user-started MPS daemon) interleaved per N; 2 repeats per point.
# ts is the matched control for mps (one factor differs); N = 1 is the slowdown reference.
#
# Registered before any run:
#   SLO: p95 latency <= 1.0 s, no OOM, 120/120 updates. Secondary, reported only: p95 <= 0.5 s.
#   Capacity = largest N where both repeats pass; the largest N where any repeat passes is also reported,
#   and where the two differ the edge is "unresolved at N, n = 2". An OOM is a failed point, not a crash to hide.
#   Stop rule: an arm stops after two consecutive N where neither repeat passes.
#   P1 capacity(mps) >= capacity(ts) + 1, point estimate ts 4, mps 6.
#      Refuted if equal and both fail at capacity + 1 by latency; if mps fails at ts capacity + 1 by OOM,
#      P1 is undecided on compute and the tie is a memory fact.
#   P2 ts throughput ceiling (step 0) = 8.2 +- 0.4 updates/s. Refuted if >= 9.0.
#   P3 mps throughput ceiling (step 0) >= 9.5 updates/s if P1's mechanism exists; <= 8.6 predicts a tie.
#   P4 MPS does not lower per-user memory: at N = 2 and 4 the median per-user proc_vram_mib_max under mps
#      is within -300..+600 MiB of ts. Refuted if mps is >= 300 MiB lower at both N.
#   P5 the memory wall lies between N = 8 and 10: >= 1 user OOMs at N = 10, and N = 12 OOMs. Refuted if N = 10 has no OOM.
#   P6 mps single-user overhead <= 5%: mps N = 1 service p50 <= 0.128 s. Refuted if > 0.128 s.
#   Step-0 gate: 4 users back to back, 40 updates each, per arm. If the ts ceiling is >= 14 updates/s, stop (exit 2):
#      0.5 s would put the ts edge on the memory wall; fix capacity.py:77 and re-register at 0.25 s as a new entry.
#      A step-0 run where any user exits non-zero or another process used the GPU is redone.
#
# Usage: bash experiments/capacity/run_G3_k2-mps.sh           (resumable: finished, clean runs are reused)
#        bash experiments/capacity/run_G3_k2-mps.sh metrics   (re-derive metrics.json from the finished runs)
# Start it only after any long co-tenant job has finished, not merely paused between stages:
# the 30 s idle wait below cannot tell a finished job from a gap between its stages.
# Env overrides: ANYSPLAT_PY (python that runs the sweep), G3_OUT (output root).
set -u
cd "$(dirname "$0")/../.."
PY=${ANYSPLAT_PY:-$HOME/Desktop/event_camera_2DGS/envs/anysplat/bin/python}
OUT=${G3_OUT:-outputs/capacity/G3}
COMMON=(--k 2 --warmup 3)

metrics() {  # writes $OUT/metrics.json: every point's numbers, step-0 ceilings, capacity per arm
  python3 - "$OUT" <<'EOF'
import json, statistics, sys
from pathlib import Path
out = Path(sys.argv[1])

def updates_per_s(run):
    rows = [json.loads(line) for log in sorted(run.glob("user*.jsonl")) for line in log.read_text().splitlines()]
    rows = [row for row in rows if row["event"] == "update"]
    return len(rows) / (max(row["end"] for row in rows) - min(row["start"] for row in rows)) if rows else None

def point(run):
    summary = json.loads((run / "summary.json").read_text())
    users = summary["users"]
    lat_p95 = [user["lat_p95"] for user in users]
    svc_p50 = [user["svc_p50"] for user in users if user["svc_p50"] is not None]
    vram = sorted(summary["proc_vram_mib_max"][str(user["pid"])] for user in users
                  if str(user["pid"]) in summary["proc_vram_mib_max"])
    passed = summary["all_meet_slo"] and not summary["foreign_procs"]
    return {"pass": passed, "pass_p95_0.5s": passed and all(value <= 0.5 for value in lat_p95),
            "contaminated": bool(summary["foreign_procs"]),
            "lat_p95_max": max((value for value in lat_p95 if value is not None), default=None),
            "svc_p50_mean": statistics.mean(svc_p50) if svc_p50 else None,
            "updates_min": min(user["updates"] for user in users), "expected": summary["expected"],
            "oom_users": sum(user["oom"] for user in users),
            "crashed_users": sum(user["rc"] not in (0, 3) for user in users),
            "updates_per_s": updates_per_s(run),
            "user_vram_mib_min_median_max": [vram[0], statistics.median(vram), vram[-1]] if vram else None,
            "gpu_mem_mib_max": summary["gpu_mem_mib_max"], "gpu_util_mean": summary["gpu_util_mean"]}

metrics = {"step0_updates_per_s": {}, "capacity": {}, "points": {}}
for arm in ("ts", "mps"):
    if (out / "step0" / arm / "summary.json").exists():
        metrics["step0_updates_per_s"][arm] = updates_per_s(out / "step0" / arm)
    points = {}
    for summary_path in (out / arm).glob("n*/r*/summary.json"):
        run = summary_path.parent
        points.setdefault(int(run.parent.name[1:]), {})[run.name] = point(run)
    metrics["points"][arm] = {str(users): points[users] for users in sorted(points)}
    both = [users for users, reps in points.items() if len(reps) == 2 and all(rep["pass"] for rep in reps.values())]
    either = [users for users, reps in points.items() if any(rep["pass"] for rep in reps.values())]
    metrics["capacity"][arm] = {"both_repeats": max(both, default=0), "any_repeat": max(either, default=0)}
(out / "metrics.json").write_text(json.dumps(metrics, indent=1))
print("capacity", json.dumps(metrics["capacity"]), "step0_updates_per_s", json.dumps(metrics["step0_updates_per_s"]))
EOF
}

if [ "${1:-}" = metrics ]; then metrics; exit; fi

wait_free() {  # 30 s of no compute process on the GPU
  local idle=0
  until [ "$idle" -ge 6 ]; do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then idle=$((idle + 1)); else idle=0; fi
    sleep 5
  done
}

state() {  # contaminated | crashed | clean
  python3 - "$1/summary.json" <<'EOF'
import json, sys
summary = json.load(open(sys.argv[1]))
print("contaminated" if summary["foreign_procs"] else
      "crashed" if any(user["rc"] != 0 for user in summary["users"]) else "clean")
EOF
}

verdict() {  # pass|fail, then the numbers
  python3 - "$1/summary.json" <<'EOF'
import json, sys
summary = json.load(open(sys.argv[1])); users = summary["users"]
print("pass" if summary["all_meet_slo"] else "fail", "p95max=%.3f" % max((user["lat_p95"] or 99) for user in users),
      "upd_min=%d/%d" % (min(user["updates"] for user in users), summary["expected"]),
      "oom=%d" % sum(user["oom"] for user in users), "util=%.0f" % (summary["gpu_util_mean"] or -1))
EOF
}

measure() {  # measure <run-dir> "<states to redo>" <capacity.py sweep args...>; up to 5 attempts
  local run=$1 redo=$2 attempt result
  shift 2
  for attempt in 1 2 3 4 5 6; do
    if [ -f "$run/summary.json" ]; then
      result=$(state "$run")
      case " $redo " in *" $result "*) ;; *) return 0 ;; esac
      mv "$run" "$OUT/_redo/${run//\//_}_${result}_$(date +%s)"
    elif [ -d "$run" ]; then
      mv "$run" "$OUT/_aborted/${run//\//_}_$(date +%s)"
    fi
    [ "$attempt" = 6 ] && break
    wait_free
    mkdir -p "$(dirname "$run")"
    "$PY" -B bench/capacity.py sweep "$@" --run-dir "$run" > "$run.out" 2>&1
  done
  echo "gave up on $run after 5 attempts"
  return 1
}

if pgrep -u "$(id -u)" -f nvidia-cuda-mps >/dev/null; then
  echo "an MPS daemon of ours is still running (a killed run?); quit it first, or wait_free never returns"
  exit 1
fi
mkdir -p "$OUT/_redo" "$OUT/_aborted"

for arm in ts mps; do
  flag=(); [ "$arm" = mps ] && flag=(--mps)
  measure "$OUT/step0/$arm" "contaminated crashed" --n 4 "${COMMON[@]}" --period 0 --max-updates 40 "${flag[@]}" || exit 1
done
metrics
if python3 -c 'import json, sys; sys.exit(json.load(open(sys.argv[1]))["step0_updates_per_s"]["ts"] < 14)' "$OUT/metrics.json"; then
  echo "GATE: ts ceiling >= 14 updates/s; period 0.5 s would hit the memory wall. Stop and re-register (see header)."
  exit 2
fi

declare -A fails=([ts]=0 [mps]=0)
for users in 1 2 3 4 5 6 7 8 10 12; do
  for arm in ts mps; do
    [ "${fails[$arm]}" -ge 2 ] && continue
    flag=(); [ "$arm" = mps ] && flag=(--mps)
    passed=0
    for rep in 1 2; do
      run=$OUT/$arm/n$users/r$rep
      measure "$run" contaminated --n "$users" "${COMMON[@]}" --period 0.5 --duration 60 "${flag[@]}" || continue
      result=$(verdict "$run")
      echo "$arm n=$users r$rep $result"
      [ "${result%% *}" = pass ] && passed=1
    done
    if [ "$passed" = 1 ]; then fails[$arm]=0; else fails[$arm]=$((fails[$arm] + 1)); fi
  done
  [ "${fails[ts]}" -ge 2 ] && [ "${fails[mps]}" -ge 2 ] && break
done
metrics
echo "G3 done"
