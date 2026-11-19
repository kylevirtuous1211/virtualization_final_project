# Shell functions shared by the capacity runners from G5 on; G3's runner keeps its own copies, as registered.
# A runner sets PY (python that runs bench/capacity.py sweep) and OUT (its output root), cds to the repository
# root, then sources this file. Every function writes under $OUT.

check_no_mps() {  # a leftover MPS daemon of ours keeps the GPU busy, so wait_free would never return
  if pgrep -u "$(id -u)" -f nvidia-cuda-mps >/dev/null; then
    echo "an MPS daemon of ours is still running (a killed run?); quit it first"
    exit 1
  fi
  mkdir -p "$OUT/_redo" "$OUT/_aborted"
}

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

sweep_arms() {  # sweep_arms "<N grid>" <capacity.py sweep args...>: the arms in $ARMS (default "ts mps"); arm fast
  # is the batch-1 server with the runner's $FAST_FLAGS, arm depthbf16 the same plus --depth-bf16 (G16)
  # interleaved per N, 2 repeats each; an arm stops after two consecutive N where neither repeat passes (G2's rule)
  local grid=$1 users arm rep run result passed
  shift
  local arms=(${ARMS:-ts mps})
  local -A fails=()
  for arm in "${arms[@]}"; do fails[$arm]=0; done
  for users in $grid; do
    for arm in "${arms[@]}"; do
      [ "${fails[$arm]}" -ge 2 ] && continue
      local flag=()
      case $arm in ts) ;; mps) flag=(--mps) ;; serial) flag=(--batch 1) ;; batched) flag=(--batch 8) ;;
        fast) flag=(--batch 1 $FAST_FLAGS) ;; depthbf16) flag=(--batch 1 $FAST_FLAGS --depth-bf16) ;;
        *) echo "sweep_arms: unknown arm $arm" >&2; return 1 ;; esac
      passed=0
      for rep in 1 2; do
        run=$OUT/$arm/n$users/r$rep
        measure "$run" contaminated --n "$users" "$@" "${flag[@]}" || continue
        result=$(verdict "$run")
        echo "$arm n=$users r$rep $result"
        [ "${result%% *}" = pass ] && passed=1
      done
      if [ "$passed" = 1 ]; then fails[$arm]=0; else fails[$arm]=$((fails[$arm] + 1)); fi
    done
    local stopped=1
    for arm in "${arms[@]}"; do [ "${fails[$arm]}" -ge 2 ] || stopped=0; done
    [ "$stopped" = 1 ] && break
  done
}

metrics() {  # metrics <arm>...: writes $OUT/metrics.json with every point's numbers and each arm's capacity
  python3 - "$OUT" "$@" <<'EOF'
import json, statistics, sys
from pathlib import Path
out, arms = Path(sys.argv[1]), sys.argv[2:]

def updates_per_s(run):
    rows = [json.loads(line) for log in sorted(run.glob("user*.jsonl")) for line in log.read_text().splitlines()]
    rows = [row for row in rows if row["event"] == "update"]
    return len(rows) / (max(row["end"] for row in rows) - min(row["start"] for row in rows)) if rows else None

def mean_of(users, key):
    values = [user[key] for user in users if user.get(key) is not None]
    return statistics.mean(values) if values else None

def point(run):
    summary = json.loads((run / "summary.json").read_text())
    users = summary["users"]
    lat_p95 = [user["lat_p95"] for user in users]
    vram = sorted(summary["proc_vram_mib_max"][str(user["pid"])] for user in users
                  if str(user["pid"]) in summary["proc_vram_mib_max"])
    passed = summary["all_meet_slo"] and not summary["foreign_procs"]
    return {"pass": passed, "contaminated": bool(summary["foreign_procs"]),
            "lat_p95_max": max((value for value in lat_p95 if value is not None), default=None),
            "svc_p50_mean": mean_of(users, "svc_p50"), "svc_p95_mean": mean_of(users, "svc_p95"),
            "svc_depth_p50_mean": mean_of(users, "svc_depth_p50"),
            "gaussians_p50": mean_of(users, "gaussians_p50"), "weight_bytes": mean_of(users, "weight_bytes"),
            "peak_alloc_max": max((user["peak_alloc"] for user in users if user["peak_alloc"]), default=None),
            "updates_min": min(user["updates"] for user in users), "expected": summary["expected"],
            "oom_users": sum(user["oom"] for user in users),
            "crashed_users": sum(user["rc"] not in (0, 3) for user in users),
            "updates_per_s": updates_per_s(run),
            "user_vram_mib_min_median_max": [vram[0], statistics.median(vram), vram[-1]] if vram else None,
            "gpu_mem_mib_max": summary["gpu_mem_mib_max"], "gpu_util_mean": summary["gpu_util_mean"],
            "vram_per_user_mib": summary["gpu_mem_mib_max"] / summary["n"] if summary["gpu_mem_mib_max"] else None,
            **({"batch_p50_mean": mean_of(users, "batch_p50"), "batch_max": max(user["batch_max"] or 0 for user in users)}
               if summary.get("batch") else {})}

metrics = {"capacity": {}, "points": {}}
if (out / "step0").is_dir():  # back-to-back runs, one per batch size (G9)
    metrics["step0_updates_per_s"] = {run.name: updates_per_s(run) for run in sorted((out / "step0").iterdir())
                                      if (run / "summary.json").exists()}
for arm in arms:
    points = {}
    for summary_path in (out / arm).glob("n*/r*/summary.json"):
        run = summary_path.parent
        points.setdefault(int(run.parent.name[1:]), {})[run.name] = point(run)
    metrics["points"][arm] = {str(users): points[users] for users in sorted(points)}
    both = [users for users, reps in points.items() if len(reps) == 2 and all(rep["pass"] for rep in reps.values())]
    either = [users for users, reps in points.items() if any(rep["pass"] for rep in reps.values())]
    metrics["capacity"][arm] = {"both_repeats": max(both, default=0), "any_repeat": max(either, default=0)}
(out / "metrics.json").write_text(json.dumps(metrics, indent=1))
print("capacity", json.dumps(metrics["capacity"]))
EOF
}
