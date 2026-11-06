#!/usr/bin/env bash
# G2 capacity sweep (LOG.md G2): AnySplat, k=8, one update per user per second, 60 s per point,
# 2 repeats, arms ts (time-slicing) and mps interleaved per N.
# An arm stops after two consecutive N where neither repeat meets the SLO.
# Finished, uncontaminated runs are reused, so the script can be re-run to resume.
# Every run waits for 30 s of an idle GPU first; a run that another GPU process touched is moved
# to runs/G2/_contaminated and redone; a run that never wrote summary.json is moved to runs/G2/_aborted.
set -u
cd "$(dirname "$0")/.."
PY=~/Desktop/event_camera_2DGS/envs/anysplat/bin/python

wait_free() {
  local idle=0
  until [ "$idle" -ge 6 ]; do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then idle=$((idle + 1)); else idle=0; fi
    sleep 5
  done
}

verdict() {  # prints: pass|fail|contaminated, then the numbers
  python3 - "$1/summary.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1])); users = s["users"]
state = "contaminated" if s["foreign_procs"] else ("pass" if s["all_meet_slo"] else "fail")
print(state, "p95max=%.3f" % max((u["lat_p95"] or 99) for u in users),
      "upd_min=%d/%d" % (min(u["updates"] for u in users), s["expected"]),
      "oom=%d" % sum(u["oom"] for u in users), "util=%.0f" % (s["gpu_util_mean"] or -1))
EOF
}

declare -A fails=([ts]=0 [mps]=0)
mkdir -p runs/G2/_contaminated runs/G2/_aborted
for n in 1 2 3 4 6 8 10 12; do
  for arm in ts mps; do
    [ "${fails[$arm]}" -ge 2 ] && continue
    passed=0
    for rep in 1 2; do
      run=runs/G2/$arm/n$n/r$rep
      for attempt in 1 2 3 4 5; do
        if [ -f "$run/summary.json" ]; then
          result=$(verdict "$run")
          [ "${result%% *}" != contaminated ] && break
          mv "$run" "runs/G2/_contaminated/${arm}_n${n}_r${rep}_$(date +%s)"
        elif [ -d "$run" ]; then
          mv "$run" "runs/G2/_aborted/${arm}_n${n}_r${rep}_$(date +%s)"
        fi
        result="error see $run.out"
        wait_free
        mkdir -p "runs/G2/$arm/n$n"
        flag=(); [ "$arm" = mps ] && flag=(--mps)
        "$PY" -B bench/capacity.py sweep --n "$n" --k 8 --period 1 --duration 60 --warmup 3 "${flag[@]}" \
          --run-dir "$run" > "$run.out" 2>&1
      done
      [ -f "$run/summary.json" ] && result=$(verdict "$run")
      echo "$arm n=$n r$rep $result"
      [ "${result%% *}" = pass ] && passed=1
    done
    if [ "$passed" = 1 ]; then fails[$arm]=0; else fails[$arm]=$((fails[$arm] + 1)); fi
  done
  [ "${fails[ts]}" -ge 2 ] && [ "${fails[mps]}" -ge 2 ] && break
done
echo "G2 done"
