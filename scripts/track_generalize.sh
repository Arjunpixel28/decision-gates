#!/usr/bin/env bash
# Generalization check: standard HumanEval (undegraded) + MBPP, baseline vs
# gated judge v1. Tests whether over-questioning on already-complete specs
# hurts the gate — the honest risk flagged in GATE1_TEST_REPORT.md.
# LiveCodeBench tasks are loaded but not graded (needs a stdin/stdout
# harness, not the assert-based runner used here — see 09_generalize_bench.py).
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
exec 204>"$ROOT/.track_generalize.lock"
flock -n 204 || { echo "track_generalize already running — exiting"; exit 0; }

# Shared GPU-exclusivity lock: BLOCKING (waits its turn) so no two tracks
# ever touch the GPU at once, however they were launched (cron or manual).
exec 209>"$ROOT/.gpu_exclusive.lock"
flock 209

VENV=/mnt/data/decision-gates/venv/bin
ADAPTER_V1=/mnt/data/decision-gates/runs/gate1-dpo/final
LOGS=$ROOT/logs
STAMPS=$ROOT/data/stamps
export HF_HOME=/mnt/data/decision-gates/hf-cache

say()    { echo "[$(date '+%F %T')] [GEN] $*"; }
done_f() { [ -f "$STAMPS/$1.done" ]; }
mark()   { touch "$STAMPS/$1.done"; say "stage $1 complete"; }
free_gpu() {
  ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | while read -r m; do
    [ -n "$m" ] && ollama stop "$m" 2>/dev/null
  done
  sleep 5
}

say "===== generalization run (re)start ====="

for ds in humaneval mbpp; do
  if ! done_f "gen_${ds}_baseline"; then
    say "=== $ds baseline ==="
    $VENV/python gate1/09_generalize_bench.py --dataset "$ds" --arm baseline \
      >> "$LOGS/gen_${ds}_baseline.log" 2>&1
    [ -s "data/generalize/results_${ds}_baseline.jsonl" ] && mark "gen_${ds}_baseline" \
      || say "WARN: $ds baseline produced no output — will retry next tick"
  fi
  if ! done_f "gen_${ds}_gated"; then
    say "=== $ds gated (judge v1) ==="
    free_gpu
    $VENV/python gate1/09_generalize_bench.py --dataset "$ds" --arm gated --adapter "$ADAPTER_V1" \
      >> "$LOGS/gen_${ds}_gated.log" 2>&1
    [ -s "data/generalize/results_${ds}_gated.jsonl" ] && mark "gen_${ds}_gated" \
      || say "WARN: $ds gated produced no output — will retry next tick"
  fi
done

say "=== computing generalization summary ==="
$VENV/python - <<'EOF' > data/humaneval/generalization_summary.md 2>> "$LOGS/gen_errors.log"
import json
from pathlib import Path
GEN = Path("data/generalize")

def stats(fn):
    p = GEN / fn
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.open() if l.strip()]
    if not rows:
        return None
    ok = sum(1 for r in rows if r.get("passed"))
    asked = [r for r in rows if "asked" in r]
    ask_rate = sum(r["asked"] for r in asked) / len(asked) if asked else None
    return ok, len(rows), ask_rate

print("| dataset | baseline pass@1 | gated v1 pass@1 | gated ask-rate (spec already complete) |")
print("|---|---|---|---|")
for ds in ["humaneval", "mbpp"]:
    b = stats(f"results_{ds}_baseline.jsonl")
    g = stats(f"results_{ds}_gated.jsonl")
    bf = f"{b[0]}/{b[1]}={b[0]/b[1]:.3f}" if b else "pending"
    gf = f"{g[0]}/{g[1]}={g[0]/g[1]:.3f}" if g else "pending"
    ar = f"{g[2]:.3f}" if g and g[2] is not None else "n/a"
    print(f"| {ds} (standard, complete specs) | {bf} | {gf} | {ar} |")
print("\nLiveCodeBench: loaded but not graded here (needs a stdin/stdout "
      "harness — see 09_generalize_bench.py note).")
print("\nInterpretation: on COMPLETE specs, a high gated ask-rate here would "
      "mean the judge over-asks when it shouldn't — report this honestly "
      "whichever way it points; it is the direct test of the over-questioning "
      "risk already visible in judge v1's held-out metrics (~59-60%).")
EOF
say "GENERALIZATION RUN DONE"
echo GENERALIZE_COMPLETE
