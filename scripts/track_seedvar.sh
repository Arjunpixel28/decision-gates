#!/usr/bin/env bash
# Seed-variance check for the headline HumanEval result. Reuses the already-
# trained judge v1 adapter and the same 155-task pool — only the generator's
# sampling seed/temperature changes across 2 additional runs (seed=0 already
# exists as results_baseline.jsonl / results_gated.jsonl).
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
exec 203>"$ROOT/.track_seedvar.lock"
flock -n 203 || { echo "track_seedvar already running — exiting"; exit 0; }

# Shared GPU-exclusivity lock: BLOCKING (waits its turn) so no two tracks
# ever touch the GPU at once, however they were launched (cron or manual).
exec 209>"$ROOT/.gpu_exclusive.lock"
flock 209

VENV=/mnt/data/decision-gates/venv/bin
ADAPTER_V1=/mnt/data/decision-gates/runs/gate1-dpo/final
LOGS=$ROOT/logs
STAMPS=$ROOT/data/stamps
export HF_HOME=/mnt/data/decision-gates/hf-cache

say()    { echo "[$(date '+%F %T')] [SEED] $*"; }
done_f() { [ -f "$STAMPS/$1.done" ]; }
mark()   { touch "$STAMPS/$1.done"; say "stage $1 complete"; }
free_gpu() {
  ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | while read -r m; do
    [ -n "$m" ] && ollama stop "$m" 2>/dev/null
  done
  sleep 5
}

say "===== seed-variance run (re)start ====="

for seed in 1 2; do
  if ! done_f "seed${seed}_baseline"; then
    say "=== baseline, seed=$seed ==="
    $VENV/python gate1/06_humaneval_bench.py arm --arm baseline --out-suffix "_seed${seed}" \
      --seed "$seed" >> "$LOGS/seed${seed}_baseline.log" 2>&1
    [ -s "data/humaneval/results_baseline_seed${seed}.jsonl" ] && mark "seed${seed}_baseline" \
      || say "WARN: seed$seed baseline produced no output — will retry next tick"
  fi
  if ! done_f "seed${seed}_gated"; then
    say "=== gated v1, seed=$seed ==="
    free_gpu
    $VENV/python gate1/06_humaneval_bench.py arm --arm gated --adapter "$ADAPTER_V1" \
      --out-suffix "_seed${seed}" --seed "$seed" >> "$LOGS/seed${seed}_gated.log" 2>&1
    [ -s "data/humaneval/results_gated_seed${seed}.jsonl" ] && mark "seed${seed}_gated" \
      || say "WARN: seed$seed gated produced no output — will retry next tick"
  fi
done

say "=== computing variance summary ==="
$VENV/python - <<'EOF' > data/humaneval/seed_variance_summary.md 2>> "$LOGS/seedvar_errors.log"
import json
from pathlib import Path
HE = Path("data/humaneval")

def rate(fn):
    p = HE / fn
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.open() if l.strip()]
    ok = sum(1 for r in rows if r.get("passed"))
    return ok / len(rows) if rows else None

print("# Seed-Variance Summary\n")
print("| arm | seed 0 | seed 1 | seed 2 | mean | spread |")
print("|---|---|---|---|---|---|")
for arm, base in [("baseline", "results_baseline{}.jsonl"), ("gated v1", "results_gated{}.jsonl")]:
    vals = [rate(base.format("")), rate(base.format("_seed1")), rate(base.format("_seed2"))]
    valid = [v for v in vals if v is not None]
    mean = sum(valid) / len(valid) if valid else None
    spread = (max(valid) - min(valid)) if len(valid) > 1 else None
    fmt = lambda v: f"{v:.3f}" if v is not None else "n/a"
    print(f"| {arm} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} | "
          f"{fmt(mean)} | {fmt(spread)} |")
EOF
say "SEED VARIANCE DONE"
echo SEEDVAR_COMPLETE
