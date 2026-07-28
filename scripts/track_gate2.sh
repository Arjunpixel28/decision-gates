#!/usr/bin/env bash
# ============================================================================
# TRACK GATE2 — plan judge, end to end, unattended.
#   g1 generate K=5 distinct plans per HumanEval task (n=155, same pool as G1)
#   g2 rank plans pairwise (LLM-judge) -> DPO pairs + hand-validate sample
#   g3 convert to chat format for the (proven) Gate 1 trainer
#   g4 train Gate 2 judge (DPO, QLoRA, same recipe/config as Gate 1)
#   g5 benchmark: gate2-only arm, then gate1+gate2 combined arm
#   g6 rewrite FINAL_REPORT.md with the Gate 2 + combined results appended
# Crash/reboot-safe: own flock + stamps; picked up by the existing watchdog
# (resume_pipeline.sh) automatically — no separate cron entry needed.
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
exec 202>"$ROOT/.track_gate2.lock"
flock -n 202 || { echo "track_gate2 already running — exiting"; exit 0; }

# Shared GPU-exclusivity lock: BLOCKING (waits its turn) so no two tracks
# ever touch the GPU at once, however they were launched (cron or manual).
exec 209>"$ROOT/.gpu_exclusive.lock"
flock 209

VENV=/mnt/data/decision-gates/venv/bin
RUNS=/mnt/data/decision-gates/runs
ADAPTER_G1=$RUNS/gate1-dpo/final
ADAPTER_G2=$RUNS/gate2-dpo/final
LOGS=$ROOT/logs
STAMPS=$ROOT/data/stamps
mkdir -p "$LOGS" "$STAMPS" data/gate2
export HF_HOME=/mnt/data/decision-gates/hf-cache
export WANDB_MODE=offline

say()    { echo "[$(date '+%F %T')] [G2] $*"; }
done_f() { [ -f "$STAMPS/$1.done" ]; }
mark()   { touch "$STAMPS/$1.done"; say "stage $1 complete"; }
free_gpu() {
  ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | while read -r m; do
    [ -n "$m" ] && ollama stop "$m" 2>/dev/null
  done
  sleep 5
}
wait_ollama() {
  for _ in $(seq 1 30); do
    curl -s -m 5 http://127.0.0.1:11434/api/tags >/dev/null && return 0
    sleep 10
  done
}

say "===== track Gate2 (re)start ====="
wait_ollama

if ! done_f g1_plans; then
  say "=== G1: generate candidate plans ==="
  $VENV/python gate2/01_generate_plans.py >> "$LOGS/g2_plans.log" 2>&1
  [ -s data/gate2/plans.jsonl ] && mark g1_plans
fi

if ! done_f g2_rank; then
  say "=== G2: rank plans -> DPO pairs ==="
  $VENV/python gate2/02_rank_plans.py >> "$LOGS/g2_rank.log" 2>&1
  NP=$(wc -l < data/gate2/dpo_pairs_train.jsonl 2>/dev/null || echo 0)
  say "gate2 pairs: $NP"
  [ "$NP" -ge 50 ] && mark g2_rank || { say "FATAL: too few gate2 pairs ($NP)"; exit 1; }
fi

if ! done_f g3_format; then
  say "=== G3: convert to chat DPO format ==="
  $VENV/python gate2/03_prep_dpo_format.py >> "$LOGS/g2_format.log" 2>&1
  mark g3_format
fi

if ! done_f g4_train; then
  say "=== G4: train Gate 2 judge ==="
  free_gpu
  # reuse Gate 1's trainer unmodified, pointed at gate2 data
  $VENV/python gate1/train_gate1_dpo.py --pairs-dir data/gate2/train_ready \
    --output-dir "$RUNS/gate2-dpo" --max-len 1536 >> "$LOGS/train_gate2.log" 2>&1
  [ -d "$ADAPTER_G2" ] && mark g4_train || { say "FATAL: gate2 training produced no adapter"; exit 1; }
fi

if ! done_f g5_bench_gate2_only; then
  say "=== G5: benchmark gate2-only arm ==="
  free_gpu
  $VENV/python gate2/04_bench_gate2.py --arm gate2_only --g2-adapter "$ADAPTER_G2" \
    >> "$LOGS/bench_gate2_only.log" 2>&1
  mark g5_bench_gate2_only
fi

if ! done_f g5_bench_combined; then
  say "=== G5: benchmark gate1+gate2 combined arm ==="
  free_gpu
  $VENV/python gate2/04_bench_gate2.py --arm combined \
    --g1-adapter "$ADAPTER_G1" --g2-adapter "$ADAPTER_G2" \
    >> "$LOGS/bench_combined.log" 2>&1
  mark g5_bench_combined
fi

say "=== G6: append Gate 2 results to FINAL_REPORT.md ==="
$VENV/python - <<'EOF' >> "$ROOT/FINAL_REPORT.md" 2>> "$LOGS/report_errors.log"
import json
from pathlib import Path
HE = Path("data/humaneval")

print("\n## Gate 2 (Plan Judge) + combined pipeline — same HumanEval pool (n=155)\n")
print("| arm | pass@1 | n |")
print("|---|---|---|")
rows_ref = [("baseline (no gate)", "results_baseline.jsonl"),
            ("gated, judge v1 only (Gate 1)", "results_gated.jsonl"),
            ("gate 2 only (plan judge)", "results_gate2_only.jsonl"),
            ("gate 1 + gate 2 combined", "results_combined.jsonl")]
for arm, fn in rows_ref:
    p = HE / fn
    if p.exists():
        rows = [json.loads(l) for l in p.open() if l.strip()]
        ok = sum(1 for r in rows if r.get("passed"))
        print(f"| {arm} | {ok}/{len(rows)} = {ok/max(len(rows),1):.3f} | {len(rows)} |")
    else:
        print(f"| {arm} | pending | - |")
print("\nHand-validation sample for Gate 2 ranking agreement: "
      "data/gate2/hand_validate_sample.jsonl (~15% of ranked tasks, "
      "manually check agreement with the LLM-judge ranking before citing "
      "Gate 2 as a validated contribution).")
EOF
say "TRACK GATE2 DONE"
echo TRACK_GATE2_COMPLETE
