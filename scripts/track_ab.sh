#!/usr/bin/env bash
# ============================================================================
# TRACK A: trustworthy benchmark (gold-validated subset, partial credit)
# TRACK B: judge v2 at 10x data with harder negatives
#
# CRASH/REBOOT SAFE:
#   - flock guard: only one instance can ever run
#   - every stage is stamped in data/stamps/ and skipped once complete
#   - inner scripts resume midway (degrade skips done ids, training resumes
#     from checkpoints, benchmark skips graded instances)
#   - cron (@reboot + every 15 min) relaunches this script; the flock and
#     stamps make that a no-op unless something actually died
# Launch:  bash scripts/resume_pipeline.sh   (or directly via nohup)
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD

exec 200>"$ROOT/.track_ab.lock"
flock -n 200 || { echo "already running — exiting"; exit 0; }

# Shared GPU-exclusivity lock: BLOCKING (waits its turn) so no two tracks
# ever touch the GPU at once, however they were launched (cron or manual).
exec 209>"$ROOT/.gpu_exclusive.lock"
flock 209

VENV=/mnt/data/decision-gates/venv/bin
RUNS=/mnt/data/decision-gates/runs
ADAPTER_V1=$RUNS/gate1-dpo/final
ADAPTER_V2=$RUNS/gate1-dpo-v2/final
LOGS=$ROOT/logs
STAMPS=$ROOT/data/stamps
mkdir -p "$LOGS" "$STAMPS" data/gate1_v2
export HF_HOME=/mnt/data/decision-gates/hf-cache
export WANDB_MODE=offline

say()    { echo "[$(date '+%F %T')] $*"; }
done_f() { [ -f "$STAMPS/$1.done" ]; }
mark()   { touch "$STAMPS/$1.done"; say "stage $1 complete"; }
free_gpu() {
  ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | while read -r m; do
    [ -n "$m" ] && ollama stop "$m" 2>/dev/null
  done
  sleep 5
}
wait_ollama() {  # after reboot ollama may still be starting
  for _ in $(seq 1 30); do
    curl -s -m 5 http://127.0.0.1:11434/api/tags >/dev/null && return 0
    sleep 10
  done
  say "WARN: ollama not answering after 5 min"
}

say "===== pipeline (re)start ====="
wait_ollama

# ---------------- TRACK A ----------------
if ! done_f a1_gold_validate; then
  say "=== A1: gold-validate held-out instances ==="
  $VENV/python gate1/05_gold_validate.py >> "$LOGS/gold_validate.log" 2>&1 \
    && [ -s data/heldout/gold_validated.json ] && mark a1_gold_validate
fi

if ! done_f a2_archive_old; then
  mv -f data/heldout/downstream_baseline.jsonl data/heldout/downstream_baseline.allinst.bak 2>/dev/null
  mv -f data/heldout/downstream_gated.jsonl data/heldout/downstream_gated.allinst.bak 2>/dev/null
  mark a2_archive_old
fi

if ! done_f a2_baseline; then
  say "=== A2: baseline arm (validated subset, partial credit) ==="
  $VENV/python gate1/04_downstream_eval.py --arm baseline --validated-only \
    >> "$LOGS/bench_baseline.log" 2>&1 && mark a2_baseline
fi

if ! done_f a2_gated_v1; then
  say "=== A2: gated v1 arm ==="
  free_gpu
  $VENV/python gate1/04_downstream_eval.py --arm gated --adapter "$ADAPTER_V1" --validated-only \
    >> "$LOGS/bench_gated.log" 2>&1 && mark a2_gated_v1
fi

# ---------------- TRACK B ----------------
if ! done_f b1_collect; then
  say "=== B1: collect v2 tasks (SWE-bench train split) ==="
  $VENV/python gate1/01_collect_tasks.py --dataset princeton-nlp/SWE-bench --split train \
    --max-tasks 2500 --heldout-frac 0 --out-dir data/gate1_v2 >> "$LOGS/collect_v2.log" 2>&1 \
    && [ -s data/gate1_v2/tasks_train.jsonl ] && mark b1_collect
fi

if ! done_f b2_degrade; then
  say "=== B2: degrade v2 tasks (long; resumes automatically) ==="
  $VENV/python gate1/02_degrade_tasks.py --tasks data/gate1_v2/tasks_train.jsonl \
    --out data/gate1_v2/degraded.jsonl >> "$LOGS/degrade_v2.log" 2>&1 && mark b2_degrade
fi

if ! done_f b3_pairs; then
  say "=== B3: build v2 DPO pairs (harder negatives) ==="
  if [ -s data/gate1_v2/dpo_pairs_train.jsonl ]; then
    mark b3_pairs   # already built before a crash
  else
    $VENV/python gate1/03_build_dpo_pairs.py --degraded data/gate1_v2/degraded.jsonl \
      --out-dir data/gate1_v2 >> "$LOGS/pairs_v2.log" 2>&1
    NP=$(wc -l < data/gate1_v2/dpo_pairs_train.jsonl 2>/dev/null || echo 0)
    say "v2 pairs: $NP"
    [ "$NP" -ge 1000 ] && mark b3_pairs || { say "FATAL: too few v2 pairs ($NP)"; exit 1; }
  fi
fi

if ! done_f b4_train; then
  say "=== B4: train judge v2 (resumes from checkpoints) ==="
  free_gpu
  # --max-len 1536: the subtitle service squats ~6GB VRAM; 2048 OOMs on long
  # batches with only ~17.6GB left. Truncation affects few pairs.
  $VENV/python gate1/train_gate1_dpo.py --pairs-dir data/gate1_v2 \
    --output-dir "$RUNS/gate1-dpo-v2" --epochs 2 --max-len 1536 >> "$LOGS/train_v2.log" 2>&1
  [ -d "$ADAPTER_V2" ] && mark b4_train || { say "FATAL: v2 training produced no adapter"; exit 1; }
fi

if ! done_f b5_eval_generate; then
  say "=== B5: v2 judge-quality eval (generate) ==="
  free_gpu
  $VENV/python gate1/eval_gate1.py --phase generate --tag gated_v2 --adapter "$ADAPTER_V2" \
    >> "$LOGS/eval_generate_gated_v2.log" 2>&1 && mark b5_eval_generate
fi

if ! done_f b5_eval_score; then
  say "=== B5: v2 judge-quality eval (score) ==="
  wait_ollama
  $VENV/python gate1/eval_gate1.py --phase judge --tag gated_v2 \
    >> "$LOGS/eval_score_gated_v2.log" 2>&1 && mark b5_eval_score
fi

if ! done_f b5_bench_v2; then
  say "=== B5: gated v2 benchmark arm ==="
  free_gpu
  $VENV/python gate1/04_downstream_eval.py --arm gated --adapter "$ADAPTER_V2" \
    --validated-only --out-suffix _v2 >> "$LOGS/bench_gated_v2.log" 2>&1 && mark b5_bench_v2
fi

say "=== B6: combined report ==="
$VENV/python - <<'EOF' > "$ROOT/FINAL_REPORT.md" 2>> "$LOGS/report_errors.log"
import json
from pathlib import Path

H = Path("data/heldout")
print("# Decision-Gate Judge — Combined Results (Tracks A+B)\n")

print("## Judge quality (frozen held-out set)\n")
print("| metric | baseline | judge v1 (276 pairs) | judge v2 (10x, hard negs) |")
print("|---|---|---|---|")
try:
    rows = {t: json.loads((H / f"eval_{t}.json").read_text())
            for t in ["baseline", "gated", "gated_v2"]}
    for k in ["ask_rate_underspecified", "under_questioning_rate",
              "over_questioning_rate", "requirement_recovery_rate"]:
        print(f"| {k} | {rows['baseline'][k]} | {rows['gated'][k]} | {rows['gated_v2'][k]} |")
except Exception as e:
    print(f"\n(incomplete: {e})")

print("\n## SWE-bench downstream — gold-validated subset, partial credit\n")
print("| arm | resolved | mean score | n |")
print("|---|---|---|---|")
for arm, fn in [("baseline", "downstream_baseline.jsonl"),
                ("gated v1", "downstream_gated.jsonl"),
                ("gated v2", "downstream_gated_v2.jsonl")]:
    p = H / fn
    if p.exists():
        rows = [json.loads(l) for l in p.open() if l.strip()]
        ran = [r for r in rows if r.get("status") in ("ran", "no_edits_applied")]
        ok = sum(1 for r in ran if r.get("resolved"))
        ms = sum(r.get("score", 0.0) for r in ran) / len(ran) if ran else 0.0
        print(f"| {arm} | {ok} | {ms:.3f} | {len(ran)} |")
    else:
        print(f"| {arm} | - | - | missing |")
EOF
say "ALL DONE — FINAL_REPORT.md written"
echo TRACKS_AB_COMPLETE
