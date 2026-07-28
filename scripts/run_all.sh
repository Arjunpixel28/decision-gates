#!/usr/bin/env bash
# ============================================================================
# ONE-SHOT PIPELINE: leave this running; it takes the project to the end goal.
#
#   stage 1  wait for DPO pair building to finish
#   stage 2  train the Gate 1 judge (QLoRA DPO, Qwen2.5-Coder-7B, this GPU)
#   stage 3  judge-quality eval on frozen held-out set: baseline vs gated
#   stage 4  real SWE-bench benchmark: baseline arm vs gated arm
#   stage 5  write FINAL_REPORT.md
#
# Launch:  nohup bash scripts/run_all.sh > ~/decision-gates/run_all.log 2>&1 &
# Logs, one per stage, all under ~/decision-gates/logs/
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
VENV=/mnt/data/decision-gates/venv/bin
RUNS=/mnt/data/decision-gates/runs
ADAPTER=$RUNS/gate1-dpo/final
LOGS=$ROOT/logs
mkdir -p "$LOGS" "$RUNS"
export HF_HOME=/mnt/data/decision-gates/hf-cache
export WANDB_MODE=offline
GEN_MODEL="qwen2.5:14b-instruct-q4_K_M"

stamp() { date '+%F %T'; }
say()   { echo "[$(stamp)] $*"; }
free_gpu() {  # unload ALL resident ollama models (anything can get loaded by
              # other users/processes on this shared box); service config untouched
  ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | while read -r m; do
    [ -n "$m" ] && ollama stop "$m" 2>/dev/null
  done
  sleep 5
}

say "=== STAGE 1: waiting for DPO pairs ==="
while :; do
  grep -q DATA_PHASE_COMPLETE "$ROOT/data_phase.log" 2>/dev/null && break
  if ! pgrep -f 03_build_dpo_pairs >/dev/null; then
    say "pair builder not running and not complete — restarting it"
    (cd gate1 && nohup bash -c "python3 03_build_dpo_pairs.py && echo DATA_PHASE_COMPLETE >> $ROOT/data_phase.log" >> "$LOGS/pairs.log" 2>&1 &)
    sleep 30
  fi
  sleep 120
done
NPAIRS=$(wc -l < data/gate1/dpo_pairs_train.jsonl)
say "pairs ready: $NPAIRS train / $(wc -l < data/gate1/dpo_pairs_eval.jsonl) eval"
[ "$NPAIRS" -ge 100 ] || { say "FATAL: too few pairs ($NPAIRS)"; exit 1; }

say "=== STAGE 2: DPO training ==="
if [ ! -d "$ADAPTER" ]; then
  free_gpu
  $VENV/python gate1/train_gate1_dpo.py --output-dir "$RUNS/gate1-dpo" \
    > "$LOGS/train.log" 2>&1
  if [ ! -d "$ADAPTER" ]; then say "FATAL: training produced no adapter — see logs/train.log"; exit 1; fi
  say "training done"
else
  say "adapter already exists — skipping training"
fi

say "=== STAGE 3: judge-quality eval (baseline vs gated) ==="
# 3a. generation passes (judge model owns the GPU; ollama stays unloaded)
free_gpu
$VENV/python gate1/eval_gate1.py --phase generate --tag baseline \
  > "$LOGS/eval_generate_baseline.log" 2>&1 || say "WARN: baseline generate failed"
$VENV/python gate1/eval_gate1.py --phase generate --tag gated --adapter "$ADAPTER" \
  > "$LOGS/eval_generate_gated.log" 2>&1 || say "WARN: gated generate failed"
# 3b. scoring passes (ollama reloads on demand)
$VENV/python gate1/eval_gate1.py --phase judge --tag baseline \
  > "$LOGS/eval_score_baseline.log" 2>&1 || say "WARN: baseline scoring failed"
$VENV/python gate1/eval_gate1.py --phase judge --tag gated \
  > "$LOGS/eval_score_gated.log" 2>&1 || say "WARN: gated scoring failed"
say "judge-quality eval done"

say "=== STAGE 4: SWE-bench downstream benchmark ==="
# baseline arm: generator only (ollama)
$VENV/python gate1/04_downstream_eval.py --arm baseline \
  > "$LOGS/bench_baseline.log" 2>&1 || say "WARN: baseline bench had errors"
# gated arm: judge pass needs the GPU first, then generator
free_gpu
$VENV/python gate1/04_downstream_eval.py --arm gated --adapter "$ADAPTER" \
  > "$LOGS/bench_gated.log" 2>&1 || say "WARN: gated bench had errors"
say "benchmark done"

say "=== STAGE 5: final report ==="
$VENV/python - <<'EOF' > "$ROOT/FINAL_REPORT.md" 2>> "$LOGS/report_errors.log"
import json
from pathlib import Path

H = Path("data/heldout")
print("# Gate 1 Judge — Final Results\n")

print("## Judge quality (frozen held-out set)\n")
print("| metric | baseline (untrained) | gated (trained judge) |")
print("|---|---|---|")
try:
    b = json.loads((H / "eval_baseline.json").read_text())
    g = json.loads((H / "eval_gated.json").read_text())
    for k in ["ask_rate_underspecified", "under_questioning_rate",
              "over_questioning_rate", "requirement_recovery_rate"]:
        print(f"| {k} | {b[k]} | {g[k]} |")
except Exception as e:
    print(f"\n(judge-quality eval incomplete: {e})")

print("\n## SWE-bench downstream (real test execution)\n")
print("| arm | resolved | ran | resolve rate |")
print("|---|---|---|---|")
for arm in ["baseline", "gated"]:
    p = H / f"downstream_{arm}.jsonl"
    if p.exists():
        rows = [json.loads(l) for l in p.open() if l.strip()]
        ran = [r for r in rows if r.get("status") in ("ran", "no_edits_applied")]
        ok = sum(1 for r in ran if r.get("resolved"))
        rate = f"{ok/len(ran):.1%}" if ran else "n/a"
        print(f"| {arm} | {ok} | {len(ran)} | {rate} |")
    else:
        print(f"| {arm} | - | - | missing |")
EOF
say "ALL STAGES DONE — read $ROOT/FINAL_REPORT.md, logs in $LOGS/"
echo PIPELINE_COMPLETE
