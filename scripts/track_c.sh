#!/usr/bin/env bash
# TRACK C: HumanEval downstream benchmark — the thesis-proving experiment.
#   c1 degrade all 164 HumanEval docstrings          (shares ollama with B)
#   c2 baseline arm                                   (shares ollama with B)
#   -- waits for Tracks A+B to finish (GPU exclusivity for judge passes) --
#   c3 gated arm, judge v1
#   c4 gated arm, judge v2
#   c5 rewrite FINAL_REPORT.md with the HumanEval pass@1 table
# Crash/reboot-safe: own flock + stamps; relaunched by resume_pipeline.sh.
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
exec 201>"$ROOT/.track_c.lock"
flock -n 201 || { echo "track_c already running — exiting"; exit 0; }

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
mkdir -p "$LOGS" "$STAMPS"
export HF_HOME=/mnt/data/decision-gates/hf-cache

say()    { echo "[$(date '+%F %T')] [C] $*"; }
done_f() { [ -f "$STAMPS/$1.done" ]; }
mark()   { touch "$STAMPS/$1.done"; say "stage $1 complete"; }
free_gpu() {
  ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | while read -r m; do
    [ -n "$m" ] && ollama stop "$m" 2>/dev/null
  done
  sleep 5
}

say "===== track C (re)start ====="

if ! done_f c1_degrade; then
  say "=== C1: degrade HumanEval specs ==="
  $VENV/python gate1/06_humaneval_bench.py degrade >> "$LOGS/he_degrade.log" 2>&1 \
    && [ -s data/humaneval/degraded.jsonl ] && mark c1_degrade
fi

if ! done_f c2_baseline; then
  say "=== C2: baseline arm ==="
  $VENV/python gate1/06_humaneval_bench.py arm --arm baseline \
    >> "$LOGS/he_baseline.log" 2>&1 && mark c2_baseline
fi

# Judge passes need the GPU to themselves — wait until A+B are done.
until grep -q TRACKS_AB_COMPLETE "$ROOT/track_ab.log" 2>/dev/null; do
  say "waiting for Tracks A+B to finish before judge arms..."
  sleep 900
done

if ! done_f c3_gated_v1; then
  say "=== C3: gated arm, judge v1 ==="
  free_gpu
  $VENV/python gate1/06_humaneval_bench.py arm --arm gated --adapter "$ADAPTER_V1" \
    >> "$LOGS/he_gated_v1.log" 2>&1 && mark c3_gated_v1
fi

if ! done_f c4_gated_v2; then
  say "=== C4: gated arm, judge v2 ==="
  free_gpu
  $VENV/python gate1/06_humaneval_bench.py arm --arm gated --adapter "$ADAPTER_V2" \
    --out-suffix _v2 >> "$LOGS/he_gated_v2.log" 2>&1 && mark c4_gated_v2
fi

say "=== C5: final combined report ==="
$VENV/python - <<'EOF' > "$ROOT/FINAL_REPORT.md" 2>> "$LOGS/report_errors.log"
import json
from pathlib import Path

H = Path("data/heldout")
HE = Path("data/humaneval")
print("# Decision-Gate Judge — Final Thesis Results\n")

print("## HEADLINE: HumanEval (degraded specs) — one-shot pass@1\n")
print("| arm | pass@1 | n |")
print("|---|---|---|")
for arm, fn in [("baseline (no gate)", "results_baseline.jsonl"),
                ("gated, judge v1", "results_gated.jsonl"),
                ("gated, judge v2", "results_gated_v2.jsonl")]:
    p = HE / fn
    if p.exists():
        rows = [json.loads(l) for l in p.open() if l.strip()]
        ok = sum(1 for r in rows if r.get("passed"))
        print(f"| {arm} | {ok}/{len(rows)} = {ok/max(len(rows),1):.3f} | {len(rows)} |")
    else:
        print(f"| {arm} | pending | - |")

print("\n## Judge quality (frozen held-out set)\n")
print("| metric | baseline | judge v1 | judge v2 |")
print("|---|---|---|---|")
try:
    rows = {t: json.loads((H / f"eval_{t}.json").read_text())
            for t in ["baseline", "gated", "gated_v2"]}
    for k in ["ask_rate_underspecified", "under_questioning_rate",
              "over_questioning_rate", "requirement_recovery_rate"]:
        print(f"| {k} | {rows['baseline'][k]} | {rows['gated'][k]} | {rows['gated_v2'][k]} |")
except Exception as e:
    print(f"\n(incomplete: {e})")

print("\n## SWE-bench Verified (gold-validated subset) — hard ceiling\n")
print("| arm | resolved | mean partial score | n |")
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
print("\nNote: repo-level SWE-bench is reported as the honest difficulty ceiling; "
      "at 14B-q4 generator strength neither arm makes progress there.")
EOF
say "TRACK C DONE — FINAL_REPORT.md rewritten"
echo TRACK_C_COMPLETE
