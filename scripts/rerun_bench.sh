#!/usr/bin/env bash
# Rerun ONLY stage 4 (SWE-bench benchmark) + stage 5 (report) with the fixed
# harness. Training and judge-quality eval results are kept as-is.
# Launch:  nohup bash scripts/rerun_bench.sh > ~/decision-gates/rerun_bench.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
VENV=/mnt/data/decision-gates/venv/bin
ADAPTER=/mnt/data/decision-gates/runs/gate1-dpo/final
LOGS=$ROOT/logs
export HF_HOME=/mnt/data/decision-gates/hf-cache

say() { echo "[$(date '+%F %T')] $*"; }
free_gpu() {
  ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | while read -r m; do
    [ -n "$m" ] && ollama stop "$m" 2>/dev/null
  done
  sleep 5
}

say "=== RERUN STAGE 4: SWE-bench benchmark (fixed harness) ==="
rm -f data/heldout/downstream_baseline.jsonl data/heldout/downstream_gated.jsonl

$VENV/python gate1/04_downstream_eval.py --arm baseline \
  > "$LOGS/bench_baseline.log" 2>&1 || say "WARN: baseline bench had errors"
say "baseline arm done"

free_gpu   # judge pass needs the GPU before the generator reloads
$VENV/python gate1/04_downstream_eval.py --arm gated --adapter "$ADAPTER" \
  > "$LOGS/bench_gated.log" 2>&1 || say "WARN: gated bench had errors"
say "gated arm done"

say "=== RERUN STAGE 5: final report ==="
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

print("\n## SWE-bench downstream (real test execution, repair-retry harness)\n")
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
say "RERUN DONE — read $ROOT/FINAL_REPORT.md"
echo BENCH_RERUN_COMPLETE
