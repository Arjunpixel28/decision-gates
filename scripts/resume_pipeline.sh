#!/usr/bin/env bash
# ============================================================================
# AUTONOMOUS DRIVER — runs until the thesis goal is reached, then stops.
# Installed in cron (@reboot + every 15 min). Handles, without human help:
#   - reboots / power loss      -> relaunch tracks (stamps skip finished work)
#   - crashed stages            -> relaunch; inner scripts resume midway
#   - ollama dead or on CPU     -> restart service (passwordless sudo)
#   - double-launch             -> flock in each track makes this a no-op
#   - goal reached              -> TRACK{S}_AB/C_COMPLETE markers end the loop
# Heartbeat: rewrites STATUS.md on every tick.
# ============================================================================
cd "$(dirname "$0")/.." || exit 1
ROOT=$PWD
mkdir -p logs data/stamps

# give the box a moment after boot for disks/network/ollama
[ "$(cut -d' ' -f1 /proc/uptime | cut -d. -f1)" -lt 120 ] && sleep 60

# ---- self-heal ollama --------------------------------------------------
# Case 1: service down entirely. Case 2: Pascal discovery bug -> CPU runner.
# Never restart while one of OUR jobs owns the GPU (training / judge passes).
gpu_busy() { pgrep -f 'train_gate1_dpo|eval_gate1.*generate|06_humaneval' >/dev/null; }
if ! curl -s -m 10 http://127.0.0.1:11434/api/tags >/dev/null; then
  echo "[watchdog $(date '+%F %T')] ollama unreachable — restarting" >> ollama_heal.log
  sudo -n systemctl restart ollama 2>>ollama_heal.log
elif ollama ps 2>/dev/null | tail -n +2 | grep -q 'CPU' && ! gpu_busy; then
  echo "[watchdog $(date '+%F %T')] ollama on CPU — restarting" >> ollama_heal.log
  sudo -n systemctl restart ollama 2>>ollama_heal.log
fi

# ---- relaunch tracks (each self-locks; stamps skip completed stages) ----
AB_DONE=$(grep -c TRACKS_AB_COMPLETE track_ab.log 2>/dev/null); AB_DONE=${AB_DONE:-0}
C_DONE=$(grep -c TRACK_C_COMPLETE track_c.log 2>/dev/null); C_DONE=${C_DONE:-0}
G2_DONE=$(grep -c TRACK_GATE2_COMPLETE track_gate2.log 2>/dev/null); G2_DONE=${G2_DONE:-0}
SEEDVAR_DONE=$(grep -c SEEDVAR_COMPLETE track_seedvar.log 2>/dev/null); SEEDVAR_DONE=${SEEDVAR_DONE:-0}
GEN_DONE=$(grep -c GENERALIZE_COMPLETE track_generalize.log 2>/dev/null); GEN_DONE=${GEN_DONE:-0}
[ "$AB_DONE" -eq 0 ] && nohup bash scripts/track_ab.sh >> track_ab.log 2>&1 &
[ "$C_DONE" -eq 0 ]  && nohup bash scripts/track_c.sh  >> track_c.log  2>&1 &
# Gate 2 waits for Gate 1's tracks so it never fights them for the GPU.
if [ "$AB_DONE" -gt 0 ] && [ "$C_DONE" -gt 0 ] && [ "$G2_DONE" -eq 0 ]; then
  nohup bash scripts/track_gate2.sh >> track_gate2.log 2>&1 &
fi
# Seed-variance: fast, reuses trained judge v1 — run immediately once Gate 1
# results exist, doesn't need to wait for Gate 2.
if [ "$AB_DONE" -gt 0 ] && [ "$C_DONE" -gt 0 ] && [ "$SEEDVAR_DONE" -eq 0 ]; then
  nohup bash scripts/track_seedvar.sh >> track_seedvar.log 2>&1 &
fi
# Generalization runs queued behind Gate 2 (GPU contention with Gate 2 training).
if [ "$G2_DONE" -gt 0 ] && [ "$GEN_DONE" -eq 0 ]; then
  nohup bash scripts/track_generalize.sh >> track_generalize.log 2>&1 &
fi
# Once seed-variance + generalization land, regenerate the Gate 1 test report.
if [ "$SEEDVAR_DONE" -gt 0 ] || [ "$GEN_DONE" -gt 0 ]; then
  /mnt/data/decision-gates/venv/bin/python gate1/08_gate1_test_report.py >> logs/test_report.log 2>&1
fi

# ---- heartbeat ----------------------------------------------------------
{
  echo "# Pipeline Status (auto-generated $(date '+%F %T'))"
  echo
  if [ "$AB_DONE" -gt 0 ] && [ "$C_DONE" -gt 0 ] && [ "$G2_DONE" -gt 0 ] \
     && [ "$SEEDVAR_DONE" -gt 0 ] && [ "$GEN_DONE" -gt 0 ]; then
    echo "## GOAL REACHED — see FINAL_REPORT.md and GATE1_TEST_REPORT.md"
  fi
  echo "## Stages complete"
  ls data/stamps/ 2>/dev/null | sed 's/^/- /'
  echo
  echo "## Latest activity"
  echo '```'
  tail -1 logs/pairs_v2.log 2>/dev/null
  tail -1 logs/train_v2.log 2>/dev/null | cut -c1-200
  grep 'pass@1' logs/he_baseline.log logs/he_gated_v1.log logs/he_gated_v2.log 2>/dev/null | tail -3
  echo '```'
  echo
  echo "## Health"
  echo '```'
  ollama ps 2>/dev/null | tail -1
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
  pgrep -af '03_buil|train_gat|06_humanev|eval_gate' 2>/dev/null | grep -v pgrep | head -3
  echo '```'
} > STATUS.md
exit 0
