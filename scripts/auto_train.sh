#!/usr/bin/env bash
# Waits for the data phase to finish, frees the GPU, launches Gate 1 DPO training.
# Run on the GPU server:  nohup bash scripts/auto_train.sh > ~/decision-gates/auto_train.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/.."
VENV=/mnt/data/decision-gates/venv
export HF_HOME=/mnt/data/decision-gates/hf-cache
export WANDB_MODE=${WANDB_MODE:-offline}   # no wandb login on this box yet

echo "[auto_train] waiting for DATA_PHASE_COMPLETE in data_phase.log ..."
until grep -q DATA_PHASE_COMPLETE ~/decision-gates/data_phase.log 2>/dev/null; do
  # If the chain died without finishing, bail loudly instead of waiting forever.
  if ! pgrep -f '0[23]_' >/dev/null && ! grep -q DATA_PHASE_COMPLETE ~/decision-gates/data_phase.log 2>/dev/null; then
    echo "[auto_train] ERROR: data phase not running and not complete — aborting"
    exit 1
  fi
  sleep 60
done

echo "[auto_train] data phase complete. Pair counts:"
wc -l data/gate1/dpo_pairs_train.jsonl data/gate1/dpo_pairs_eval.jsonl

n_pairs=$(wc -l < data/gate1/dpo_pairs_train.jsonl)
if [ "$n_pairs" -lt 100 ]; then
  echo "[auto_train] ERROR: only $n_pairs training pairs — refusing to train on that"
  exit 1
fi

echo "[auto_train] unloading ollama model to free VRAM (service config untouched)"
ollama stop qwen2.5:14b-instruct-q4_K_M || true
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader

echo "[auto_train] launching DPO training"
$VENV/bin/python gate1/train_gate1_dpo.py \
  --output-dir /mnt/data/decision-gates/runs/gate1-dpo \
  && echo TRAINING_COMPLETE || echo TRAINING_FAILED
