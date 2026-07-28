#!/usr/bin/env bash
# Serve the frozen generator on GPU 0 via llama.cpp (OpenAI-compatible API).
# vLLM is not an option on Pascal; llama.cpp CUDA works well on P6000.
#
# The generator is deliberately never trained (guide Step 1): it stays frozen
# so the judge can't learn its blind spots.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
# Qwen2.5-Coder-14B q4_K_M (~9GB) fits comfortably in 24GB with long context,
# and being larger than the 7-8B judge keeps generator != judge.
GGUF="${GGUF:-$MODEL_DIR/qwen2.5-coder-14b-instruct-q4_k_m.gguf}"
PORT="${PORT:-8080}"

if [ ! -f "$GGUF" ]; then
  echo "Downloading generator GGUF to $GGUF ..."
  mkdir -p "$MODEL_DIR"
  pip show -q huggingface_hub || pip install huggingface_hub
  hf download Qwen/Qwen2.5-Coder-14B-Instruct-GGUF \
    qwen2.5-coder-14b-instruct-q4_k_m.gguf --local-dir "$MODEL_DIR"
fi

CUDA_VISIBLE_DEVICES=0 "$HOME/llama.cpp/build/bin/llama-server" \
  --model "$GGUF" \
  --n-gpu-layers 999 \
  --ctx-size 16384 \
  --parallel 4 \
  --host 127.0.0.1 --port "$PORT"

# Data-generation scripts talk to http://127.0.0.1:8080/v1/chat/completions
