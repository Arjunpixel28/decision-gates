#!/usr/bin/env bash
# Step 0 — environment for 2x Quadro P6000 (Pascal, sm_61).
# Run on the GPU machine, not the Mac.
set -euo pipefail

if ! command -v nvidia-smi >/dev/null; then
  echo "ERROR: nvidia-smi not found. Run this on the GPU machine." >&2
  exit 1
fi
nvidia-smi

# Pascal notes:
#  - CUDA 12.x still supports sm_61; use cu121 wheels (safest Pascal coverage).
#  - Do NOT install flash-attn (Ampere+ only).
#  - Do NOT install vllm (needs compute capability >= 7.0); we serve the
#    frozen generator with llama.cpp instead.
conda create -n gates python=3.11 -y || true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gates

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.44" "trl>=0.9" peft datasets accelerate bitsandbytes
pip install wandb requests

# Generator server: llama.cpp with CUDA (works fine on Pascal).
if [ ! -d "$HOME/llama.cpp" ]; then
  git clone https://github.com/ggerganov/llama.cpp "$HOME/llama.cpp"
  cmake -S "$HOME/llama.cpp" -B "$HOME/llama.cpp/build" -DGGML_CUDA=ON
  cmake --build "$HOME/llama.cpp/build" --config Release -j "$(nproc)"
fi

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not visible to PyTorch"
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}, {p.total_memory/1e9:.1f} GB, sm_{p.major}{p.minor}")
    assert (p.major, p.minor) >= (6, 1), "unexpected pre-Pascal device"
print("bf16 supported:", torch.cuda.is_bf16_supported(), "(expected: False on P6000)")
EOF

echo "Environment OK. Next: python sanity_check.py"
