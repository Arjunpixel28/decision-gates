# Decision-Gate Coding Model — 2× Quadro P6000 Edition

Implementation of `../build-guide-decision-gates.md`, adapted from the original
3× RTX 5090 plan to the hardware we actually have right now:
**2× Quadro P6000 (24GB each, Pascal, compute capability 6.1)**.

## What changes vs. the build guide

| Guide assumed (5090s) | This repo (P6000s) | Why |
|---|---|---|
| bf16 training | **fp16 mixed precision with fp32 master weights** (or pure fp32 fallback) | Pascal has no bf16 |
| flash-attn | **SDPA (default attention)** | flash-attn needs Ampere+ |
| vLLM for the generator | **llama.cpp server (GGUF)** | vLLM requires compute capability ≥ 7.0 |
| GPU 0 serves, GPUs 1–2 train | **GPU 0 serves generator, GPU 1 trains** | 2 cards, not 3 |
| Full fine-tune option for 7–9B | **QLoRA only** | 24GB + no NVLink + Pascal speed |
| Qwen3 8B full-speed | Qwen3-8B / Qwen2.5-Coder-7B-Instruct, 4-bit base | fits in 24GB with headroom |

Expect training runs to be **3–6× slower** than the guide's estimates (Pascal has
no tensor cores). A DPO LoRA run that took 2–6h on 5090s will take roughly a
day on one P6000. That's fine — the bottleneck in this project is data quality,
not FLOPs, exactly as the guide says.

## Layout

```
setup_env.sh              # Step 0: Pascal-safe environment
sanity_check.py           # Step 0: 1-step LoRA train — run this FIRST
scripts/serve_generator.sh# llama.cpp OpenAI-compatible server on GPU 0
gate1/
  01_collect_tasks.py     # pull well-specified SWE-bench Verified tasks
  02_degrade_tasks.py     # generator strips details -> underspecified + gold
  03_build_dpo_pairs.py   # (prompt -> chosen, rejected) DPO pairs
  train_gate1_dpo.py      # QLoRA DPO on GPU 1
  eval_gate1.py           # over/under-questioning + question quality metrics
```

## Run order (on the GPU machine)

```bash
bash setup_env.sh
conda activate gates
python sanity_check.py                       # must pass before anything else
bash scripts/serve_generator.sh &            # GPU 0
python gate1/01_collect_tasks.py
python gate1/02_degrade_tasks.py
python gate1/03_build_dpo_pairs.py
CUDA_VISIBLE_DEVICES=1 python gate1/train_gate1_dpo.py
python gate1/eval_gate1.py --adapter runs/gate1-dpo/final
```

## Rules carried over from the guide (still apply)

- GPU 0 is inference-only. Never time-share it with training.
- `data/heldout/` is frozen from day 1 — nothing ever trains on it.
- Track every run with W&B (`wandb login` before training).
- Version every dataset (each script writes a `meta.json` with git hash + params).
