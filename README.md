# Decision Before Code: A Three-Gate Architecture for One-Shot Quality Code Generation

**Arjun Mani** · Independent Researcher · 2026

📄 Paper: [arXiv link coming soon]

## Key Result

| Arm | pass@1 | Tasks passed |
|---|---|---|
| Baseline (no gate) | 0.419 | 65/155 |
| **Gated, Judge v1 (Gate 1) ★** | **0.639** | **99/155** |
| Gated, Judge v2 | 0.548 | 85/155 |
| Gate 2 only | 0.323 | 50/155 |
| Gate 1 + Gate 2 | 0.503 | 78/155 |

★ McNemar exact test p<0.000001, 95% CI [+0.142, +0.297], seed-stable across 3 runs

## What this is

A Decision-First Architecture for coding agents with three explicit decision gates:

- **Gate 1 (Specification Judge):** DPO-trained judge that detects underspecified requirements and asks clarifying questions before any code is written. +22pp improvement on identical generator.
- **Gate 2 (Plan Judge):** Selects among K=5 candidate implementation plans. Regressed on HumanEval due to plan homogeneity — documented negative result.
- **Gate 3 (Hack-Resistant Verifier):** Process reward model checking code against original spec, not proxy tests. Corrected retraining running.

## Key findings

1. Decision quality is a scaling lever independent of base model capacity
2. DPO negative-mining for clarification agents must be calibrated carefully — harder negatives caused ask-rate collapse from 91.9% → 18.9%
3. Plan selection (Gate 2) requires benchmark-rubric alignment — HumanEval has low plan-choice variance

## Models and hardware

- Generator: Qwen2.5-14B-Instruct (4-bit, frozen, served via Ollama)
- Judge: Qwen2.5-Coder-7B-Instruct + QLoRA adapters (rank 16, alpha 32)
- Hardware: Single NVIDIA Quadro P6000 (24GB VRAM)
- Training: DPO via TRL library, fp16

## Citation

Coming soon — arXiv preprint 2026
