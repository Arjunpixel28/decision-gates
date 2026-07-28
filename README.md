# Decision Before Code: A Three-Gate Architecture for One-Shot Quality Code Generation

**Arjun Mani** · Independent Researcher · 2026

[![Paper](https://img.shields.io/badge/paper-Zenodo-blue)](https://zenodo.org/records/21630006)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

📄 Paper: [https://zenodo.org/records/21630006](https://zenodo.org/records/21630006)

---

## About

Coding agents today generate code the moment a request arrives, even when
the request is underspecified, the plan is unvetted, or the resulting code
only *looks* correct. This repo is the reference implementation for a
**decision-first architecture**: three independent, DPO/SFT-trained judge
gates that sit *before* and *around* a frozen code generator, each targeting
one specific failure mode —

- asking before assuming (Gate 1),
- picking a good plan before implementing it (Gate 2), and
- catching code that games its tests instead of solving the task (Gate 3).

The core finding is that **decision quality is a lever independent of base
model capacity** — a small, cheaply-trained judge sitting in front of a
frozen 14B generator recovers a large fraction of the quality gap normally
associated with using a bigger model. All training and evaluation here runs
on a single consumer/workstation-class Pascal GPU (no datacenter hardware,
no bf16, no flash-attention) specifically to keep the recipe reproducible
outside a lab.

## Key Result

| Arm | pass@1 | Tasks passed |
|---|---|---|
| Baseline (no gate) | 0.419 | 65/155 |
| **Gated, Judge v1 (Gate 1) ★** | **0.639** | **99/155** |
| Gated, Judge v2 | 0.548 | 85/155 |
| Gate 2 only | 0.323 | 50/155 |
| Gate 1 + Gate 2 | 0.503 | 78/155 |

★ McNemar exact test p<0.000001, 95% CI [+0.142, +0.297], seed-stable across 3 runs

## The three gates

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

## Repository layout

```
gate1/            Gate 1 (Specification Judge): data prep, DPO training, benchmarks
gate2/            Gate 2 (Plan Judge): plan generation, ranking, benchmarks
gate3/            Gate 3 (Hack-Resistant Verifier), original pipeline
gate3_scripts/    Gate 3, corrected pipeline (real hack mining, SFT + DPO training)
scripts/          Orchestration: sync-to-GPU, resume/track pipeline runs
merge_adapter.py  Merge a trained LoRA adapter into its base model
sanity_check.py   Quick environment/GPU sanity check before a full run
setup_env.sh      Environment bootstrap for a fresh P6000 box
requirements.txt  Verified-compatible pinned package set
```

## Getting started

```bash
git clone git@github.com:Arjunpixel28/decision-gates.git
cd decision-gates
bash setup_env.sh
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python sanity_check.py
```

## Citation

Arjun Mani, "Decision Before Code: A Three-Gate Architecture for One-Shot Quality Code Generation," Zenodo, 2026. https://zenodo.org/records/21630006

## License

This work is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) — see [LICENSE](LICENSE). Non-commercial use with attribution only; no derivatives/modified redistributions.
