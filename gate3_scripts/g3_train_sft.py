"""Gate 3 step 5 (revised): SFT instead of DPO for the structured verdict format.

Diagnosis (see FINAL_REPORT.md / conversation): two DPO attempts (12 steps at
lr=5e-6, then ~72 steps at lr=2e-5) both failed to make the model actually
GENERATE its trained VERDICT/EVIDENCE/SPEC_CHECK format at greedy decode --
the second run drove DPO loss to ~0 (severe overfit on the relative
chosen>rejected log-likelihood objective) yet the model still emitted free
prose even on its own training examples. DPO's objective (prefer chosen over
rejected in relative log-likelihood) is not the same objective as "generate
this exact text" -- that's a supervised (next-token / SFT) objective.

This trains directly on (prompt -> chosen) pairs from the same
dpo_pairs_train/eval.jsonl (only the chosen half is used; rejected is
discarded, since SFT doesn't need a contrastive pair) so the model learns to
produce the structured format as its target completion.

Same P6000 constraints as gate1/gate3 DPO recipe:
  - fp16, no bf16 (Pascal)
  - SDPA attention
  - 4-bit nf4 base + LoRA r=16/alpha=32
  - batch=1, grad-accum=16
"""
import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "decision-gates" / "gate1"))
from common import DATA_DIR  # noqa: E402


def load_sft_dataset(path: str, tok) -> Dataset:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    texts = []
    for r in rows:
        msgs = [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["chosen"]}]
        texts.append(tok.apply_chat_template(msgs, tokenize=False))
    return Dataset.from_dict({"text": texts})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--output-dir", default=str(Path.home() / "decision-gates" / "runs" / "gate3-sft"))
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--pairs-dir", default=str(DATA_DIR / "gate3"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        attn_implementation="sdpa",
        device_map={"": 0},
    )

    train = load_sft_dataset(f"{args.pairs_dir}/dpo_pairs_train.jsonl", tok)
    evald = load_sft_dataset(f"{args.pairs_dir}/dpo_pairs_eval.jsonl", tok)

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=args.output_dir,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            fp16=True,
            bf16=False,
            max_seq_length=args.max_len,
            dataset_text_field="text",
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=20,
            save_strategy="steps",
            save_steps=20,
            save_total_limit=1,
            report_to=[],
            run_name="gate3-sft-p6000",
            packing=False,
        ),
        train_dataset=train,
        eval_dataset=evald,
        processing_class=tok,
        peft_config=LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    last_ckpt = None
    if os.path.isdir(args.output_dir):
        ckpts = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if ckpts:
            last_ckpt = os.path.join(args.output_dir, sorted(ckpts, key=lambda d: int(d.split("-")[1]))[-1])
            print(f"resuming from {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(f"{args.output_dir}/final")
    print(f"adapter saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
