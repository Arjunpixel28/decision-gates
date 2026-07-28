"""Gate 3 step 5: QLoRA DPO training, same recipe as Gate 1 (P6000 edition).

  - fp16 AMP, no bf16 (Pascal)
  - SDPA attention
  - 4-bit nf4 base + LoRA r=16/alpha=32
  - per_device_train_batch_size=1, per_device_eval_batch_size=1 (critical --
    default of 8 causes crash loop on this hardware)
  - report_to=[] (wandb disabled -- disk is at 100%, avoid extra log growth)
  - budget: leaves headroom for the ~6GB permanently held by the service on
    port 8092; 4-bit 7B + LoRA + batch=1 fits well under 17GB.

Run:  CUDA_VISIBLE_DEVICES=0 python gate3/train_gate3_dpo.py
"""
import argparse
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

import sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "decision-gates" / "gate1"))
from common import DATA_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--output-dir", default=str(Path.home() / "decision-gates" / "runs" / "gate3-dpo"))
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=6.0)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--pairs-dir", default=str(DATA_DIR / "gate3"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
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

    train = load_dataset("json", data_files=f"{args.pairs_dir}/dpo_pairs_train.jsonl", split="train")
    evald = load_dataset("json", data_files=f"{args.pairs_dir}/dpo_pairs_eval.jsonl", split="train")

    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(
            output_dir=args.output_dir,
            beta=args.beta,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            fp16=True,
            bf16=False,
            max_length=args.max_len,
            max_prompt_length=args.max_len // 2,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=1,
            report_to=[],
            run_name="gate3-dpo-p6000",
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
