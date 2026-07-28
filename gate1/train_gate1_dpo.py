"""Gate 1, step 2b: QLoRA DPO training — P6000 edition.

Pascal adaptations vs. the build guide:
  - fp16 AMP instead of bf16 (Pascal has no bf16)
  - SDPA attention instead of flash-attn
  - 4-bit base (bnb nf4) with float16 compute so a 7-8B model + reference-free
    DPO fits in 24GB
  - single GPU (GPU 1); GPU 0 stays reserved for the generator server

Run:  CUDA_VISIBLE_DEVICES=1 python gate1/train_gate1_dpo.py
"""

import argparse
import os

# Must be set before CUDA initializes. Fixes the fragmentation-driven OOM we
# hit mid-run on a 24GB card ("reserved but unallocated" memory growing step
# over step until an allocation fails despite headroom existing on paper).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from common import DATA_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--output-dir", default="runs/gate1-dpo")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    # Trimmed from 4096: DPO's concatenated chosen+rejected forward pass is the
    # single biggest memory cost, and this card has no headroom to spare once
    # something else (e.g. another Ollama model) touches the GPU mid-run.
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--pairs-dir", default=str(DATA_DIR / "gate1"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # Pascal: fp16 compute, not bf16
            bnb_4bit_use_double_quant=True,
        ),
        attn_implementation="sdpa",
        device_map={"": 0},
    )

    train = load_dataset("json", data_files=f"{args.pairs_dir}/dpo_pairs_train.jsonl", split="train")
    evald = load_dataset("json", data_files=f"{args.pairs_dir}/dpo_pairs_eval.jsonl", split="train")

    trainer = DPOTrainer(
        model=model,
        # No ref_model: with a PEFT adapter, TRL uses the frozen base as the
        # reference — halves memory, which matters at 24GB.
        args=DPOConfig(
            output_dir=args.output_dir,
            beta=args.beta,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,  # default 8 OOMs at eval_steps — the
                                           # step-50 crash-loop culprit
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
            save_steps=2,
            save_total_limit=2,
            report_to=["wandb"],
            run_name="gate1-dpo-p6000",
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
    # Resume from the last checkpoint if this is a retry after a crash — don't
    # waste the steps that already completed.
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
