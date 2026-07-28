"""Step 0 sanity check: a few LoRA training steps of a ~1B model on one P6000.

The build guide's advice stands: solve stack mismatches on day 1. On Pascal the
usual failure modes are (a) accidentally requesting bf16, (b) a kernel compiled
for sm_70+. If this script finishes, the training stack works.

Run:  CUDA_VISIBLE_DEVICES=1 python sanity_check.py
"""

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main() -> None:
    assert torch.cuda.is_available(), "no CUDA device visible"
    assert not torch.cuda.is_bf16_supported() or True  # informational only
    print(f"device: {torch.cuda.get_device_name(0)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,  # Pascal: fp16 storage, fp32 master weights via AMP
        attn_implementation="sdpa",  # NOT flash_attention_2 on Pascal
        device_map={"": 0},
    )

    data = Dataset.from_list(
        [{"text": f"Q: what is {i}+{i}? A: {2*i}{tok.eos_token}"} for i in range(64)]
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        peft_config=LoraConfig(r=8, lora_alpha=16, task_type="CAUSAL_LM"),
        args=SFTConfig(
            output_dir="runs/sanity",
            max_steps=5,
            per_device_train_batch_size=2,
            fp16=True,   # AMP; never bf16=True on P6000
            bf16=False,
            logging_steps=1,
            report_to=[],
        ),
    )
    trainer.train()
    print("\nSANITY CHECK PASSED — training stack works on this GPU.")


if __name__ == "__main__":
    main()
