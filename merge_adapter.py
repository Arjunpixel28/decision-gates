import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ap = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model and save the result.")
ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct")
ap.add_argument("--adapter", default="runs/gate1-dpo/final")
ap.add_argument("--out", default="runs/gate1-dpo/merged")
args = ap.parse_args()

BASE = args.base
ADAPTER = args.adapter
OUT = args.out

print("loading base model...")
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="cpu")
tok = AutoTokenizer.from_pretrained(BASE)

print("loading adapter...")
model = PeftModel.from_pretrained(model, ADAPTER)

print("merging...")
model = model.merge_and_unload()

print(f"saving to {OUT}...")
model.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
print("done")
