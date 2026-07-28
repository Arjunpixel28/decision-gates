"""Gate 1, step 2c: evaluation on the FROZEN held-out split — two-phase for one GPU.

Phase "generate": load the judge (base or base+adapter, fp16), produce responses
for every held-out task (underspecified AND fully-specified prompt), save JSONL,
exit (frees VRAM).
Phase "judge": with Ollama back on the GPU, score requirement recovery of the
saved responses and compute all metrics.

Metrics:
  - ask rate on underspecified prompts (want high) / under-questioning (want ~0)
  - over-questioning rate on fully-specified prompts (want ~0)
  - requirement recovery rate: fraction of stripped requirements addressed

Run (orchestrated by scripts/run_all.sh):
  python eval_gate1.py --phase generate --tag baseline
  python eval_gate1.py --phase generate --tag gated --adapter <path>
  python eval_gate1.py --phase judge --tag baseline
  python eval_gate1.py --phase judge --tag gated
"""

import argparse
import json
import re

from common import DATA_DIR, chat_json, read_jsonl

SYSTEM = (
    "You are a senior engineer reviewing an incoming task before any code is "
    "written. If the request is missing information that would change the "
    "implementation, ask at most 5 batched clarifying questions. If the request "
    "is already complete, say so and state your implementation plan."
)

JUDGE_RECOVERY = """Gold omitted requirements:
{reqs}

A model asked these questions about the underspecified request:
{response}

For each gold requirement, would answering the model's questions recover it?
Return ONLY JSON: {{"recovered": <int>, "total": <int>}}"""


def asked_questions(text: str) -> bool:
    return len(re.findall(r"\?", text)) >= 1


def phase_generate(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="sdpa", device_map={"": 0}
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def generate(user_msg: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(ids, max_new_tokens=512, do_sample=False)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    heldout = read_jsonl(DATA_DIR / "heldout" / "gate1_degraded.jsonl")[: args.limit]
    out_path = DATA_DIR / "heldout" / f"responses_{args.tag}.jsonl"
    with out_path.open("w") as f:
        for i, t in enumerate(heldout):
            row = {
                "task_id": t["task_id"],
                "resp_underspec": generate(t["underspecified_request"]),
                "resp_fullspec": generate(t["gold_spec"][:6000]),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1) % 5 == 0:
                print(f"generated {i + 1}/{len(heldout)}", flush=True)
    print(f"responses saved to {out_path}", flush=True)


def phase_judge(args) -> None:
    heldout = {t["task_id"]: t for t in read_jsonl(DATA_DIR / "heldout" / "gate1_degraded.jsonl")}
    responses = read_jsonl(DATA_DIR / "heldout" / f"responses_{args.tag}.jsonl")[: args.limit]

    ask_under = ask_full = recovered = total = 0
    for r in responses:
        t = heldout[r["task_id"]]
        if asked_questions(r["resp_underspec"]):
            ask_under += 1
            reqs = "\n".join(f"- {x}" for x in t["removed_requirements"])
            try:
                j = chat_json(
                    [{"role": "user", "content": JUDGE_RECOVERY.format(reqs=reqs, response=r["resp_underspec"])}],
                    temperature=0.0,
                )
                recovered += min(int(j["recovered"]), len(t["removed_requirements"]))
                total += len(t["removed_requirements"])
            except Exception as e:
                print(f"judge failed on {r['task_id']}: {e}", flush=True)
                total += len(t["removed_requirements"])
        else:
            total += len(t["removed_requirements"])
        if asked_questions(r["resp_fullspec"]):
            ask_full += 1

    n = len(responses)
    results = {
        "tag": args.tag,
        "n": n,
        "ask_rate_underspecified": round(ask_under / n, 3),
        "under_questioning_rate": round(1 - ask_under / n, 3),
        "over_questioning_rate": round(ask_full / n, 3),
        "requirement_recovery_rate": round(recovered / max(total, 1), 3),
    }
    print(json.dumps(results, indent=2))
    (DATA_DIR / "heldout" / f"eval_{args.tag}.json").write_text(json.dumps(results, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["generate", "judge"], required=True)
    ap.add_argument("--tag", required=True, help="baseline | gated")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    phase_generate(args) if args.phase == "generate" else phase_judge(args)


if __name__ == "__main__":
    main()
