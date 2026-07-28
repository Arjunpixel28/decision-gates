"""Track C: HumanEval downstream benchmark — the thesis-proving instrument.

Function-level tasks (a 14B solves a meaningful fraction, unlike repo-level
SWE-bench) with synthetically degraded specs, mirroring HumanEvalComm:

  degrade:  strip specifics (edge cases, formats, constraints) from each
            HumanEval docstring; the stripped list is gold. OOD for the judge
            (it trained on GitHub-issue-style text, not docstrings) — clean.
  baseline: generator implements straight from the vague spec.
  gated:    judge asks -> oracle answers from the ORIGINAL spec -> generator
            implements with the recovered info.
  score:    run the official HumanEval tests. pass@1 per arm.

Usage:
  06_humaneval_bench.py degrade
  06_humaneval_bench.py arm --arm baseline
  06_humaneval_bench.py arm --arm gated --adapter <path> [--out-suffix _v2]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset

from common import DATA_DIR, chat, chat_json, read_jsonl

HE_DIR = DATA_DIR / "humaneval"

SYSTEM_JUDGE = (
    "You are a senior engineer reviewing an incoming task before any code is "
    "written. If the request is missing information that would change the "
    "implementation, ask at most 5 batched clarifying questions. If the request "
    "is already complete, say so and state your implementation plan."
)

DEGRADE_PROMPT = """You create training data for a study on underspecified coding requests.

Below is a Python function stub with a DETAILED docstring. Rewrite the
docstring to be SHORT and VAGUE (1-2 sentences): keep the general goal but
remove specifics — edge cases, exact formats, boundary behavior, examples,
constraints. Keep the function signature IDENTICAL. List every removed
requirement as an atomic, answerable item (2-6 items).

Return ONLY JSON:
{{"degraded_prompt": "<full stub with vague docstring>", "removed_requirements": ["...", ...]}}

ORIGINAL:
{prompt}
"""

ORACLE_PROMPT = """You wrote this fully-detailed function specification:

{gold}

An engineer asked these clarifying questions before implementing:

{questions}

Answer each question concisely and accurately using ONLY the details above."""

IMPLEMENT_PROMPT = """Implement this Python function.

{stub}
{qa}{plan}
Reply with ONLY the complete implementation (imports + full function
definition) in a single ```python code block. No explanations."""

PLAN_PROMPT = """You are about to implement this function:

{stub}
{qa}
Before writing code, produce a decision record. Return ONLY JSON:
{{"assumptions": ["what you assume where the spec is still silent", ...],
  "plan": "one-paragraph chosen approach",
  "rationale": "why this approach over alternatives (edge cases, complexity)"}}"""


def extract_code(reply: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", reply, re.S)
    return (m.group(1) if m else reply).strip()


def run_candidate(code: str, test: str, entry_point: str, py: str) -> bool:
    program = f"{code}\n\n{test}\n\ncheck({entry_point})\n"
    try:
        r = subprocess.run([py, "-c", program], capture_output=True, text=True, timeout=20)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def cmd_degrade(args) -> None:
    ds = load_dataset("openai_humaneval", split="test")
    out = HE_DIR / "degraded.jsonl"
    done = {r["task_id"] for r in read_jsonl(out)} if out.exists() else set()
    HE_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("a") as f:
        for r in ds:
            if r["task_id"] in done:
                continue
            try:
                d = chat_json([{"role": "user", "content": DEGRADE_PROMPT.format(prompt=r["prompt"])}],
                              temperature=0.7)
                assert r["entry_point"] in d["degraded_prompt"]
                assert 2 <= len(d["removed_requirements"]) <= 8
            except Exception as e:
                print(f"skip {r['task_id']}: {e}", flush=True)
                continue
            f.write(json.dumps({
                "task_id": r["task_id"], "gold_prompt": r["prompt"],
                "degraded_prompt": d["degraded_prompt"],
                "removed_requirements": d["removed_requirements"],
                "test": r["test"], "entry_point": r["entry_point"],
            }) + "\n")
            f.flush()
            n += 1
            if n % 20 == 0:
                print(f"{n} degraded", flush=True)
    print(f"degrade done (+{n})", flush=True)


def judge_batch(stubs: list[str], adapter: str | None, model_dir: str) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    m = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, attn_implementation="sdpa", device_map={"": 0})
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, adapter)
    m.eval()
    outs = []
    for i, s in enumerate(stubs):
        msgs = [{"role": "system", "content": SYSTEM_JUDGE},
                {"role": "user", "content": f"Implement this function:\n\n{s}"}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(m.device)
        out = m.generate(ids, max_new_tokens=400, do_sample=False)
        outs.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
        if (i + 1) % 20 == 0:
            print(f"judge {i + 1}/{len(stubs)}", flush=True)
    del m
    torch.cuda.empty_cache()
    return outs


def cmd_arm(args) -> None:
    tasks = read_jsonl(HE_DIR / "degraded.jsonl")[: args.limit]
    out = HE_DIR / f"results_{args.arm}{args.out_suffix}.jsonl"
    done = {r["task_id"] for r in read_jsonl(out)} if out.exists() else set()

    questions: dict[str, str] = {}
    if args.arm == "gated":
        qcache = HE_DIR / f"judge_questions{args.out_suffix}.jsonl"
        cached = {r["task_id"]: r["reply"] for r in read_jsonl(qcache)} if qcache.exists() else {}
        missing = [t for t in tasks if t["task_id"] not in cached]
        if missing:
            replies = judge_batch([t["degraded_prompt"] for t in missing], args.adapter, args.judge_model)
            with qcache.open("a") as f:
                for t, rep in zip(missing, replies):
                    cached[t["task_id"]] = rep
                    f.write(json.dumps({"task_id": t["task_id"], "reply": rep}) + "\n")
        questions = cached

    n_pass = n_all = 0
    with out.open("a") as f:
        for t in tasks:
            if t["task_id"] in done:
                continue
            row = {"task_id": t["task_id"], "arm": args.arm + args.out_suffix}
            try:
                qa, plan_txt, record = "", "", None
                if args.arm == "gated":
                    q = questions[t["task_id"]]
                    row["asked"] = "?" in q
                    ans = ""
                    if row["asked"]:
                        ans = chat([{"role": "user", "content": ORACLE_PROMPT.format(
                            gold=t["gold_prompt"], questions=q)}], temperature=0.3)
                        qa = f"\nCLARIFYING Q&A WITH THE SPEC AUTHOR:\nQ: {q}\nA: {ans}\n"
                    # Decision Record (thesis §3.1): assumptions/plan/rationale
                    # are produced BEFORE the code — causal, not reconstructed.
                    try:
                        rec = chat_json([{"role": "user", "content": PLAN_PROMPT.format(
                            stub=t["degraded_prompt"], qa=qa)}], temperature=0.3)
                        plan_txt = f"\nYOUR PLAN (follow it):\n{rec.get('plan', '')}\n"
                        record = {
                            "task_id": t["task_id"], "arm": args.arm + args.out_suffix,
                            "requirements": {"given": t["degraded_prompt"],
                                             "elicited_questions": q if row["asked"] else None,
                                             "elicited_answers": ans or None},
                            "assumptions": rec.get("assumptions", []),
                            "alternatives_considered": None,  # Gate 2 not built yet
                            "chosen_plan": rec.get("plan"),
                            "rationale": rec.get("rationale"),
                        }
                    except Exception:
                        record = None  # never let record generation sink the arm
                # seed varies both the RNG and temperature slightly so repeated
                # runs are genuine independent draws, not identical greedy output
                reply = chat([{"role": "user", "content": IMPLEMENT_PROMPT.format(
                    stub=t["degraded_prompt"], qa=qa, plan=plan_txt)}],
                    temperature=0.2 + (0.15 if args.seed else 0.0),
                    seed=args.seed if args.seed else None)
                code = extract_code(reply)
                row["passed"] = run_candidate(code, t["test"], t["entry_point"], sys.executable)
                row["status"] = "ran"
                if record is not None:
                    record["verification_findings"] = {
                        "official_tests_passed": row["passed"]}
                    with (HE_DIR / f"decision_records{args.out_suffix}.jsonl").open("a") as rf:
                        rf.write(json.dumps(record) + "\n")
            except Exception as e:
                row["status"] = f"error: {str(e)[:150]}"
                row["passed"] = False
            f.write(json.dumps(row) + "\n")
            f.flush()
            n_all += 1
            n_pass += bool(row.get("passed"))
            if n_all % 20 == 0:
                print(f"{n_all} done, pass@1 so far {n_pass/n_all:.3f}", flush=True)

    rows = read_jsonl(out)
    total = len(rows)
    passed = sum(1 for r in rows if r.get("passed"))
    print(f"\n{args.arm}{args.out_suffix}: pass@1 = {passed}/{total} = {passed/max(total,1):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("degrade")
    arm = sub.add_parser("arm")
    arm.add_argument("--arm", choices=["baseline", "gated"], required=True)
    arm.add_argument("--adapter", default=None)
    arm.add_argument("--judge-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    arm.add_argument("--out-suffix", default="")
    arm.add_argument("--limit", type=int, default=None)
    arm.add_argument("--seed", type=int, default=0,
                     help="varies generation temperature/sampling for seed-variance runs")
    args = ap.parse_args()
    cmd_degrade(args) if args.cmd == "degrade" else cmd_arm(args)


if __name__ == "__main__":
    main()
