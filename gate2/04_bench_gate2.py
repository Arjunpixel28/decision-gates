"""Gate 2 downstream benchmark, on the SAME HumanEval task pool as Gate 1
(data/humaneval/degraded.jsonl, n=155) for a direct comparison:

  baseline        : generator implements directly (already measured, 0.419)
  gate2_only      : generator proposes K plans -> Gate 2 judge picks one ->
                    generator implements following the chosen plan
  combined        : Gate 1 asks -> oracle answers -> generator proposes K
                    plans (now spec-complete) -> Gate 2 picks -> implements

THREE-PHASE STRUCTURE (mirrors 06_humaneval_bench.py) — this is a hard
requirement on a single 24GB GPU: Ollama (the frozen 14B generator) and the
local judge model (transformers, fp16) cannot be loaded simultaneously.
Interleaving them per-task, as an earlier version of this script did,
crashes every single task with CUDA OOM. So:
  Phase A (Ollama loaded)      : generate K plans for every task -> cache
  Phase B (judge model loaded) : pick a plan for every task -> cache
  Phase C (Ollama loaded)      : implement following the chosen plan, test
Each phase is resumable independently via its cache file.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gate1"))
from common import DATA_DIR, chat  # noqa: E402
from importlib import util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "he", Path(__file__).resolve().parent.parent / "gate1" / "06_humaneval_bench.py")
he = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(he)

from common2 import GATE2_DIR, read_jsonl, write_jsonl  # noqa: E402

SYSTEM_PLAN_JUDGE = (
    "You are a senior engineer choosing an implementation approach before writing "
    "code. Given a function specification, propose the single best plan, weighing "
    "security, fit-to-spec, maintainability, and complexity cost. State the plan "
    "and your rationale."
)

GEN_PLANS_PROMPT = """Propose {k} genuinely different implementation approaches
for this function (vary algorithm/data structure, not wording). Return ONLY
JSON: {{"plans": ["<approach 1>", "<approach 2>", ...]}}

{stub}
"""

IMPLEMENT_WITH_PLAN_PROMPT = """Implement this Python function, following the
chosen plan below.

{stub}
{qa}
CHOSEN PLAN: {plan}

Reply with ONLY the complete implementation in a single ```python code block."""


def phase_a_generate_plans(tasks, questions, args, cache_path: Path) -> dict:
    """Ollama loaded. Returns {task_id: [plan strings]}."""
    done = {r["task_id"]: r["plans"] for r in read_jsonl(cache_path)} if cache_path.exists() else {}
    with cache_path.open("a") as f:
        for i, t in enumerate(tasks):
            if t["task_id"] in done:
                continue
            stub = t["degraded_prompt"]
            if args.arm == "combined":
                q = questions.get(t["task_id"], "")
                if "?" in q:
                    ans = chat([{"role": "user", "content": he.ORACLE_PROMPT.format(
                        gold=t["gold_prompt"], questions=q)}], temperature=0.3)
                    stub = f"{stub}\nCLARIFYING Q&A:\nQ: {q}\nA: {ans}\n"
            try:
                resp = chat([{"role": "user", "content": GEN_PLANS_PROMPT.format(
                    k=args.k, stub=stub)}], temperature=0.6)
                plans = json.loads(resp[resp.index("{"):resp.rindex("}") + 1])["plans"]
            except Exception:
                plans = [resp] if "resp" in dir() else ["(plan generation failed)"]
            done[t["task_id"]] = plans
            f.write(json.dumps({"task_id": t["task_id"], "plans": plans, "stub_used": stub}) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"phase A: {i + 1}/{len(tasks)} planned", flush=True)
    return done


def free_ollama() -> None:
    """Unload every resident Ollama model so the judge model can have the
    full 24GB. Ollama auto-reloads on the next chat() call in phase C."""
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines()[1:]:
            name = line.split()[0] if line.split() else None
            if name:
                subprocess.run(["ollama", "stop", name], timeout=15)
    except Exception as e:
        print(f"warn: could not free ollama: {e}", flush=True)
    import time
    time.sleep(5)


def phase_b_judge_pick(tasks, plans_by_task, args, cache_path: Path) -> dict:
    """Judge model loaded ONCE. Returns {task_id: chosen_plan_text}."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    done = {r["task_id"]: r["chosen_plan"] for r in read_jsonl(cache_path)} if cache_path.exists() else {}
    missing = [t for t in tasks if t["task_id"] not in done]
    if not missing:
        return done

    free_ollama()
    tok = AutoTokenizer.from_pretrained(args.judge_model)
    m = AutoModelForCausalLM.from_pretrained(
        args.judge_model, torch_dtype=torch.float16, attn_implementation="sdpa", device_map={"": 0})
    m = PeftModel.from_pretrained(m, args.g2_adapter)
    m.eval()

    with cache_path.open("a") as f:
        for i, t in enumerate(missing):
            plans = plans_by_task.get(t["task_id"], [])
            if not plans:
                continue
            plans_block = "\n\n".join(f"PLAN {j}: {p}" for j, p in enumerate(plans))
            msgs = [{"role": "system", "content": SYSTEM_PLAN_JUDGE},
                    {"role": "user", "content": f"{t['degraded_prompt']}\n\nCandidate plans:\n{plans_block}\n\n"
                     "Which plan is best? Reply with just the plan text you'd follow."}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(m.device)
            out = m.generate(ids, max_new_tokens=300, do_sample=False)
            chosen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            done[t["task_id"]] = chosen
            f.write(json.dumps({"task_id": t["task_id"], "chosen_plan": chosen}) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"phase B: {i + 1}/{len(missing)} judged", flush=True)

    del m
    torch.cuda.empty_cache()
    return done


def phase_c_implement(tasks, questions, chosen_plans, args, out_path: Path) -> None:
    """Ollama loaded. Implements + runs official tests."""
    done = {r["task_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    n_pass = n_all = 0
    with out_path.open("a") as f:
        for t in tasks:
            if t["task_id"] in done:
                continue
            row = {"task_id": t["task_id"], "arm": args.arm}
            try:
                qa = ""
                if args.arm == "combined":
                    q = questions.get(t["task_id"], "")
                    if "?" in q:
                        ans = chat([{"role": "user", "content": he.ORACLE_PROMPT.format(
                            gold=t["gold_prompt"], questions=q)}], temperature=0.3)
                        qa = f"\nCLARIFYING Q&A:\nQ: {q}\nA: {ans}\n"
                plan = chosen_plans.get(t["task_id"], "")
                reply = chat([{"role": "user", "content": IMPLEMENT_WITH_PLAN_PROMPT.format(
                    stub=t["degraded_prompt"], qa=qa, plan=plan)}], temperature=0.2)
                code = he.extract_code(reply)
                row["passed"] = he.run_candidate(code, t["test"], t["entry_point"], sys.executable)
                row["status"] = "ran"
            except Exception as e:
                row["status"] = f"error: {str(e)[:150]}"
                row["passed"] = False
            f.write(json.dumps(row) + "\n")
            f.flush()
            n_all += 1
            n_pass += bool(row.get("passed"))
            if n_all % 20 == 0:
                print(f"phase C: {n_all} done, pass@1 so far {n_pass/n_all:.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["gate2_only", "combined"], required=True)
    ap.add_argument("--g1-adapter", default=None)
    ap.add_argument("--g2-adapter", required=True)
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tasks = read_jsonl(DATA_DIR / "humaneval" / "degraded.jsonl")[: args.limit]

    # Gate 1 questions (combined arm only): reuse Gate 1's cache if present,
    # else generate now — this is itself a judge-model pass, so do it before
    # phase A touches Ollama.
    questions: dict = {}
    if args.arm == "combined":
        qcache = DATA_DIR / "humaneval" / "judge_questions.jsonl"
        if qcache.exists():
            questions = {r["task_id"]: r["reply"] for r in read_jsonl(qcache)}
        missing = [t for t in tasks if t["task_id"] not in questions]
        if missing:
            free_ollama()
            replies = he.judge_batch([t["degraded_prompt"] for t in missing], args.g1_adapter, args.judge_model)
            with qcache.open("a") as f:
                for t, rep in zip(missing, replies):
                    questions[t["task_id"]] = rep
                    f.write(json.dumps({"task_id": t["task_id"], "reply": rep}) + "\n")

    plans_cache = GATE2_DIR / f"plans_bench_{args.arm}.jsonl"
    picks_cache = GATE2_DIR / f"picks_bench_{args.arm}.jsonl"
    out_path = DATA_DIR / "humaneval" / f"results_{args.arm}.jsonl"

    print("=== Phase A: generate plans (Ollama) ===", flush=True)
    plans_by_task = phase_a_generate_plans(tasks, questions, args, plans_cache)

    print("=== Phase B: judge picks plan (local model, Ollama unloaded by caller) ===", flush=True)
    chosen_plans = phase_b_judge_pick(tasks, plans_by_task, args, picks_cache)

    print("=== Phase C: implement + test (Ollama) ===", flush=True)
    phase_c_implement(tasks, questions, chosen_plans, args, out_path)

    rows = read_jsonl(out_path)
    ok = sum(1 for r in rows if r.get("passed"))
    print(f"\n{args.arm}: pass@1 = {ok}/{len(rows)} = {ok/max(len(rows),1):.3f}")


if __name__ == "__main__":
    main()
