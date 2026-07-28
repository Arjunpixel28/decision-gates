"""Gate 1 generalization check: standard benchmarks, COMPLETE specs (no
synthetic degradation). Tests the honest risk flagged in the test report —
does the judge's tendency to ask on fully-specified prompts hurt it here?

Datasets: HumanEval (standard, undegraded), MBPP (sanitized), LiveCodeBench
(a recent, contamination-resistant split). Same baseline-vs-gated-v1 design.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, chat  # noqa: E402
from importlib import util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("he", Path(__file__).resolve().parent / "06_humaneval_bench.py")
he = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(he)

GEN_DIR = DATA_DIR / "generalize"
GEN_DIR.mkdir(parents=True, exist_ok=True)

IMPLEMENT_PROMPT = """Implement this function.

{stub}
{qa}
Reply with ONLY the complete implementation in a single ```python code block."""


def run_mbpp_candidate(code: str, test: str, py: str) -> bool:
    import subprocess
    program = f"{code}\n\n{test}\n\ncheck()\n"
    try:
        r = subprocess.run([py, "-c", program], capture_output=True, text=True, timeout=20)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def load_tasks(dataset: str, limit: int | None) -> list[dict]:
    from datasets import load_dataset

    if dataset == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        tasks = [{"task_id": r["task_id"], "prompt": r["prompt"],
                  "test": r["test"], "entry_point": r["entry_point"]} for r in ds]
    elif dataset == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        tasks = []
        for r in ds:
            # MBPP asserts call the function by its real name directly, so the
            # generated code just needs to define that name — no renaming.
            asserts = "\n".join(f"    {t}" for t in r["test_list"])
            test = f"def check():\n{asserts}\n"
            tasks.append({"task_id": f"mbpp_{r['task_id']}",
                          "prompt": r["prompt"] + "\n\nTests it must pass:\n" +
                                    "\n".join(r["test_list"][:1]),
                          "test": test, "entry_point": None})
    elif dataset == "livecodebench":
        # NOTE: LiveCodeBench uses stdin/stdout test cases, not assert-based
        # checks like HumanEval/MBPP — grading it correctly needs its own
        # harness (subprocess stdin piping + expected-output diffing), not
        # this assert-based runner. Tasks are loaded for future use but
        # skipped at execution time (test=None) rather than silently scored
        # wrong. Wire a dedicated grader before reporting LCB numbers.
        ds = load_dataset("livecodebench/code_generation_lite", split="test", version_tag="release_v5")
        tasks = [{"task_id": r["question_id"], "prompt": r["question_content"],
                  "test": None, "entry_point": None} for r in ds]
    else:
        raise ValueError(dataset)
    return tasks[:limit] if limit else tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["humaneval", "mbpp", "livecodebench"], required=True)
    ap.add_argument("--arm", choices=["baseline", "gated"], required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--limit", type=int, default=150)
    args = ap.parse_args()

    tasks = load_tasks(args.dataset, args.limit)
    out = GEN_DIR / f"results_{args.dataset}_{args.arm}.jsonl"
    done = {r["task_id"] for r in he.read_jsonl(out)} if out.exists() else set()

    questions = {}
    if args.arm == "gated":
        qcache = GEN_DIR / f"judge_questions_{args.dataset}.jsonl"
        cached = {r["task_id"]: r["reply"] for r in he.read_jsonl(qcache)} if qcache.exists() else {}
        missing = [t for t in tasks if t["task_id"] not in cached]
        if missing:
            replies = he.judge_batch([t["prompt"] for t in missing], args.adapter, args.judge_model)
            with qcache.open("a") as f:
                for t, rep in zip(missing, replies):
                    cached[t["task_id"]] = rep
                    f.write(json.dumps({"task_id": t["task_id"], "reply": rep}) + "\n")
        questions = cached

    n_pass = n_all = n_asked = 0
    with out.open("a") as f:
        for t in tasks:
            if t["task_id"] in done or t["test"] is None:  # livecodebench needs its own grader; skip execution
                continue
            row = {"task_id": t["task_id"], "arm": args.arm}
            try:
                qa = ""
                if args.arm == "gated":
                    q = questions[t["task_id"]]
                    row["asked"] = "?" in q
                    n_asked += row["asked"]
                    if row["asked"]:
                        ans = chat([{"role": "user", "content": he.ORACLE_PROMPT.format(
                            gold=t["prompt"], questions=q)}], temperature=0.3)
                        qa = f"\nCLARIFYING Q&A:\nQ: {q}\nA: {ans}\n"
                reply = chat([{"role": "user", "content": IMPLEMENT_PROMPT.format(
                    stub=t["prompt"], qa=qa)}], temperature=0.2)
                code = he.extract_code(reply)
                if args.dataset == "mbpp":
                    row["passed"] = run_mbpp_candidate(code, t["test"], sys.executable)
                else:
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
                print(f"{n_all} done, pass@1 so far {n_pass/n_all:.3f}", flush=True)

    rows = he.read_jsonl(out)
    ok = sum(1 for r in rows if r.get("passed"))
    over_ask = f", over-ask rate on COMPLETE specs: {n_asked/max(n_all,1):.3f}" if args.arm == "gated" else ""
    print(f"\n{args.dataset}/{args.arm}: pass@1 = {ok}/{len(rows)} = {ok/max(len(rows),1):.3f}{over_ask}")


if __name__ == "__main__":
    main()
