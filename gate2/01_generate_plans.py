"""Gate 2, step 1: generate K distinct candidate plans per task.

Reuses the same HumanEval degraded-spec pool Gate 1 was evaluated on (n=155)
so Gate 2's effect is measured on the identical task distribution, and the
combined Gate1+Gate2 pipeline can be evaluated apples-to-apples against the
Gate-1-only result already in FINAL_REPORT.md.

For each task: prompt the (frozen) generator for K=5 plans, each forced to
take a genuinely different algorithmic/architectural approach — not just
reworded — so the judge has real signal to rank on.
"""

import argparse
import json

from common2 import GATE2_DIR, DATA_DIR, chat_json, read_jsonl, write_meta

PLAN_STYLES = [
    "the most straightforward brute-force approach",
    "an approach optimized for time complexity, even if more complex to implement",
    "an approach optimized for readability/simplicity over performance",
    "an approach using a different core data structure or algorithmic technique than the obvious one",
    "an approach that is maximally defensive about edge cases (empty input, negative numbers, type errors, etc.)",
]

PLAN_PROMPT = """You are proposing ONE implementation approach for this function
(not writing full code yet — just the plan). Your approach MUST follow this
constraint: {style}

Function to implement:
{stub}

Return ONLY JSON:
{{"plan": "2-4 sentence description of the approach",
  "edge_cases_handled": ["...", ...],
  "complexity": "rough time/space complexity",
  "risks": "what could go wrong or be incomplete with this approach"}}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(DATA_DIR / "humaneval" / "degraded.jsonl"))
    ap.add_argument("--out", default=str(GATE2_DIR / "plans.jsonl"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tasks = read_jsonl(__import__("pathlib").Path(args.tasks))[: args.limit]
    out_path = __import__("pathlib").Path(args.out)
    done = {r["task_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()

    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    n_done = 0
    f = out_path.open("a")

    def build(t: dict) -> None:
        nonlocal n_done
        if t["task_id"] in done:
            return
        plans = []
        for style in PLAN_STYLES[: args.k]:
            try:
                p = chat_json(
                    [{"role": "user", "content": PLAN_PROMPT.format(style=style, stub=t["gold_prompt"])}],
                    temperature=0.5,
                )
                p["style"] = style
                plans.append(p)
            except Exception as e:
                print(f"skip plan ({style}) for {t['task_id']}: {e}", flush=True)
        if len(plans) < 2:
            return
        with lock:
            f.write(json.dumps({"task_id": t["task_id"], "gold_prompt": t["gold_prompt"],
                                 "test": t["test"], "entry_point": t["entry_point"],
                                 "plans": plans}) + "\n")
            f.flush()
            n_done += 1
            if n_done % 20 == 0:
                print(f"{n_done}/{len(tasks)} tasks planned", flush=True)

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(build, tasks))
    f.close()
    write_meta(GATE2_DIR / "meta_plans.json", vars(args))
    print(f"done: {n_done} new tasks with plans -> {out_path}")


if __name__ == "__main__":
    main()
