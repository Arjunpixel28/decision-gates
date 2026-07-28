"""Gate 2, step 2: pairwise rank plans -> DPO pairs.

Automated ranking via the frozen generator acting as LLM-judge (standard
practice per RLHFlow recipes cited in the thesis blueprint), scoring each
plan on the rubric: security, fit-to-spec, maintainability/simplicity,
complexity cost. For each task, the best-vs-worst plan becomes one DPO pair
(clean margin > noisy adjacent pairs).

Honesty safeguard: writes a random 15% sample to hand_validate_sample.jsonl
for Arjun to manually check agreement (report inter-rater agreement in the
thesis rather than treating the LLM-judge as ground truth silently).
"""

import argparse
import json
import random

from common2 import GATE2_DIR, chat_json, read_jsonl, write_jsonl, write_meta

RANK_PROMPT = """Task specification:
{stub}

Rank these {n} candidate implementation plans best-to-worst on: security,
fit-to-spec, maintainability/simplicity, complexity cost (lower is better
unless correctness needs it). A plan that ignores obvious edge cases ranks
low regardless of simplicity.

{plans_block}

Return ONLY JSON: {{"ranking": [<plan indices best to worst>], "why_best": "...", "why_worst": "..."}}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hand-validate-frac", type=float, default=0.15)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = read_jsonl(GATE2_DIR / "plans.jsonl")
    out_path = GATE2_DIR / "dpo_pairs_raw.jsonl"
    done = {r["task_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()

    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    n_done = 0
    f = out_path.open("a")
    sample_rows = []

    def rank(t: dict) -> None:
        nonlocal n_done
        if t["task_id"] in done:
            return
        plans = t["plans"]
        block = "\n\n".join(f"PLAN {i}: {p['plan']} (complexity: {p.get('complexity','?')}, "
                             f"edge cases: {p.get('edge_cases_handled',[])}, risks: {p.get('risks','')})"
                             for i, p in enumerate(plans))
        try:
            r = chat_json([{"role": "user", "content": RANK_PROMPT.format(
                stub=t["gold_prompt"], n=len(plans), plans_block=block)}], temperature=0.0)
            order = r["ranking"]
            assert set(order) == set(range(len(plans)))
        except Exception as e:
            print(f"skip rank {t['task_id']}: {e}", flush=True)
            return
        best, worst = plans[order[0]], plans[order[-1]]
        pair = {
            "task_id": t["task_id"],
            "prompt": t["gold_prompt"],
            "chosen": json.dumps({"plan": best["plan"], "rationale": r.get("why_best", "")}),
            "rejected": json.dumps({"plan": worst["plan"], "rationale": r.get("why_worst", "")}),
            "full_ranking": order,
        }
        with lock:
            f.write(json.dumps(pair) + "\n")
            f.flush()
            n_done += 1
            if rng.random() < args.hand_validate_frac:
                sample_rows.append({**pair, "all_plans": plans})
            if n_done % 20 == 0:
                print(f"{n_done}/{len(rows)} ranked", flush=True)

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(rank, rows))
    f.close()

    write_jsonl(GATE2_DIR / "hand_validate_sample.jsonl", sample_rows)
    write_meta(GATE2_DIR / "meta_rank.json", vars(args))

    pairs = read_jsonl(out_path)
    rng.shuffle(pairs)
    n_eval = max(1, len(pairs) // 20)
    write_jsonl(GATE2_DIR / "dpo_pairs_train.jsonl", pairs[n_eval:])
    write_jsonl(GATE2_DIR / "dpo_pairs_eval.jsonl", pairs[:n_eval])
    print(f"done: {len(pairs)} pairs, {len(sample_rows)} flagged for hand validation")


if __name__ == "__main__":
    main()
