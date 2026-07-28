"""Gate 1, step 2a-3: build (prompt -> chosen, rejected) DPO pairs.

Per degraded task:
  chosen   = batched clarifying questions (<=5) whose answers recover the
             stripped requirements
  rejected = one of: immediately implementing, irrelevant questions, or an
             excessive unbatched interrogation

Plus ~30% fully-specified prompts where chosen = proceed WITHOUT questions —
this trains the ask-vs-assume decision, not just question-asking.
"""

import argparse
import json
import random

from common import DATA_DIR, chat, chat_json, read_jsonl, write_jsonl, write_meta

SYSTEM = (
    "You are a senior engineer reviewing an incoming task before any code is "
    "written. If the request is missing information that would change the "
    "implementation, ask at most 5 batched clarifying questions. If the request "
    "is already complete, say so and state your implementation plan."
)

GOOD_Q_PROMPT = """A user sent this underspecified request:

{request}

We know these requirements were omitted:
{reqs}

Write the ideal response: a brief note that you need a few details, then AT MOST
5 batched clarifying questions that would recover exactly the omitted
requirements (merge related ones). No filler, no questions about things not in
the list. Return only the response text."""

BAD_MODES = ["implement", "irrelevant", "excessive", "shallow", "partial"]

BAD_PROMPTS = {
    "implement": "Respond to this request by immediately proposing a concrete "
    "implementation with confident assumptions and no questions:\n\n{request}",
    "irrelevant": "Respond to this request with 4-5 clarifying questions that "
    "sound professional but are about UNIMPORTANT details (naming, code style, "
    "logging format) and miss what actually matters:\n\n{request}",
    "excessive": "Respond to this request with 12+ nitpicky clarifying "
    "questions, one at a time, covering every conceivable detail:\n\n{request}",
    # Harder negatives: plausible at a glance, deficient on close reading.
    # (v1's caricature negatives were separable instantly — reward acc hit 1.0.)
    "shallow": "Respond to this request with 3-5 clarifying questions that a "
    "junior engineer would ask: generic, template-like questions (which "
    "framework version? any deadline? where is the repo?) that sound "
    "reasonable but do NOT target the specific missing requirements:\n\n{request}",
    "partial": "Respond to this request with 1-2 good clarifying questions "
    "about one missing aspect, then confidently state assumptions for "
    "everything else and say you'll proceed on that basis:\n\n{request}",
}

PROCEED_PROMPT = """This request is fully specified:

{request}

Write the ideal response: briefly confirm the spec is complete (one sentence)
and lay out a short implementation plan. Do NOT ask any clarifying questions.
Return only the response text."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fully-specified-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--degraded", default=str(DATA_DIR / "gate1" / "degraded.jsonl"))
    ap.add_argument("--out-dir", default=str(DATA_DIR / "gate1"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from pathlib import Path

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    degraded = read_jsonl(Path(args.degraded))[: args.limit]

    # Crash-safe: pairs stream to a raw file as they're built; restarts skip
    # tasks already present instead of redoing days of generation.
    raw_path = out_dir / "dpo_pairs_raw.jsonl"
    done_ids = {r["task_id"].split("/full")[0] for r in read_jsonl(raw_path)} if raw_path.exists() else set()
    if done_ids:
        print(f"resuming: {len(done_ids)} tasks already built", flush=True)

    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    n_done = 0
    raw_f = raw_path.open("a")

    def build(item: tuple) -> None:
        nonlocal n_done
        t, bad_mode, with_full = item
        if t["task_id"] in done_ids:
            return
        new = []
        req_list = "\n".join(f"- {r}" for r in t["removed_requirements"])
        try:
            chosen = chat(
                [{"role": "user", "content": GOOD_Q_PROMPT.format(request=t["underspecified_request"], reqs=req_list)}],
                temperature=0.6,
            )
            rejected = chat(
                [{"role": "user", "content": BAD_PROMPTS[bad_mode].format(request=t["underspecified_request"])}],
                temperature=0.9,
            )
            new.append(
                {
                    "task_id": t["task_id"],
                    "kind": f"underspecified/{bad_mode}",
                    "prompt": [{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": t["underspecified_request"]}],
                    "chosen": [{"role": "assistant", "content": chosen}],
                    "rejected": [{"role": "assistant", "content": rejected}],
                }
            )
        except Exception as e:
            print(f"skip {t['task_id']}: {e}", flush=True)

        # Fully-specified examples: chosen = proceed, rejected = over-asking.
        if with_full:
            try:
                proceed = chat(
                    [{"role": "user", "content": PROCEED_PROMPT.format(request=t["gold_spec"][:6000])}],
                    temperature=0.6,
                )
                overask = chat(
                    [{"role": "user", "content": BAD_PROMPTS["excessive"].format(request=t["gold_spec"][:6000])}],
                    temperature=0.9,
                )
                new.append(
                    {
                        "task_id": t["task_id"] + "/full",
                        "kind": "fully-specified/proceed",
                        "prompt": [{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": t["gold_spec"][:6000]}],
                        "chosen": [{"role": "assistant", "content": proceed}],
                        "rejected": [{"role": "assistant", "content": overask}],
                    }
                )
            except Exception as e:
                print(f"skip full/{t['task_id']}: {e}", flush=True)

        with lock:
            for p in new:
                raw_f.write(json.dumps(p) + "\n")
            raw_f.flush()
            n_done += 1
            if n_done % 25 == 0:
                print(f"{n_done}/{len(degraded)} tasks this run", flush=True)

    # Pre-draw randomness so results don't depend on thread scheduling.
    work = [
        (t, rng.choice(BAD_MODES),
         rng.random() < args.fully_specified_frac / (1 - args.fully_specified_frac))
        for t in degraded
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(build, work))
    raw_f.close()

    # Final assembly from the raw stream (covers this run + prior runs).
    pairs = read_jsonl(raw_path)
    rng.shuffle(pairs)
    n_eval = max(1, len(pairs) // 20)
    write_jsonl(out_dir / "dpo_pairs_train.jsonl", pairs[n_eval:])
    write_jsonl(out_dir / "dpo_pairs_eval.jsonl", pairs[:n_eval])
    write_meta(out_dir / "meta_pairs.json", vars(args))

    kinds = {}
    for p in pairs:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    print(f"{len(pairs)} pairs total: {json.dumps(kinds, indent=2)}")


if __name__ == "__main__":
    main()
