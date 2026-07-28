"""Gate 1, step 2a-2: synthetic degradation.

For each well-specified task, the frozen generator produces:
  - an underspecified rewrite ("build a chat app" style)
  - the explicit list of removed requirements (this becomes gold)

Requires scripts/serve_generator.sh running on GPU 0.
Resumable: already-processed task_ids are skipped on rerun.
"""

import argparse

from common import DATA_DIR, chat_json, read_jsonl, write_meta

DEGRADE_PROMPT = """You create training data for a study on underspecified software requests.

Below is a WELL-SPECIFIED task. Rewrite it as a SHORT, VAGUE request a hurried
user might type (1-3 sentences), and list every concrete requirement you removed.

Rules:
- The vague request must still identify the same general goal.
- Removed requirements must be atomic and answerable (each one could be
  recovered by a single clarifying question). Cover categories where present:
  security, UI/UX, scale/performance, features, constraints, environment.
- 3 to 8 removed requirements.

Return ONLY JSON: {{"underspecified_request": str, "removed_requirements": [str, ...]}}

WELL-SPECIFIED TASK:
{spec}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap for smoke tests")
    ap.add_argument("--tasks", default=str(DATA_DIR / "gate1" / "tasks_train.jsonl"))
    ap.add_argument("--out", default=str(DATA_DIR / "gate1" / "degraded.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from pathlib import Path

    tasks = read_jsonl(Path(args.tasks))
    out_path = Path(args.out)
    done = {r["task_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks[: args.limit] if t["task_id"] not in done]

    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    n_ok = n_fail = 0

    def degrade(task: dict) -> None:
        nonlocal n_ok, n_fail
        try:
            result = chat_json(
                [{"role": "user", "content": DEGRADE_PROMPT.format(spec=task["gold_spec"][:6000])}],
                temperature=0.8,
            )
            reqs = result["removed_requirements"]
            assert isinstance(reqs, list) and 3 <= len(reqs) <= 8
            assert len(result["underspecified_request"]) < len(task["gold_spec"]) / 2
        except Exception as e:  # malformed generation: skip, don't crash the run
            with lock:
                n_fail += 1
                print(f"skip {task['task_id']}: {e}", flush=True)
            return
        with lock:
            f.write(json.dumps({**task, **result}) + "\n")
            f.flush()
            n_ok += 1
            if n_ok % 25 == 0:
                print(f"{n_ok}/{len(todo)} degraded ({n_fail} skipped)", flush=True)

    with out_path.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(degrade, todo))

    write_meta(DATA_DIR / "gate1" / "meta_degrade.json", vars(args))
    print(f"done: {n_ok} new, {n_fail} skipped, output {out_path}")


if __name__ == "__main__":
    main()
