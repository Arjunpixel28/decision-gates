"""Gate 1, step 2a-1: collect well-specified tasks.

Pulls SWE-bench Verified and keeps issues whose problem statements are long and
concrete enough to act as gold specs — the stripped details in the next step
become the answers our clarifying questions must recover.

Also carves out a frozen held-out split IMMEDIATELY (guide rule: nothing ever
trains on data/heldout/).
"""

import argparse
import random
import re

from datasets import load_dataset

from common import DATA_DIR, write_jsonl, write_meta


def is_well_specified(problem: str) -> bool:
    """Heuristic filter: long enough, and contains concrete anchors
    (code refs, expected behavior, reproduction steps)."""
    if len(problem) < 600:
        return False
    anchors = 0
    anchors += bool(re.search(r"```|`[^`]+`", problem))                # code refs
    anchors += bool(re.search(r"(?i)expected|should|must", problem))   # acceptance criteria
    anchors += bool(re.search(r"(?i)reproduce|steps|example|traceback", problem))
    return anchors >= 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tasks", type=int, default=3000)
    ap.add_argument("--heldout-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default=None, help="override output dir (e.g. data/gate1_v2)")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    rows = [
        {
            "task_id": r["instance_id"],
            "repo": r["repo"],
            "gold_spec": r["problem_statement"],
        }
        for r in ds
        if is_well_specified(r["problem_statement"])
    ]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.max_tasks]

    n_held = int(len(rows) * args.heldout_frac)
    heldout, train = rows[:n_held], rows[n_held:]

    out_dir = (DATA_DIR.parent / args.out_dir) if args.out_dir else (DATA_DIR / "gate1")
    write_jsonl(out_dir / "tasks_train.jsonl", train)
    if n_held:
        write_jsonl(DATA_DIR / "heldout" / "gate1_tasks.jsonl", heldout)
    write_meta(out_dir / "meta_collect.json", vars(args))
    print(f"kept {len(rows)} well-specified tasks: {len(train)} train, {n_held} held-out (frozen)")


if __name__ == "__main__":
    main()
