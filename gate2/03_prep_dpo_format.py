"""Gate 2, step 3: convert plan-ranking pairs into the chat-format DPO shape
train_gate1_dpo.py already expects (prompt/chosen/rejected as message lists) —
reuses the proven Gate 1 trainer unmodified, just pointed at gate2 data."""

from pathlib import Path

from common2 import GATE2_DIR, read_jsonl, write_jsonl

SYSTEM = (
    "You are a senior engineer choosing an implementation approach before writing "
    "code. Given a function specification, propose the single best plan, weighing "
    "security, fit-to-spec, maintainability, and complexity cost. State the plan "
    "and your rationale."
)


def convert(path_in: Path, path_out: Path) -> None:
    rows = read_jsonl(path_in)
    out = [
        {
            "task_id": r["task_id"],
            "prompt": [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": r["prompt"]}],
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}],
        }
        for r in rows
    ]
    write_jsonl(path_out, out)
    print(f"{path_in.name}: {len(out)} pairs -> {path_out}")


if __name__ == "__main__":
    # train_gate1_dpo.py --pairs-dir expects exactly these two filenames.
    train_dir = GATE2_DIR / "train_ready"
    train_dir.mkdir(exist_ok=True)
    convert(GATE2_DIR / "dpo_pairs_train.jsonl", train_dir / "dpo_pairs_train.jsonl")
    convert(GATE2_DIR / "dpo_pairs_eval.jsonl", train_dir / "dpo_pairs_eval.jsonl")
