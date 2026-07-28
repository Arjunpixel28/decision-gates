"""Gate 3, step 3: convert hack-verdict DPO pairs into the chat-format shape
train_gate1_dpo.py already expects (prompt/chosen/rejected as message lists) —
reuses the proven Gate 1 trainer unmodified, just pointed at gate3 data."""

from pathlib import Path

from common3 import DATA_DIR, read_jsonl, write_jsonl

G3_DIR = DATA_DIR / "gate3"

SYSTEM = (
    "You are a senior engineer reviewing a code submission for reward hacking "
    "before it ships: test-gaming, hardcoded outputs, stubbed logic, or a "
    "weakened test harness. Judge the submission against the specification's "
    "intent, not just whether it passes the given test."
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
    train_dir = G3_DIR / "train_ready"
    train_dir.mkdir(exist_ok=True, parents=True)
    convert(G3_DIR / "dpo_pairs_train.jsonl", train_dir / "dpo_pairs_train.jsonl")
    convert(G3_DIR / "dpo_pairs_eval.jsonl", train_dir / "dpo_pairs_eval.jsonl")
