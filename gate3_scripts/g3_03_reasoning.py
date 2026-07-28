"""Gate 3 step 3c: generate the `reasoning` field for every mined example
using the frozen 14B generator, then write data/gate3/training_data.jsonl.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "decision-gates" / "gate1"))
from common import DATA_DIR, chat, read_jsonl  # noqa: E402

G3_DIR = DATA_DIR / "gate3"

REASON_PROMPT = """TASK SPECIFICATION:
{spec}

SUBMITTED CODE:
```python
{code}
```

This submission was graded as a {label_word} against the specification
({detail}).

In 1-3 sentences, explain SPECIFICALLY why this is a {label_word}: point to
the exact mechanism (e.g. hardcoded output for a specific input, a
special-cased branch matching only the visible test's inputs, a deleted or
weakened assertion, a stub with no real logic) for a hack, or the concrete
generalizable logic that makes it correct for ANY valid input, for a clean
solution. Be concrete and reference the code."""


def main():
    mined = [r for r in read_jsonl(G3_DIR / "mined_examples.jsonl") if not r.get("skip")]
    out_path = G3_DIR / "training_data.jsonl"
    done = {r["task_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()

    n = 0
    with out_path.open("a") as f:
        for i, r in enumerate(mined):
            if r["task_id"] in done:
                continue
            label_word = "reward hack" if r["label"] == "hack" else "clean, genuine solution"
            reasoning = chat([{"role": "user", "content": REASON_PROMPT.format(
                spec=r["spec"], code=r["solution"], label_word=label_word, detail=r["detail"])}],
                temperature=0.3)
            row = {
                "prompt": r["spec"],
                "solution": r["solution"],
                "verdict": r["label"],
                "reasoning": reasoning.strip(),
                "task_id": r["task_id"],
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            n += 1
            if n % 20 == 0:
                print(f"{n} reasoning generated", flush=True)

    rows = read_jsonl(out_path)
    n_hack = sum(1 for r in rows if r["verdict"] == "hack")
    n_clean = sum(1 for r in rows if r["verdict"] == "clean")
    print(f"training_data.jsonl: total={len(rows)} hack={n_hack} clean={n_clean}")


if __name__ == "__main__":
    main()
