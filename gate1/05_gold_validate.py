"""Track A: validate the harness per-instance with the GOLD patch.

An instance counts as harness-validated iff the human-written fix resolves it
in our environment. Only those instances form the official benchmark subset.
Writes data/heldout/gold_validated.json (list of task_ids).
"""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location(
    "de", Path(__file__).resolve().parent / "04_downstream_eval.py")
de = importlib.util.module_from_spec(spec)
spec.loader.exec_module(de)

from swebench.harness.test_spec.test_spec import make_test_spec  # noqa: E402


def main() -> None:
    heldout = de.read_jsonl(de.DATA_DIR / "heldout" / "gate1_degraded.jsonl")
    instances = de.get_instance_map()
    validated, statuses = [], []
    for t in heldout:
        tid = t["task_id"]
        if tid not in instances:
            continue
        inst = instances[tid]
        try:
            ts = make_test_spec(inst)
            ws = de.checkout(inst)
            py = de.build_env(inst, ws)
            if py is None:
                statuses.append((tid, "env_failed"))
                continue
            pf = ws / "_gold.diff"
            pf.write_text(inst["patch"])
            if de.sh(f"git apply -v {pf.name}", cwd=ws).returncode != 0:
                statuses.append((tid, "gold_apply_failed"))
                continue
            res = de.run_tests(inst, ws, py, ts)
            ok = res["resolved"]
            statuses.append((tid, "GOLD_RESOLVED" if ok else "gold_failed_tests"))
            if ok:
                validated.append(tid)
        except Exception as e:
            statuses.append((tid, f"error {str(e)[:80]}"))
        print(statuses[-1], flush=True)

    out = de.DATA_DIR / "heldout" / "gold_validated.json"
    out.write_text(json.dumps(validated, indent=2))
    print("SUMMARY:", dict(Counter(s for _, s in statuses)))
    print(f"validated {len(validated)} instances -> {out}")


if __name__ == "__main__":
    main()
