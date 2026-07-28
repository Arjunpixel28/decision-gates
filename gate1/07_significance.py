"""Statistical significance for the HumanEval baseline-vs-gated result.

Paired design (same 155 tasks, each has a baseline outcome and a gated
outcome) -> McNemar's test, not an independent-proportions test. Also reports
a 95% CI on the paired difference via bootstrap, since McNemar's alone
doesn't give an effect-size interval.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "humaneval"


def load(tag: str) -> dict:
    rows = [json.loads(l) for l in open(DATA_DIR / f"results_{tag}.jsonl") if l.strip()]
    return {r["task_id"]: bool(r.get("passed")) for r in rows}


def mcnemar(a: dict, b: dict, name_a: str, name_b: str) -> None:
    common = sorted(set(a) & set(b))
    # b01: a fails, b passes (b wins this task); b10: a passes, b fails
    b01 = sum(1 for t in common if not a[t] and b[t])
    b10 = sum(1 for t in common if a[t] and not b[t])
    n = b01 + b10
    print(f"\n{name_a} vs {name_b}  (n={len(common)} paired tasks)")
    print(f"  {name_a} pass / {name_b} fail : {b10}")
    print(f"  {name_a} fail / {name_b} pass : {b01}")
    if n == 0:
        print("  no discordant pairs — cannot test")
        return
    # exact McNemar = binomial test of b01 vs n at p=0.5
    res = binomtest(b01, n, 0.5)
    print(f"  McNemar exact p-value: {res.pvalue:.6f}")
    print(f"  {'SIGNIFICANT at alpha=0.05' if res.pvalue < 0.05 else 'NOT significant at alpha=0.05'}")

    # bootstrap 95% CI on the paired pass-rate difference
    rng = np.random.default_rng(7)
    diffs = []
    tasks = np.array(common)
    a_arr = np.array([a[t] for t in common])
    b_arr = np.array([b[t] for t in common])
    for _ in range(10000):
        idx = rng.integers(0, len(tasks), len(tasks))
        diffs.append(b_arr[idx].mean() - a_arr[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  pass-rate diff ({name_b} - {name_a}): {b_arr.mean()-a_arr.mean():+.3f}  "
          f"95% bootstrap CI [{lo:+.3f}, {hi:+.3f}]")


def main() -> None:
    baseline = load("baseline")
    gated_v1 = load("gated")
    pairs = [("baseline", "gated_v1", baseline, gated_v1)]

    v2_path = DATA_DIR / "results_gated_v2.jsonl"
    if v2_path.exists():
        gated_v2 = load("gated_v2")
        pairs.append(("baseline", "gated_v2", baseline, gated_v2))
        pairs.append(("gated_v1", "gated_v2", gated_v1, gated_v2))

    for name_a, name_b, a, b in pairs:
        mcnemar(a, b, name_a, name_b)


if __name__ == "__main__":
    main()
