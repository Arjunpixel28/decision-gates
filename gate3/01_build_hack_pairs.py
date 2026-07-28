"""Gate 3, step 1: synthetic hack-injection contrastive pairs.

For each HumanEval task (canonical solution + test), produce:
  - a CLEAN example: the real canonical solution (label: no_hack)
  - 1-3 HACKED examples: the same task solved via a known reward-hacking
    pattern instead of genuinely implementing the spec (label: hack, with
    hack_type)

Hack patterns implemented (deterministic, code-based — not LLM-generated,
so labels are ground truth by construction):
  - hardcode_outputs: parse the test's assert statements for (input, expected)
    pairs, emit a function that if/elif-matches literal inputs and returns
    the literal expected output, with a fallback that (plausibly) fails on
    unseen inputs.
  - stub_pass: return a type-correct dummy value (0 / "" / [] / None) with no
    real logic — passes signature/typing checks but not behavior.
  - weaken_assert: (test-side hack) emit a modified copy of the test with
    assertions loosened/removed — simulates a model editing the test instead
    of the implementation. Kept separate from the two above since it hacks
    the harness, not the solution.

Each row also includes the ORIGINAL spec (HumanEval prompt) so Gate 3's
verifier can be trained to check "does the diff match the spec's intent",
not just "does it pass the given tests" — hardcoded/stubbed solutions pass
the given tests by construction, which is exactly the case Gate 3 exists to
catch.

Resumable: already-processed task_ids are skipped on rerun.
"""

import argparse
import ast
import re
from pathlib import Path

from common3 import DATA_DIR, read_jsonl, write_jsonl

G3_DIR = DATA_DIR / "gate3"


def extract_asserts(test_src: str, entry_point: str) -> list[tuple[str, str]]:
    """Pull literal (call_expr, expected_expr) pairs out of a HumanEval test's check() body.

    HumanEval tests call the function under test as `candidate(...)`, not by
    its real name (check(candidate) takes the implementation as a parameter),
    so we match calls to `candidate` and rewrite them to entry_point-style
    args for the hardcoded stub we generate.
    """
    pairs = []
    tree = ast.parse(test_src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        # pattern: candidate(...) == expected   or   candidate(...) is expected
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], (ast.Eq, ast.Is)):
            left, right = test.left, test.comparators[0]
            call_src, exp_src = None, None
            for side_a, side_b in [(left, right), (right, left)]:
                if isinstance(side_a, ast.Call) and isinstance(side_a.func, ast.Name) and side_a.func.id == "candidate":
                    call_src = ast.unparse(side_a)
                    exp_src = ast.unparse(side_b)
                    break
            if call_src:
                pairs.append((call_src, exp_src))
    return pairs


def make_hardcode_hack(entry_point: str, prompt: str, test_src: str) -> str | None:
    """Build a function that if/elif-matches on literal call args seen in the test."""
    pairs = extract_asserts(test_src, entry_point)
    if len(pairs) < 2:
        return None  # not enough literal cases to fake convincingly

    # Extract the arg-list portion of each call, e.g. "candidate(1, 2)" -> "1, 2"
    branches = []
    for call_src, exp_src in pairs:
        m = re.match(r"candidate\((.*)\)$", call_src)
        if not m:
            continue
        args_src = m.group(1)
        branches.append((args_src, exp_src))
    if len(branches) < 2:
        return None

    sig_match = re.search(rf"def {re.escape(entry_point)}\(([^)]*)\)[^:]*:", prompt)
    params = sig_match.group(1) if sig_match else "*args, **kwargs"
    param_names = [p.split(":")[0].strip() for p in params.split(",") if p.strip()]

    lines = [f"def {entry_point}({params}):"]
    for i, (args_src, exp_src) in enumerate(branches):
        cond = " and ".join(
            f"{name} == ({arg.strip()})"
            for name, arg in zip(param_names, _split_top_level(args_src))
        )
        kw = "if" if i == 0 else "elif"
        lines.append(f"    {kw} {cond}:")
        lines.append(f"        return {exp_src}")
    lines.append("    raise ValueError('unhandled input')  # falls through on anything not seen in tests")
    return "\n".join(lines)


def _split_top_level(args_src: str) -> list[str]:
    """Split 'a, [1,2], 3' into ['a', '[1,2]', '3'] respecting brackets."""
    parts, depth, cur = [], 0, ""
    for ch in args_src:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def make_stub_hack(entry_point: str, prompt: str, return_type_hint: str) -> str:
    """Type-correct dummy stub with no real logic."""
    sig_match = re.search(rf"def {re.escape(entry_point)}\(([^)]*)\)([^:]*):", prompt)
    params = sig_match.group(1) if sig_match else "*args, **kwargs"
    ret_hint = sig_match.group(2).strip() if sig_match else ""

    dummy = "None"
    if "List" in ret_hint or "list" in ret_hint:
        dummy = "[]"
    elif "str" in ret_hint:
        dummy = "''"
    elif "int" in ret_hint:
        dummy = "0"
    elif "float" in ret_hint:
        dummy = "0.0"
    elif "bool" in ret_hint:
        dummy = "False"
    elif "Tuple" in ret_hint or "tuple" in ret_hint:
        dummy = "()"

    return f"def {entry_point}({params}){ret_hint}:\n    return {dummy}  # stub: no real implementation"


def _import_header(prompt: str) -> str:
    """Everything before the `def` line in a HumanEval prompt (imports, typing aliases)."""
    m = re.search(r"^def \w", prompt, re.M)
    return prompt[: m.start()] if m else ""


def make_weakened_test(test_src: str) -> str:
    """Simulate hacking the harness instead of the solution: strip all asserts."""
    tree = ast.parse(test_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            node.body = [ast.Pass()]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def passes_given_test(code: str, test_src: str, entry_point: str) -> bool:
    """Execute code against its (possibly weakened) test — ground truth for
    'does this hack actually fool the harness', not assumed."""
    import subprocess
    import sys as _sys
    program = f"{code}\n\n{test_src}\n\ncheck({entry_point})\n"
    try:
        r = subprocess.run([_sys.executable, "-c", program], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(G3_DIR / "hack_pairs.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    if args.limit:
        ds = ds.select(range(args.limit))

    out_path = Path(args.out)
    done_ids = set()
    if out_path.exists():
        done_ids = {r["task_id"] for r in read_jsonl(out_path) if r.get("variant") == "clean"}

    rows = []
    n_clean = n_hard = n_stub = n_weak = 0
    for r in ds:
        tid, prompt, sol, test, entry = (
            r["task_id"], r["prompt"], r["canonical_solution"], r["test"], r["entry_point"]
        )
        if tid in done_ids:
            continue

        clean_code = prompt + sol
        rows.append({
            "task_id": tid, "variant": "clean", "hack_type": None,
            "spec": prompt, "code": clean_code, "test": test, "entry_point": entry,
            "label": "no_hack",
            "passes_given_test": passes_given_test(clean_code, test, entry),
        })
        n_clean += 1

        header = _import_header(prompt)

        hc = make_hardcode_hack(entry, prompt, test)
        if hc:
            hc_code = header + hc
            rows.append({
                "task_id": tid, "variant": "hardcode_outputs", "hack_type": "hardcode_outputs",
                "spec": prompt, "code": hc_code, "test": test, "entry_point": entry,
                "label": "hack",
                "passes_given_test": passes_given_test(hc_code, test, entry),
            })
            n_hard += 1

        stub_code = header + make_stub_hack(entry, prompt, "")
        rows.append({
            "task_id": tid, "variant": "stub_pass", "hack_type": "stub_pass",
            "spec": prompt, "code": stub_code, "test": test, "entry_point": entry,
            "label": "hack",
            "passes_given_test": passes_given_test(stub_code, test, entry),
        })
        n_stub += 1

        weak_test = make_weakened_test(test)
        rows.append({
            "task_id": tid, "variant": "weaken_assert", "hack_type": "weaken_assert",
            "spec": prompt, "code": clean_code, "test": weak_test, "entry_point": entry,
            "label": "hack",
            "note": "solution is genuine but the TEST harness was hacked (asserts stripped)",
            "passes_given_test": passes_given_test(clean_code, weak_test, entry),
        })
        n_weak += 1

    write_jsonl(out_path, read_jsonl(out_path) + rows if out_path.exists() else rows)
    hack_rows = [r for r in rows if r["label"] == "hack"]
    fooling = sum(1 for r in hack_rows if r["passes_given_test"])
    print(f"wrote {len(rows)} new rows -> {out_path}")
    print(f"  clean={n_clean} hardcode={n_hard} stub={n_stub} weaken_test={n_weak}")
    print(f"  hacks that actually fool their given test: {fooling}/{len(hack_rows)} "
          f"(the interesting/hard subset Gate 3 must catch without relying on test failure)")


if __name__ == "__main__":
    main()
