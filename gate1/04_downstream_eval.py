"""Gate 1 downstream benchmark: real SWE-bench test execution, gated vs. baseline.

Per frozen held-out task:
  BASELINE arm: generator implements directly from the underspecified request.
  GATED arm:    judge (our trained 7B) asks clarifying questions ->
                oracle answers them from the gold spec ->
                generator implements with the recovered information.

Both arms use ORACLE RETRIEVAL (the files touched by the gold patch, at
base_commit) — standard SWE-bench practice for non-agentic evaluation — and
emit SEARCH/REPLACE edit blocks, which apply far more reliably than raw diffs
from mid-size models.

Scoring: an instance is RESOLVED iff every FAIL_TO_PASS test passes and no
PASS_TO_PASS test breaks, executed in a per-instance venv. Instances whose
environment can't be built are excluded from BOTH arms (fair comparison).

Run (on GPU server):
  VENV=/mnt/data/decision-gates/venv
  $VENV/bin/python gate1/04_downstream_eval.py --arm baseline
  $VENV/bin/python gate1/04_downstream_eval.py --arm gated --adapter /mnt/data/decision-gates/runs/gate1-dpo/final
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from datasets import load_dataset
from swebench.harness.constants import TestStatus
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec.test_spec import make_test_spec

from common import DATA_DIR, chat, read_jsonl

WORK = Path("/mnt/data/decision-gates/swebench")
SYSTEM_JUDGE = (
    "You are a senior engineer reviewing an incoming task before any code is "
    "written. If the request is missing information that would change the "
    "implementation, ask at most 5 batched clarifying questions. If the request "
    "is already complete, say so and state your implementation plan."
)

ORACLE_PROMPT = """You are the user who wrote this fully-detailed task:

{gold}

An engineer asked you these clarifying questions before starting:

{questions}

Answer each question concisely and accurately using ONLY the details above."""

IMPLEMENT_PROMPT = """Fix the following issue in the repository {repo}.

ISSUE:
{issue}
{qa}
RELEVANT FILES (current content, truncated):
{files}

Reply with one or more edit blocks in EXACTLY this format, nothing else:

```
FILE: path/to/file.py
<<<<<<< SEARCH
(exact lines copied from the file)
=======
(replacement lines)
>>>>>>> REPLACE
```

The SEARCH text must be copied verbatim from the file. Keep edits minimal."""


def sh(cmd: str, cwd: Path | None = None, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def get_instance_map() -> dict:
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    return {r["instance_id"]: r for r in ds}


def checkout(inst: dict) -> Path:
    repo_dir = WORK / "repos" / inst["repo"].replace("/", "__")
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        r = sh(f"git clone https://github.com/{inst['repo']} {repo_dir}")
        assert r.returncode == 0, f"clone failed: {r.stderr[-300:]}"
    ws = WORK / "ws" / inst["instance_id"]
    if ws.exists():
        sh(f"rm -rf {ws}")
    ws.parent.mkdir(parents=True, exist_ok=True)
    r = sh(f"git worktree add --force {ws} {inst['base_commit']}", cwd=repo_dir)
    assert r.returncode == 0, f"worktree failed: {r.stderr[-300:]}"
    return ws


def gold_patch_files(inst: dict, ws: Path) -> str:
    paths = re.findall(r"^\+\+\+ b/(\S+)", inst["patch"], re.M)
    chunks = []
    for p in paths[:4]:
        fp = ws / p
        if fp.exists():
            text = fp.read_text(errors="replace")[:12000]
            chunks.append(f"--- {p} ---\n{text}")
    return "\n\n".join(chunks) or "(files not found)"


def apply_edit_blocks(reply: str, ws: Path) -> tuple[int, list[Path], str]:
    """Apply SEARCH/REPLACE blocks.
    Returns (n_applied, edited_files, feedback) — feedback describes what
    failed so the generator can repair its edits on a retry."""
    pat = re.compile(
        r"FILE:\s*(\S+)\s*\n<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
        re.S,
    )
    blocks = pat.findall(reply)
    if not blocks:
        return 0, [], "your reply contained no correctly formatted edit blocks"
    n, edited, problems = 0, [], []
    for path, search, replace in blocks:
        fp = ws / path.strip()
        if not fp.exists():
            problems.append(f"file not found: {path.strip()}")
            continue
        text = fp.read_text(errors="replace")
        if search not in text:
            # fallback: retry with trailing whitespace stripped per line
            norm = "\n".join(l.rstrip() for l in search.split("\n"))
            text_norm = "\n".join(l.rstrip() for l in text.split("\n"))
            if norm in text_norm:
                idx = text_norm.index(norm)
                # map normalized offset back by counting lines
                start_line = text_norm[:idx].count("\n")
                lines = text.split("\n")
                end_line = start_line + norm.count("\n") + 1
                lines[start_line:end_line] = replace.split("\n")
                fp.write_text("\n".join(lines))
                n += 1
                edited.append(fp)
                continue
            problems.append(f"SEARCH text not found verbatim in {path.strip()}")
            continue
        fp.write_text(text.replace(search, replace, 1))
        n += 1
        edited.append(fp)
    return n, edited, "; ".join(problems)


def syntax_errors(edited: list[Path], py: Path) -> str:
    """Compile-check edited python files; returns error text ('' if clean)."""
    errs = []
    for fp in edited:
        if fp.suffix != ".py":
            continue
        r = sh(f"{py} -m py_compile {fp}", timeout=60)
        if r.returncode != 0:
            errs.append(r.stderr[-500:])
    return "\n".join(errs)


def build_env(inst: dict, ws: Path) -> Path | None:
    """Best-effort per-instance venv. Returns python path or None."""
    venv = WORK / "envs" / inst["instance_id"]
    py = venv / "bin" / "python"
    if not py.exists():
        sh(f"python3 -m venv {venv}")
        sh(f"{py} -m pip install -q --upgrade pip setuptools wheel", timeout=600)
        r = sh(f"{py} -m pip install -q -e . pytest", cwd=ws, timeout=2400)
        if r.returncode != 0:
            return None
    return py


def get_test_command(test_spec) -> str | None:
    """Extract the exact test invocation swebench uses for this instance, from
    between its Start/End Test Output markers in the official eval script.
    Repo-specific quirks (Django docstring-named tests, pytest node ids, tox,
    etc.) are all handled correctly this way instead of hand-rolled per repo."""
    m = re.search(
        r": '>>>>> Start Test Output'\n(.*?)\n: '>>>>> End Test Output'",
        test_spec.eval_script, re.S,
    )
    return m.group(1).strip() if m else None


FAILED_RUN = {"resolved": False, "f2p_frac": 0.0, "p2p_ok": False, "score": 0.0}


def run_tests(inst: dict, ws: Path, py: Path, test_spec) -> dict:
    """Apply the instance's test_patch (adds/updates the actual test being
    checked — REQUIRED, without it the test doesn't fully exist in the repo),
    run the official test command against our venv, and grade with swebench's
    own log parser (handles per-repo output formats correctly).

    Returns partial-credit grading, not just all-or-nothing:
      f2p_frac  fraction of FAIL_TO_PASS tests now passing
      p2p_ok    no PASS_TO_PASS regressions
      score     f2p_frac if p2p_ok else 0 (progress only counts w/o breakage)
      resolved  the strict SWE-bench criterion
    """
    if inst.get("test_patch"):
        patch_file = ws / "_test_patch.diff"
        patch_file.write_text(inst["test_patch"])
        r = sh(f"git apply -v {patch_file.name}", cwd=ws)
        if r.returncode != 0:
            return dict(FAILED_RUN)

    cmd = get_test_command(test_spec)
    if not cmd:
        return dict(FAILED_RUN)
    script = ws / "_run_test.sh"
    script.write_text(cmd)
    env = os.environ.copy()
    env["PATH"] = f"{py.parent}:{env.get('PATH', '')}"
    try:
        r = subprocess.run(
            ["bash", "_run_test.sh"], cwd=ws, capture_output=True, text=True,
            timeout=1200, env=env,
        )
    except subprocess.TimeoutExpired:
        return dict(FAILED_RUN)
    log = (r.stdout or "") + (r.stderr or "")

    parser = MAP_REPO_TO_PARSER.get(inst["repo"])
    if parser is None:
        return dict(FAILED_RUN)
    status_map = parser(log, test_spec)

    f2p = json.loads(inst["FAIL_TO_PASS"])
    p2p = json.loads(inst["PASS_TO_PASS"])
    passed = TestStatus.PASSED.value
    skipped = TestStatus.SKIPPED.value
    f2p_frac = (sum(1 for t in f2p if status_map.get(t) == passed) / len(f2p)) if f2p else 0.0
    p2p_ok = all(status_map.get(t) in (passed, skipped) for t in p2p)
    return {
        "resolved": bool(f2p) and f2p_frac == 1.0 and p2p_ok,
        "f2p_frac": round(f2p_frac, 3),
        "p2p_ok": p2p_ok,
        "score": round(f2p_frac if p2p_ok else 0.0, 3),
    }


def generate_judge_questions(requests: list[str], adapter: str | None, model_dir: str) -> list[str]:
    """Run our judge locally (transformers), one model load for all requests.
    Frees the GPU afterwards so the generator can use it."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    m = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, attn_implementation="sdpa", device_map={"": 0}
    )
    if adapter:
        m = PeftModel.from_pretrained(m, adapter)
    m.eval()
    replies = []
    for req in requests:
        msgs = [{"role": "system", "content": SYSTEM_JUDGE}, {"role": "user", "content": req}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(m.device)
        out = m.generate(ids, max_new_tokens=512, do_sample=False)
        replies.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
    del m
    torch.cuda.empty_cache()
    return replies


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "gated"], required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--validated-only", action="store_true",
                    help="only instances where the gold patch resolves in this harness")
    ap.add_argument("--out-suffix", default="", help="e.g. _v2 for the retrained judge")
    args = ap.parse_args()

    heldout = read_jsonl(DATA_DIR / "heldout" / "gate1_degraded.jsonl")[: args.limit]
    if args.validated_only:
        vpath = DATA_DIR / "heldout" / "gold_validated.json"
        valid = set(json.loads(vpath.read_text()))
        heldout = [t for t in heldout if t["task_id"] in valid]
        print(f"validated-only: {len(heldout)} instances", flush=True)
    instances = get_instance_map()
    out_path = DATA_DIR / "heldout" / f"downstream_{args.arm}{args.out_suffix}.jsonl"
    done = {r["task_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()

    # Judge pass first (GPU), so judge and generator never co-occupy VRAM.
    questions: dict[str, str] = {}
    if args.arm == "gated":
        qcache = DATA_DIR / "heldout" / f"judge_questions{args.out_suffix}.jsonl"
        cached = {r["task_id"]: r["reply"] for r in read_jsonl(qcache)} if qcache.exists() else {}
        questions.update({t["task_id"]: cached[t["task_id"]] for t in heldout if t["task_id"] in cached})
        missing = [t for t in heldout if t["task_id"] not in questions]
        if missing:
            replies = generate_judge_questions(
                [t["underspecified_request"] for t in missing], args.adapter, args.judge_model
            )
            with qcache.open("a") as f:
                for t, reply in zip(missing, replies):
                    questions[t["task_id"]] = reply
                    f.write(json.dumps({"task_id": t["task_id"], "reply": reply}) + "\n")

    results = []
    with out_path.open("a") as fout:
        for t in heldout:
            if t["task_id"] in done or t["task_id"] not in instances:
                continue
            inst = instances[t["task_id"]]
            row = {"task_id": t["task_id"], "arm": args.arm}
            try:
                test_spec = make_test_spec(inst)
                ws = checkout(inst)
                py = build_env(inst, ws)
                if py is None:
                    row["status"] = "env_failed"
                else:
                    qa = ""
                    if args.arm == "gated":
                        q = questions[t["task_id"]]
                        if "?" in q:
                            answers = chat(
                                [{"role": "user", "content": ORACLE_PROMPT.format(gold=t["gold_spec"][:6000], questions=q)}],
                                temperature=0.3,
                            )
                            qa = f"\nCLARIFYING Q&A WITH THE USER:\nQ: {q}\nA: {answers}\n"
                    convo = [{"role": "user", "content": IMPLEMENT_PROMPT.format(
                        repo=inst["repo"], issue=t["underspecified_request"], qa=qa,
                        files=gold_patch_files(inst, ws))}]
                    # Up to 3 attempts: unmatched SEARCH text or a syntax-breaking
                    # edit gets fed back to the generator for repair.
                    applied = False
                    for attempt in range(3):
                        reply = chat(convo, temperature=0.2, max_tokens=4096)
                        sh("git checkout -- .", cwd=ws)  # clean slate per attempt
                        n, edited, feedback = apply_edit_blocks(reply, ws)
                        if n > 0:
                            serr = syntax_errors(edited, py)
                            if not serr:
                                applied = True
                                break
                            feedback = f"your edit broke the file — it no longer compiles:\n{serr}"
                        convo += [
                            {"role": "assistant", "content": reply},
                            {"role": "user", "content":
                             f"That didn't work: {feedback}\n"
                             "Copy the SEARCH lines EXACTLY from the file content above "
                             "(including indentation) and send corrected edit blocks in "
                             "the same format. Send only edit blocks."},
                        ]
                    if not applied:
                        row["status"] = "no_edits_applied"
                        row.update(FAILED_RUN)
                    else:
                        row["attempts"] = attempt + 1
                        row.update(run_tests(inst, ws, py, test_spec))
                        row["status"] = "ran"
            except Exception as e:
                row["status"] = f"error: {str(e)[:200]}"
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            results.append(row)
            print(row, flush=True)

    ran = [r for r in results if r.get("status") in ("ran", "no_edits_applied")]
    solved = sum(1 for r in ran if r.get("resolved"))
    mean_score = sum(r.get("score", 0.0) for r in ran) / len(ran) if ran else 0.0
    print(f"\n{args.arm}: {solved}/{len(ran)} resolved, mean partial score {mean_score:.3f} "
          f"({len(results) - len(ran)} excluded for env/infra reasons)")


if __name__ == "__main__":
    main()
