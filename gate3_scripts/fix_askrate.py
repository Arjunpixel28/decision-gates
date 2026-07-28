"""Step 2: fix the ask-rate measurement bug and rerun on judge v1 transcripts.

Old bug: asked = "?" in reply -- true on any "?" anywhere, including inside
generated code/comments in the SAME transcript (judge replies sometimes
sketch an implementation plan alongside questions).

Fix: a genuine clarifying question is one directed at the user, appearing
BEFORE any code block starts. We take the text up to the first ``` fence
(or the whole reply if there is none), and check for a question mark in
that prefix AND that the prefix contains interrogative/request phrasing
typical of a clarifying question (not just any '?').
"""
import json
import re
from pathlib import Path

DATA_DIR = Path.home() / "decision-gates" / "data"
HE_DIR = DATA_DIR / "humaneval"
REPORT = Path.home() / "decision-gates" / "FINAL_REPORT.md"

QUESTION_HINTS = re.compile(
    r"\b(could you|can you|would you|what|which|should|do you|does the|"
    r"is it|are there|clarify|clarification|please (specify|confirm|provide)|"
    r"what's|what is|how should|need to know|let me know)\b",
    re.IGNORECASE,
)


def genuinely_asked(reply: str) -> bool:
    pre_code = reply.split("```", 1)[0]
    if "?" not in pre_code:
        return False
    # require at least one sentence ending in '?' that also reads as a
    # question directed at the user (interrogative phrasing), not a stray
    # '?' e.g. inside a regex/docstring fragment quoted before the fence.
    for sent in re.split(r"(?<=[?.!])\s+", pre_code):
        if sent.strip().endswith("?") and QUESTION_HINTS.search(sent):
            return True
    return False


def main():
    qpath = HE_DIR / "judge_questions.jsonl"
    rows = [json.loads(l) for l in qpath.open() if l.strip()]

    n_old_asked = 0
    n_new_asked = 0
    for r in rows:
        reply = r["reply"]
        old = "?" in reply
        new = genuinely_asked(reply)
        n_old_asked += old
        n_new_asked += new

    n = len(rows)
    old_rate = n_old_asked / n
    new_rate = n_new_asked / n
    print(f"n={n} old_ask_rate(buggy '?' in reply)={old_rate:.3f} "
          f"new_ask_rate(genuine clarifying question)={new_rate:.3f}")

    report = REPORT.read_text()
    note = (
        "\n## Ask-rate bug fix (judge v1)\n\n"
        "Previous ask-rate used a naive `\"?\" in reply` check, which counts any "
        "question mark anywhere in the transcript (including inside generated "
        "code/comments) as \"asked\" -- a false-positive-prone measurement.\n\n"
        "Fixed detector: a genuine clarifying question must appear in the reply "
        "text BEFORE any code block and must read as a question directed at the "
        "user (interrogative phrasing), not just contain a stray `?`.\n\n"
        f"| metric | buggy `\"?\" in reply` | fixed (question before code, directed at user) |\n"
        f"|---|---|---|\n"
        f"| ask_rate (judge v1, n={n}) | {old_rate:.3f} | {new_rate:.3f} |\n"
    )
    report = report.rstrip() + "\n" + note
    REPORT.write_text(report)
    print("appended corrected ask-rate section to FINAL_REPORT.md")


if __name__ == "__main__":
    main()
