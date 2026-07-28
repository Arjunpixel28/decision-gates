"""Step 1: dedupe FINAL_REPORT.md sections, keeping only the LAST occurrence
of each '## ...' block. Numbers are never modified."""
import re
from pathlib import Path

REPORT = Path.home() / "decision-gates" / "FINAL_REPORT.md"

text = REPORT.read_text()
lines = text.split("\n")

# Split into sections keyed by heading line (## ...). Preamble (before first ##)
# is kept once, at the top.
sections = []  # list of (heading, body_lines)
preamble = []
cur_heading = None
cur_body = []
for line in lines:
    if line.startswith("## "):
        if cur_heading is not None:
            sections.append((cur_heading, cur_body))
        elif cur_body:
            preamble = cur_body
        cur_heading = line
        cur_body = [line]
    else:
        cur_body.append(line)
if cur_heading is not None:
    sections.append((cur_heading, cur_body))

# Keep only the LAST occurrence of each heading text.
last_idx = {}
for i, (h, _) in enumerate(sections):
    last_idx[h] = i

kept = [sections[i] for i in sorted(set(last_idx.values()))]
# preserve original relative order of the kept (last) occurrences
kept.sort(key=lambda pair: last_idx[pair[0]])

out_lines = []
if preamble:
    out_lines.extend(preamble)
for h, body in kept:
    out_lines.extend(body)

out_text = "\n".join(out_lines)
out_text = re.sub(r"\n{3,}", "\n\n\n", out_text).rstrip() + "\n"

REPORT.write_text(out_text)
print(f"cleaned report: {len(lines)} -> {len(out_lines)} lines, {len(sections)} -> {len(kept)} sections")
