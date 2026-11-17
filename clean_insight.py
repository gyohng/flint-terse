"""Clean insight-only code notes: drop code-like lines (any indented line, def/return/import, code
fences, bare 'python'), and drop rows whose notes still look like code afterwards.

  python3 clean_insight.py data/spark-insight.jsonl --output data/spark-insight-clean.jsonl
"""

import argparse
import json
import re

from transformers import AutoTokenizer

CODE = re.compile(r"^(\s{4,}|\t)|^\s*(def |class |return |import |from \S+ import |for .*:$|if .*:$|while .*:$|```)"
                  r"|^\s*(python|py)\s*$|^\s*\w+\s*=\s*\[\s*$|^\s*[\]\)}]\s*$")


# Checklist filler: lines that only say something does NOT matter.
FILLER = re.compile(r"^\W*(no |not |none\b|n/a)|not relevant|not needed|not required|no .*concerns?|not tested|"
                    r"not used|irrelevant|\(non-obvious facts only\)|^\W*non-obvious facts:?\s*$", re.I)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained("models/flint")
    kept, dropped, changed = [], 0, 0
    for line in open(args.input):
        row = json.loads(line)
        notes = row["assistant"]["reasoning_content"]
        if notes:
            clean = "\n".join(l for l in notes.splitlines() if not CODE.search(l) and not FILLER.search(l)).strip()
            clean = re.sub(r"\n{3,}", "\n\n", clean)
            if clean != notes:
                changed += 1
            if len(clean) < 20 or sum(c in "()[]=:" for c in clean) > 0.15 * len(clean):
                dropped += 1
                continue
            row["assistant"]["reasoning_content"] = clean
            row["think_tokens"] = len(tok.encode(clean, add_special_tokens=False))
        kept.append(row)
    with open(args.output, "w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{args.output}: kept {len(kept)}, dropped {dropped}, cleaned {changed}; "
          f"empty {sum(r['think_tokens'] == 0 for r in kept)}")


if __name__ == "__main__":
    main()
