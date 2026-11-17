"""Build the prompt pool for on-policy corpus generation, excluding every evaluation prompt.

  python3 build_pool.py --output data/spark-pool.jsonl
"""

import argparse
import glob
import json
import random
import re

from corpus import read_rows, write_rows

EVAL = ["data/release-eval.jsonl", "data/confirmation-eval.jsonl", "data/final-eval.jsonl", "data/fresh-eval.jsonl",
        "data/direct-eval.jsonl", "data/code-prose-eval.jsonl", "data/retention-eval.jsonl", "data/release-chat.jsonl",
        "data/holdout.jsonl", "data/openclaw-eval.jsonl", "data/eval.jsonl"]
SOURCES = {
    "code": ["data/code-prompts.jsonl", "data/code-prose-extra-prompts.jsonl", "data/code-prose-prompts.jsonl"],
    "prose": ["data/expanded-prompts.jsonl", "data/casual-prompts.jsonl", "data/prose-prompts.jsonl",
              "data/prose-extra-prompts.jsonl", "data/direct-prompts.jsonl", "data/python-prose-prompts.jsonl"],
    "math": ["data/expanded-prompts.jsonl"],
    "tool": ["data/openclaw-prompts.jsonl", "data/tool-prompts.jsonl"],
}
KEEP = ("id", "family", "messages", "tools", "tests", "test_imports", "expected_number", "expected_tool",
        "expected_json", "expected")


def key(row):
    text = " ".join(m.get("content") or "" for m in row["messages"] if m["role"] == "user")
    return re.sub(r"\W+", " ", text.lower()).strip()[:300]


def first_line(row):
    user = [m.get("content") or "" for m in row["messages"] if m["role"] == "user"]
    return re.sub(r"\W+", " ", user[0].split("\n")[0].lower()).strip() if user else ""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/spark-pool.jsonl")
    p.add_argument("--math", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    banned, banned_lines = set(), set()
    for path in EVAL:
        for row in read_rows(path):
            banned.add(key(row))
            if row.get("family") in ("code", "math") or "tests" in row:
                banned_lines.add(first_line(row))
    pool, seen = [], set()
    for family, paths in SOURCES.items():
        for path in paths:
            for row in read_rows(path):
                fam = row.get("family") or family
                if family == "tool":
                    fam = "tool"
                if fam != family:
                    continue
                k = key(row)
                if k in banned or k in seen or first_line(row) in banned_lines:
                    continue
                seen.add(k)
                pool.append({f: row[f] for f in KEEP if f in row} | {"family": fam})
    rng = random.Random(args.seed)
    math = [r for r in pool if r["family"] == "math"]
    rng.shuffle(math)
    pool = [r for r in pool if r["family"] != "math"] + math[:args.math]
    rng.shuffle(pool)
    write_rows(args.output, pool)
    counts = {}
    for r in pool:
        counts[r["family"]] = counts.get(r["family"], 0) + 1
    print(f"{args.output}: {len(pool)} prompts {counts}")


if __name__ == "__main__":
    main()
