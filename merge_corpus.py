"""Merge generated corpora into one training file: shortest accepted notes per prompt, eval prompts
removed, notes that copy the reply (draft-in-thinking) dropped.

  python3 merge_corpus.py data/spark-r0.jsonl data/spark-r0-repair.jsonl --extra data/adaptive-*.jsonl \
      --output data/spark-train-r0.jsonl
"""

import argparse
import collections
import re

from corpus import read_rows, write_rows

HELDOUT = ["data/spark-code-chat-eval.jsonl", "data/spark-prose-eval.jsonl"]


def grams(text, n=4):
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def copies_reply(row):
    notes = row["assistant"].get("reasoning_content") or ""
    reply = row["assistant"].get("content") or ""
    g = grams(notes)
    return len(g) >= 12 and len(g & grams(reply)) / len(g) > 0.5


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--extra", nargs="*", default=[], help="Reviewed corpora added as-is (approved rows)")
    p.add_argument("--output", required=True)
    p.add_argument("--priority", action="store_true",
                   help="Earlier inputs win for a prompt (instead of the shortest notes)")
    p.add_argument("--min-code-notes", type=int, default=30, help="Drop code rows with shorter notes (chars)")
    args = p.parse_args()
    held = {r["id"] for path in HELDOUT for r in read_rows(path)}
    best, dropped = {}, collections.Counter()
    for path in args.inputs:
        for row in read_rows(path):
            if row["id"] in held:
                dropped["held-out"] += 1
                continue
            if copies_reply(row):
                dropped["copies reply"] += 1
                continue
            notes = (row["assistant"].get("reasoning_content") or "").strip()
            if row.get("family") == "code" and (len(notes) < args.min_code_notes or notes.lower() == "python"):
                dropped["code without notes"] += 1
                continue
            old = best.get(row["id"])
            if old is None or (not args.priority and row["think_tokens"] < old["think_tokens"]):
                best[row["id"]] = row
    rows = list(best.values())
    for path in args.extra:
        for row in read_rows(path):
            if row.get("approved") and row["id"] not in best and row["id"] not in held:
                rows.append(row)
    write_rows(args.output, rows)
    fam = collections.Counter(r.get("family", "?") for r in rows)
    print(f"{args.output}: {len(rows)} rows {dict(fam)}; dropped {dict(dropped)}")
    gen = [r for r in rows if "native_think_tokens" in r]
    if gen:
        print(f"generated rows thinking: {sum(r['native_think_tokens'] for r in gen)} -> "
              f"{sum(r['think_tokens'] for r in gen)} tokens; empty notes: {sum(r['think_tokens'] == 0 for r in gen)}")


if __name__ == "__main__":
    main()
