"""Adaptive code corpus (round 4): easy tasks get short verified notes, medium tasks trap notes,
hard tasks moderately compressed or full base reasoning; plus math/prose/tool rows from round 2.

  python3 build_r4.py --output data/spark-train-r4.jsonl
"""

import argparse
import collections
import json
import statistics

from corpus import read_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/spark-train-r4.jsonl")
    args = p.parse_args()
    held = {r["id"] for r in read_rows("data/spark-code-chat-eval.jsonl")}
    log = read_rows("data/spark-insight.log.jsonl")
    easy = {r["id"] for r in log if any(t["notes"] == "" and t["passed"] == 4 for t in r.get("tries", []))}
    medium = {r["id"]: r for r in read_rows("data/spark-insight-clean.jsonl") if r["think_tokens"] > 0}
    hard = {r["id"]: r for r in read_rows("data/spark-hard.jsonl")}
    # Short verified notes for easy tasks: the adapters' own correct RL samples, else round-2 notes.
    short = {}
    for path in ("runs/rl-it1.jsonl", "runs/rl-it2.jsonl"):
        for r in read_rows(path):
            notes = (r["assistant"].get("reasoning_content") or "").strip()
            if r["correct"] and len(notes) >= 20 and (r["id"] not in short or r["think_tokens"] < short[r["id"]]["think_tokens"]):
                short[r["id"]] = r
    for r in read_rows("data/spark-r2-compress.jsonl"):
        if r["family"] == "code" and r["id"] not in short and len(r["assistant"]["reasoning_content"]) >= 20:
            short[r["id"]] = r
    code, src = [], collections.Counter()
    for i in sorted(easy | set(medium) | set(hard)):
        if i in held:
            continue
        if i in hard:
            row, kind = hard[i], "hard"
        elif i in medium:
            row, kind = medium[i], "medium"
        elif i in short:
            row, kind = short[i], "easy"
        else:
            continue
        row = {k: v for k, v in row.items() if k not in ("advantage", "reward", "correct", "teacher_reasoning")}
        row["approved"] = True
        code.append(row)
        src[kind] += 1
    other = [r for r in read_rows("data/spark-train-r2.jsonl") if r.get("family") in ("math", "prose", "tool")]
    with open(args.output, "w") as f:
        for r in code + other:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for kind in ("easy", "medium", "hard"):
        lens = [r["think_tokens"] for r in code if
                (kind == "hard" and r["id"] in hard) or (kind == "medium" and r["id"] in medium and r["id"] not in hard)
                or (kind == "easy" and r["id"] not in hard and r["id"] not in medium)]
        if lens:
            print(f"{kind}: {len(lens)} rows, median notes {statistics.median(lens):.0f} tokens")
    print(f"{args.output}: {len(code)} code rows {dict(src)} + {len(other)} other")


if __name__ == "__main__":
    main()
