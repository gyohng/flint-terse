"""Pair reviewed correct responses with checked, complete model mistakes."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from corpus import check_answer, parse_assistant, read_rows, split_thinking


def add_preferences(rows, candidates):
    by_id = defaultdict(list)
    for candidate in candidates:
        by_id[candidate["id"]].append(candidate)
    result = []
    for row in rows:
        if row.get("approved") is not True:
            raise ValueError(f"{row['id']}: review the positive response first")
        wrong = []
        for candidate in by_id[row["id"]]:
            if candidate["messages"] != row["messages"] or candidate.get("tools") != row.get("tools"):
                raise ValueError(f"{row['id']}: candidate prompt differs")
            if not candidate["complete"]:
                continue
            reasoning, answer, closed = split_thinking(candidate["text"])
            if closed and check_answer(row, answer, True) is False:
                assistant = parse_assistant(reasoning, answer, row.get("tools", []))
                wrong.append((candidate["output_tokens"], assistant, candidate["ending"]))
        row = dict(row)
        row.pop("rejected_assistant", None)
        row.pop("rejected_ending", None)
        if wrong:
            # Teach against the tempting short mistake; retain useful longer working.
            _, assistant, ending = min(wrong, key=lambda r: r[0])
            row.update(rejected_assistant=assistant, rejected_ending=ending)
        result.append(row)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", nargs="+", help="Reviewed paired corpora, in teacher-feature order")
    p.add_argument("--candidates", nargs="+", required=True, help="batch-capture --candidates-output files")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = [r for path in args.input for r in read_rows(path)]
    if len({r["id"] for r in rows}) != len(rows):
        p.error("Duplicate corpus IDs")
    candidates = [r for path in args.candidates for r in read_rows(path)]
    rows = add_preferences(rows, candidates)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = Counter(r["split"] for r in rows if "rejected_assistant" in r)
    print(f"Saved {len(rows)} pairs; correctness preferences: {dict(counts)}")


if __name__ == "__main__":
    main()
