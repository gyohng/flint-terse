"""Pair reviewed correct responses with checked, complete model mistakes."""

import argparse
import json
import re
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
            if not closed or not answer.strip():
                continue
            if "expected_number" in row:
                numbers = re.findall(r"[+-]?\d+(?:\.\d+)?(?:/[+-]?\d+)?", answer.replace(",", ""))
                if len(numbers) == 1 and check_answer(row, numbers[0], True):
                    # SFT handles formatting; don't label a correct calculation wrong for '$'.
                    continue
            if check_answer(row, answer, True) is False:
                assistant = parse_assistant(reasoning, answer, row.get("tools", []))
                ending = candidate.get('ending', '<|observation|>' if assistant.get('tool_calls') else '<|user|>')
                wrong.append((candidate["output_tokens"], assistant, ending))
        row = dict(row)
        row.pop("rejected_assistant", None)
        row.pop("rejected_ending", None)
        if wrong:
            # Teach against the tempting short mistake; retain useful longer working.
            _, assistant, ending = min(wrong, key=lambda r: r[0])
            row.update(rejected_assistant=assistant, rejected_ending=ending)
        result.append(row)
    return result


def repair_answers(rows, candidates):
    by_id = defaultdict(list)
    for candidate in candidates:
        by_id[candidate["id"]].append(candidate)
    result = []
    for row in rows:
        if (row.get("approved") is not True or not row.get("tests")
            or check_answer(row, row["assistant"]["content"], True) is not True):
            raise ValueError(f"{row['id']}: repair targets need reviewed, test-verified code")
        usable = []
        for candidate in by_id[row["id"]]:
            if candidate["messages"] != row["messages"] or candidate.get("tools") != row.get("tools"):
                raise ValueError(f"{row['id']}: candidate prompt differs")
            notes = candidate["assistant"].get("reasoning_content", "")
            if candidate["finish_reason"] != "stop" or any(tag in notes for tag in ("<think>", "</think>")):
                continue
            usable.append(notes)
        if not usable:
            continue
        repaired = {k: v for k, v in row.items() if k not in ("rejected_assistant", "rejected_ending")}
        repaired.update(assistant={**row["assistant"], "reasoning_content": min(usable, key=len)}, answer_only=True)
        result.append(repaired)
    return result


def retain_policy(rows, candidates, preserve=True):
    by_id = defaultdict(list)
    for candidate in candidates:
        by_id[candidate['id']].append(candidate)
    retained = []
    for row in rows:
        usable = []
        for candidate in by_id[row['id']]:
            if candidate['messages'] != row['messages'] or candidate.get('tools') != row.get('tools'):
                raise ValueError(f"{row['id']}: candidate prompt differs")
            if not candidate['complete']:
                continue
            notes, answer, clean = split_thinking(candidate['text'])
            if clean and answer.strip() and check_answer(row, answer, True) is True:
                length = (candidate.get('reasoning_tokens', len(notes)), candidate['output_tokens'])
                usable.append((length, parse_assistant(notes, answer, row.get('tools', []))))
        if usable:
            kept = {k: v for k, v in row.items() if k not in
                    ('teacher_reasoning', 'rejected_assistant', 'rejected_ending', 'answer_only', 'preserve_only')}
            kept.update(assistant=min(usable, key=lambda r: r[0])[1], approved=True)
            if preserve:
                kept['preserve_only'] = True
            else:
                kept['teacher_reasoning'] = kept['assistant']['reasoning_content']
            retained.append(kept)
    return retained


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", nargs="+", help="Reviewed paired corpora, in teacher-feature order")
    p.add_argument("--candidates", nargs="+", required=True, help="batch-capture --candidates-output files")
    p.add_argument("--output", required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--answer-repair", action="store_true", help="Pair sampled notes with verified code; train with --answer-only")
    mode.add_argument('--policy-retention', action='store_true', help='Retain checked correct responses on the model’s own reasoning paths')
    mode.add_argument('--shortest-correct', action='store_true', help='Supervise the shortest checked thinking, keeping its generated final answer')
    args = p.parse_args()
    rows = [r for path in args.input for r in read_rows(path)]
    if len({r["id"] for r in rows}) != len(rows):
        p.error("Duplicate corpus IDs")
    candidates = [r for path in args.candidates for r in read_rows(path)]
    build = retain_policy if args.policy_retention else repair_answers if args.answer_repair else add_preferences
    rows = retain_policy(rows, candidates, preserve=False) if args.shortest_correct else build(rows, candidates)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = Counter(r["split"] for r in rows if args.policy_retention or args.shortest_correct or args.answer_repair or "rejected_assistant" in r)
    kind = 'shortest correct responses' if args.shortest_correct else 'policy retention' if args.policy_retention else 'answer repairs' if args.answer_repair else 'correctness preferences'
    print(f"Saved {len(rows)} pairs; {kind}: {dict(counts)}")


if __name__ == "__main__":
    main()
