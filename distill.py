"""Demonstrate terse thinking to Flint, then distill checked traces without the demonstrations."""

import argparse

from checkpoint import DEFAULT_MODEL
from gguf_model import load_tokenizer
from corpus import read_rows, select_trace, split_thinking, training_tokens, write_rows


def with_examples(messages, examples):
    n = 0
    while n < len(messages) and messages[n]["role"] == "system":
        n += 1
    return messages[:n] + examples + messages[n:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("build", "select", "pair"))
    p.add_argument("input", nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument('--model', default=DEFAULT_MODEL, help='GGUF or tokenizer checkpoint for pairing and token limits')
    p.add_argument("--examples", default="data/style-examples.jsonl")
    p.add_argument("--baseline", nargs="+", help="Retain the original checked trace when the terse teacher fails or is longer")
    p.add_argument("--max-tokens", type=int, default=8192)
    args = p.parse_args()
    rows = [r for path in args.input for r in read_rows(path)]
    if len({r["id"] for r in rows}) != len(rows):
        p.error("Duplicate task IDs")
    if args.command == "build":
        examples = [message for row in read_rows(args.examples) for message in row["messages"]]
        output = []
        for row in rows:
            messages = row["messages"]
            output.append({**row, "source_messages": messages,
                           "messages": with_examples(messages, examples)})
        write_rows(args.output, output)
        print(f"Conditioned {len(output)} prompts with {len(examples) // 2} demonstrations")
        return
    tokenizer = load_tokenizer(args.model)
    baseline = {r["id"]: r for path in (args.baseline or []) for r in read_rows(path)}
    if args.command == "pair":
        output = []
        for row in rows:
            original = baseline.get(row["id"])
            if original is None or original.get("approved") is not True:
                continue
            if row.get("approved") is not True:
                p.error(f"{row['id']}: compact trace needs review")
            if original["messages"] != row["messages"] or original.get("tools") != row.get("tools"):
                p.error(f"{row['id']}: baseline is a different task")
            paired = {**row, "teacher_reasoning": original["assistant"]["reasoning_content"],
                      "assistant": {**original["assistant"], "reasoning_content": row["assistant"]["reasoning_content"]},
                      "ending": original["ending"]}
            training_tokens(paired, tokenizer)
            output.append(paired)
        if not output:
            p.error("No approved native/compact pairs")
        write_rows(args.output, output)
        print(f"Paired {len(output)} traces; original final replies preserved")
        return
    output, lengths, retained = [], [], 0
    keys = ("id", "family", "tools", "expected", "expected_number", "expected_tool", "expected_json",
            "tests", "test_imports", "split", "assistant", "ending", "loss_weight")
    for row in rows:
        original = baseline.get(row["id"])
        if original is not None and (original["messages"] != row["source_messages"] or original.get("tools") != row.get("tools")):
            p.error(f"{row['id']}: baseline is a different task")
        candidates = [r for r in (row, original) if r is not None and r.get("complete")
                      and r.get("correct") is True and split_thinking(r["text"])[2]]
        if not candidates:
            continue
        chosen = select_trace(candidates)
        retained += chosen is original
        trace = {**{k: chosen[k] for k in keys if k in chosen}, "messages": row["source_messages"], "approved": True}
        tokens, _ = training_tokens(trace, tokenizer)
        if len(tokens) > args.max_tokens:
            continue
        output.append(trace)
        lengths.append(chosen["reasoning_tokens"])
    write_rows(args.output, output)
    print(f"Accepted {len(output)}/{len(rows)} traces ({retained} native); mean thinking {sum(lengths) / max(1, len(lengths)):.1f} tokens")


if __name__ == "__main__":
    main()
