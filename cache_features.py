"""Cache frozen Flint hidden states for output-head training."""

import argparse
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from corpus import BASE, read_rows, training_tokens
from distill import with_examples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=BASE)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--examples", help="Add thinking demonstrations to cache teacher features")
    p.add_argument("--weight", type=float, default=1, help="Relative loss weight per example")
    p.add_argument("--original-reasoning", action="store_true", help="Use teacher_reasoning from a paired corpus")
    args = p.parse_args()
    if Path(args.output).exists():
        p.error(f"{args.output} exists; choose a new output")
    if args.weight <= 0:
        p.error("Weight must be positive")
    rows = [row for path in args.input for row in read_rows(path)]
    if args.original_reasoning:
        rows = [{**row, "assistant": {**row["assistant"], "reasoning_content": row["teacher_reasoning"]}} for row in rows]
    if args.examples:
        examples = [message for row in read_rows(args.examples) for message in row["messages"]]
        rows = [{**row, "messages": with_examples(row["messages"], examples)} for row in rows]
    model, tokenizer = load(args.model)
    model.eval()
    close = tokenizer.convert_tokens_to_ids("</think>")
    arrays = {k: [] for k in ("hidden", "targets", "thought_weights", "answer_weights", "weights", "valid")}
    for i, row in enumerate(rows):
        ids, offset = training_tokens(row, tokenizer)
        if len(ids) > args.max_tokens:
            raise ValueError(f"{row['id']}: exceeds --max-tokens")
        closing = ids.index(close, offset)
        cache = make_prompt_cache(model)
        for start in range(0, offset - 1, 512):
            model.model(mx.array([ids[start:min(start + 512, offset - 1)]]), cache=cache)
            mx.eval([c.state for c in cache])
        for start in range(offset - 1, len(ids) - 1, 256):
            end = min(start + 256, len(ids) - 1)
            hidden = model.model(mx.array([ids[start:end]]), cache=cache)[0]
            answer = mx.arange(start + 1, end + 1) > closing
            values = {
                "hidden": hidden.astype(mx.float16), "targets": mx.array(ids[start + 1:end + 1]),
                "thought_weights": mx.where(answer, 0, args.weight / (closing - offset + 1)),
                "answer_weights": mx.where(answer, args.weight / (len(ids) - closing - 1), 0),
                "weights": mx.full((end - start,), args.weight / (len(ids) - offset)),
                "valid": mx.full((end - start,), row["split"] == "valid"),
            }
            mx.eval(values)
            for key, value in values.items():
                arrays[key].append(value)
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            print(f"Features {i + 1}/{len(rows)}: {row['id']}", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(args.output, {k: mx.concatenate(v) for k, v in arrays.items()},
                        metadata={"model": args.model, "examples": args.examples or "", "weight": str(args.weight)})
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
