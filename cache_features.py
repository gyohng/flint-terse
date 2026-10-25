"""Cache frozen Flint hidden states for probability-preserving training."""

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
    p.add_argument("--start-weight", type=float, default=1, help="Weight the first eight reasoning tokens")
    p.add_argument("--answer-features", nargs="+", help="Original-reasoning features for answer preservation, in corpus order")
    p.add_argument("--original-reasoning", action="store_true", help="Use teacher_reasoning from a paired corpus")
    args = p.parse_args()
    if Path(args.output).exists():
        p.error(f"{args.output} exists; choose a new output")
    if min(args.weight, args.start_weight) <= 0:
        p.error("Weight must be positive")
    teacher, cursor = None, 0
    if args.answer_features:
        if args.original_reasoning or args.examples:
            p.error("Answer teachers require unconditioned compact student traces")
        parts = [mx.load(path, return_metadata=True) for path in args.answer_features]
        if any(meta["model"] != args.model for _, meta in parts):
            p.error("Answer teachers must use the same frozen model")
        teacher = {k: mx.concatenate([part[k] for part, _ in parts]) for k in ("hidden", "targets")}
        del parts
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
    if teacher is not None:
        arrays["answer_teacher_hidden"] = []
    for i, row in enumerate(rows):
        ids, offset = training_tokens(row, tokenizer)
        if len(ids) > args.max_tokens:
            raise ValueError(f"{row['id']}: exceeds --max-tokens")
        closing = ids.index(close, offset)
        weight = args.weight * row.get("loss_weight", 1)
        thought_mass = closing - offset + 1 + (args.start_weight - 1) * min(8, closing - offset - 1)
        if teacher is not None:
            original = {**row, "assistant": {**row["assistant"], "reasoning_content": row["teacher_reasoning"]}}
            original_ids, original_offset = training_tokens(original, tokenizer)
            original_close = original_ids.index(close, original_offset)
            count = len(original_ids) - original_offset
            if teacher["targets"][cursor:cursor + count].tolist() != original_ids[original_offset:]:
                p.error(f"{row['id']}: teacher feature order/token mismatch")
            if ids[closing + 1:] != original_ids[original_close + 1:]:
                p.error(f"{row['id']}: final reply changed")
            answer_start = cursor + original_close - original_offset + 1
        cache = make_prompt_cache(model)
        for start in range(0, offset - 1, 512):
            model.model(mx.array([ids[start:min(start + 512, offset - 1)]]), cache=cache)
            mx.eval([c.state for c in cache])
        for start in range(offset - 1, len(ids) - 1, 256):
            end = min(start + 256, len(ids) - 1)
            hidden = model.model(mx.array([ids[start:end]]), cache=cache)[0]
            positions = mx.arange(start + 1, end + 1)
            answer = positions > closing
            first = (positions > offset) & (positions <= min(offset + 8, closing - 1))
            values = {
                "hidden": hidden.astype(mx.float16), "targets": mx.array(ids[start + 1:end + 1]),
                "thought_weights": mx.where(answer, 0, weight * mx.where(first, args.start_weight, 1) / thought_mass),
                "answer_weights": mx.where(answer, weight / (len(ids) - closing - 1), 0),
                "weights": mx.full((end - start,), weight / (len(ids) - offset)),
                "valid": mx.full((end - start,), row["split"] == "valid"),
            }
            if teacher is not None:
                reference = teacher["hidden"][answer_start + mx.maximum(positions - closing - 1, 0)]
                values["answer_teacher_hidden"] = mx.where(answer[:, None], reference, hidden).astype(mx.float16)
            mx.eval(values)
            for key, value in values.items():
                arrays[key].append(value)
        if teacher is not None:
            cursor += count
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            print(f"Features {i + 1}/{len(rows)}: {row['id']}", flush=True)
    if teacher is not None and cursor != len(teacher["targets"]):
        p.error("Unused teacher features")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(args.output, {k: mx.concatenate(v) for k, v in arrays.items()},
                        metadata={"model": args.model, "examples": args.examples or "", "weight": str(args.weight)})
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
