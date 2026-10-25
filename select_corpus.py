"""Likelihood-filter reference-checked shorthand, preserving its fixed answer."""

import argparse
import json
from pathlib import Path

from mlx_lm import load

from corpus import BASE, read_rows, trace_cache, trace_scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=BASE)
    p.add_argument("--answer-slack", type=float, default=0.2)
    p.add_argument("--reasoning-slack", type=float, default=6.0)
    args = p.parse_args()
    model, tokenizer = load(args.model)
    rows = read_rows(args.input)
    selected = 0
    with Path(args.output).open("x") as f:
        for i, row in enumerate(rows, 1):
            if row.get("approved") is not True:
                raise ValueError(f"{row['id']}: reference proof must be checked first")
            cache = trace_cache(model, tokenizer, row)
            original = {**row, "assistant": {**row["assistant"], "reasoning_content": row["original_reasoning"]}}
            baseline = trace_scores(model, tokenizer, original, cache)
            passing = []
            for note in row["candidates"]:
                candidate = {**row, "assistant": {**row["assistant"], "reasoning_content": note}}
                scores = trace_scores(model, tokenizer, candidate, cache)
                if (scores["reasoning_nll"] <= baseline["reasoning_nll"] + args.reasoning_slack
                        and scores["answer_nll"] <= baseline["answer_nll"] + args.answer_slack):
                    candidate["scores"] = {"original": baseline, "edited": scores}
                    passing.append((scores["reasoning_tokens"], candidate))
            if passing:
                _, candidate = min(passing, key=lambda x: x[0])
                candidate.pop("candidates")
                f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                f.flush()
                selected += 1
            if i % 10 == 0 or i == len(rows):
                print(f"{i}/{len(rows)} scored; {selected} retained", flush=True)


if __name__ == "__main__":
    main()
