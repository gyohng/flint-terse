"""Test early endings of native traces, then keep the shortest checked-correct one."""

import argparse
import re

from transformers import AutoTokenizer

from corpus import BASE, prompt_text, read_rows, select_trace, split_thinking, write_rows


def prefixes(reasoning, tokenizer):
    boundaries = [m.start() for m in re.finditer(r"(?<=[.!?])\s+|\n+", reasoning)]
    lengths = [(end, len(tokenizer.encode(reasoning[:end], add_special_tokens=False))) for end in boundaries]
    total = len(tokenizer.encode(reasoning, add_special_tokens=False))
    chosen = set()
    for fraction in (.1, .25, .5):
        end = next((end for end, length in lengths if length >= max(16, total * fraction)), None)
        if end is not None and end < len(reasoning) * .8:
            chosen.add(end)
    return [reasoning[:end].strip() for end in sorted(chosen)]


def score_cuts(rows):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = load(BASE)
    close = tokenizer.convert_tokens_to_ids("</think>")
    groups = {}
    for row in rows:
        groups.setdefault(row["source_id"], []).append(row)
    for i, (source, candidates) in enumerate(groups.items()):
        previous, cache = [], make_prompt_cache(model)
        for row in sorted(candidates, key=lambda r: len(r["generation_prefix"])):
            prefix = row["generation_prefix"].removesuffix("</think>")
            tokens = tokenizer.encode(prompt_text(row, tokenizer) + prefix, add_special_tokens=False)
            if tokens[:len(previous)] != previous:
                previous, cache = [], make_prompt_cache(model)
            for start in range(len(previous), len(tokens), 512):
                hidden = model.model(mx.array([tokens[start:start + 512]]), cache=cache)
                mx.eval(hidden, [c.state for c in cache])
            logits = model.lm_head(hidden[:, -1]).astype(mx.float32)[0]
            probability = mx.exp(logits[close] - mx.logsumexp(logits)).item()
            previous = tokens
            yield {**row, "stop_probability": probability}
        print(f"Scored {i + 1}/{len(groups)}: {source}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("build", "score", "select"))
    p.add_argument("input", nargs="+", help="Native checked traces")
    p.add_argument("--candidates", nargs="+", help="Generated continuations, required for select")
    p.add_argument("--output", required=True)
    p.add_argument("--min-stop-prob", type=float, default=0, help="Select only cuts already plausible under the native stopping distribution")
    args = p.parse_args()
    rows = [r for path in args.input for r in read_rows(path)
            if r.get("complete") and r.get("correct") is True and split_thinking(r["text"])[2]]
    if len({r["id"] for r in rows}) != len(rows):
        p.error("Duplicate native task IDs")
    if args.command == "score":
        write_rows(args.output, score_cuts(rows))
        return
    if args.command == "build":
        tokenizer = AutoTokenizer.from_pretrained(BASE)
        candidates = []
        for row in rows:
            for i, prefix in enumerate(prefixes(row["assistant"]["reasoning_content"], tokenizer)):
                candidates.append({**row, "id": f"{row['id']}/cut-{i}", "source_id": row["id"],
                                   "generation_prefix": f"<think>{prefix}</think>"})
        write_rows(args.output, candidates)
        print(f"Built {len(candidates)} early-ending candidates from {len(rows)} correct native traces")
        return
    if not args.candidates:
        p.error("select requires --candidates")
    candidates = {r["id"]: [r] for r in rows}
    for path in args.candidates:
        for row in read_rows(path):
            if args.min_stop_prob:
                if "stop_probability" not in row:
                    p.error("Run score on the candidates before using --min-stop-prob")
                if row["stop_probability"] < args.min_stop_prob:
                    continue
            candidates[row["source_id"]].append(row)
    output, shorter, before, after = [], 0, 0, 0
    keys = ("family", "messages", "tools", "expected", "expected_number", "expected_tool", "expected_json",
            "tests", "test_imports", "split", "assistant", "ending", "stop_probability")
    for row in rows:
        chosen = select_trace(candidates[row["id"]])
        shorter += chosen["reasoning_tokens"] < row["reasoning_tokens"]
        before += row["reasoning_tokens"]
        after += chosen["reasoning_tokens"]
        output.append({"id": row["id"], **{k: chosen[k] for k in keys if k in chosen}, "approved": True})
    write_rows(args.output, output)
    print(f"Selected {len(output)} traces; {shorter} shortened; thinking tokens {before} -> {after}")


if __name__ == "__main__":
    main()
