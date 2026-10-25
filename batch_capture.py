"""Collect native Flint traces with MLX continuous batching."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import apply_top_k, apply_top_p, make_sampler

from corpus import BASE, check_answer, configure_adapter_tokenizer, parse_assistant, prompt_text, read_rows, report, select_trace, split_thinking


def request_sampler(seed, temperature):
    # Early finishes must not change another task's random draws in paired evaluations.
    key = mx.random.key(seed)

    def sample(logprobs):
        nonlocal key
        if temperature == 0:
            return mx.argmax(logprobs, axis=-1)
        key, draw = mx.random.split(key)
        filtered = apply_top_k(apply_top_p(logprobs, .95), 20)
        return mx.random.categorical(filtered / temperature, key=draw)

    return sample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("--output", required=True)
    p.add_argument("--candidates-output", help="Keep every attempt for correctness preference pairs")
    p.add_argument("--model", default=BASE)
    p.add_argument("--adapter")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--independent-seeds", action="store_true", help="Keep each task's random stream independent of batch scheduling")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    if min(args.batch_size, args.samples, args.max_tokens) < 1:
        p.error("batch size, samples and max tokens must be positive")
    rows = read_rows(args.input)
    if args.limit:
        rows = rows[:args.limit]
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    configure_adapter_tokenizer(tokenizer, args.adapter)
    mx.random.seed(args.seed)
    prompts = [tokenizer.encode(prompt_text(row, tokenizer) + row.get("generation_prefix", ""), add_special_tokens=False)
               for row in rows for _ in range(args.samples)]
    gen = BatchGenerator(
        model, max_tokens=args.max_tokens, stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        sampler=make_sampler(temp=args.temperature, top_p=.95, top_k=20),
        completion_batch_size=args.batch_size, prefill_batch_size=args.batch_size,
        prefill_step_size=512,
    )
    samplers = [request_sampler(args.seed + i, args.temperature) for i in range(len(prompts))] if args.independent_seeds else None
    uids = gen.insert(prompts, samplers=samplers)
    # Let completed requests release their random-key graphs during large collections.
    del samplers
    task = {uid: i // args.samples for i, uid in enumerate(uids)}
    tokens = {uid: [] for uid in uids}
    candidates = [[] for _ in rows]
    finished, cursor, completed = {}, 0, 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = None
    if args.candidates_output:
        path = Path(args.candidates_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = path.open("x")
    try:
        with output.open("x") as f, gen.stats() as stats:
            while responses := gen.next_generated():
                for response in responses:
                    uid = response.uid
                    if response.finish_reason != "stop":
                        tokens[uid].append(response.token)
                    if response.finish_reason is None:
                        continue
                    index = task[uid]
                    row = rows[index]
                    prefix = row.get("generation_prefix", "")
                    text = prefix + tokenizer.decode(tokens[uid])
                    reasoning, answer, closed = split_thinking(text)
                    complete = closed and response.finish_reason == "stop"
                    try:
                        assistant = parse_assistant(reasoning, answer, row.get("tools", []))
                    except (KeyError, ValueError):
                        assistant = {"reasoning_content": reasoning, "content": answer}
                        complete = False
                    result = {
                        **row, "assistant": assistant, "text": text, "complete": complete,
                        "correct": check_answer(row, answer, complete), "approved": False,
                        "reasoning_tokens": len(tokenizer.encode(reasoning, add_special_tokens=False)),
                        "output_tokens": len(tokenizer.encode(prefix, add_special_tokens=False)) + len(tokens[uid]) + int(response.finish_reason == "stop"),
                        "finish_reason": response.finish_reason,
                        "model": args.model, "adapter": args.adapter, "seed": args.seed,
                        "temperature": args.temperature, "max_tokens": args.max_tokens,
                        "samples": args.samples, "sampling_batch_size": args.batch_size,
                    }
                    if response.finish_reason == "stop":
                        result["ending"] = tokenizer.decode([response.token])
                    if args.independent_seeds:
                        result["independent_seeds"] = True
                    if raw:
                        raw.write(json.dumps(result, ensure_ascii=False) + "\n")
                        raw.flush()
                    candidates[index].append(result)
                    del tokens[uid]
                    if len(candidates[index]) == args.samples:
                        selected = select_trace(candidates[index])
                        if args.samples > 1:
                            selected["sampled_output_tokens"] = sum(r["output_tokens"] for r in candidates[index])
                            selected["correct_samples"] = sum(r["correct"] is True for r in candidates[index])
                            selected["completed_samples"] = sum(r["complete"] for r in candidates[index])
                        finished[index] = selected
                        candidates[index] = []
                        completed += 1
                        print(f"{completed}/{len(rows)} {selected['id']}: think={selected['reasoning_tokens']}, "
                              f"correct={selected['correct']}", flush=True)
                    while cursor in finished:
                        selected = finished.pop(cursor)
                        f.write(json.dumps(selected, ensure_ascii=False) + "\n")
                        f.flush()
                        cursor += 1
        print(f"Generation: {stats.generation_tps:.1f} tokens/s; peak memory {stats.peak_memory:.2f} GB")
    finally:
        gen.close()
        if raw:
            raw.close()
    report("capture", read_rows(output))


if __name__ == "__main__":
    main()
