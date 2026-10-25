"""Learn reserved input-token embeddings; preserve replies with full-reasoning teacher probabilities."""

import argparse
import copy
import json
import random
import time
from os.path import commonprefix
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.lora import LoRAEmbedding

from corpus import BASE, read_rows, training_tokens
from train import training_attention


def adapt_prompt(model, token_ids, initial_ids):
    model.freeze()
    base = model.model.embed_tokens
    initial = base(mx.array(initial_ids)).astype(mx.float32)
    embedding = LoRAEmbedding.from_base(base, r=len(token_ids), scale=1, dropout=0)
    embedding.lora_a = (mx.arange(embedding.lora_a.shape[0])[:, None] == mx.array(token_ids)[None, :]).astype(mx.float32)
    embedding.lora_b = initial - base(mx.array(token_ids)).astype(mx.float32)
    embedding.freeze(recurse=False, keys="lora_a")
    model.model.embed_tokens = embedding
    return embedding, initial


def prefix_cache(model, prompt, cache=None):
    cache = make_prompt_cache(model) if cache is None else copy.deepcopy(cache)
    for start in range(0, len(prompt), 512):
        model.model(mx.array([prompt[start:start + 512]]), cache=cache)
        mx.eval([c.state for c in cache])
    return cache


def prompt_loss(model, inputs, targets, cache, prefix_length, closing, reference, answer_kl, start_weight=1):
    hidden = model.model(inputs, cache=copy.deepcopy(cache))[:, prefix_length:]
    logits = model.lm_head(hidden).astype(mx.float32)
    losses = nn.losses.cross_entropy(logits[:, :closing + 1], targets[:, :closing + 1])
    positions = mx.arange(closing + 1)
    weights = mx.where((positions > 0) & (positions <= min(8, closing - 1)), start_weight, 1)
    thought = (losses * weights).sum() / weights.sum()
    answer = nn.log_softmax(logits[:, closing + 1:], axis=-1)
    preservation = (mx.exp(reference) * (reference - answer)).sum(axis=-1).mean()
    return thought + answer_kl * preservation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("--teacher-features", nargs="+", required=True, help="Original-reasoning features in corpus order")
    p.add_argument("--model", default=BASE)
    p.add_argument("--output", default="adapters/prompt")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--accumulate", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--max-change", type=float, default=1)
    p.add_argument("--strength", dest="prompt_strength", type=float, default=1,
                   help="Blend the saved prefix with its initialization (0 to 1)")
    p.add_argument("--answer-kl", type=float, default=1)
    p.add_argument("--start-weight", type=float, default=1, help="Weight the first eight reasoning tokens to teach a direct start")
    p.add_argument("--prompt-role", choices=("system", "developer", "user"), default="system")
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--init-text", default="Brief thoughts. Full answers.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        p.error(f"{output} exists; choose a new output")
    if min(args.steps, args.accumulate, args.learning_rate, args.max_change, args.start_weight) <= 0 or args.answer_kl < 0:
        p.error("Steps, accumulation, learning rate and max change must be positive; KL nonnegative")
    if not 0 <= args.prompt_strength <= 1:
        p.error("Strength must be between 0 and 1")
    features = [mx.load(path, return_metadata=True) for path in args.teacher_features]
    if any(meta["model"] != args.model for _, meta in features):
        p.error("Teacher features must come from the same frozen model")
    hidden = mx.concatenate([f["hidden"] for f, _ in features])
    cached_targets = mx.concatenate([f["targets"] for f, _ in features])
    del features
    model, tokenizer = load(args.model)
    if model.model_type != "flint" or model.args.tie_word_embeddings:
        p.error("This recipe requires Flint's separate input embedding and output head")
    initial_ids = tokenizer.encode(args.init_text, add_special_tokens=False)
    reserved = sorted((i, t) for t, i in tokenizer.get_added_vocab().items() if t.startswith("<|reserved_"))
    if not 0 < len(initial_ids) <= len(reserved):
        p.error("Initialization must fit in the available reserved tokens")
    selected = reserved[-len(initial_ids):]
    token_ids, token_names = [i for i, _ in selected], [t for _, t in selected]
    opening, close, assistant = [tokenizer.convert_tokens_to_ids(t) for t in ("<think>", "</think>", "<|assistant|>")]
    control_ids = ([tokenizer.convert_tokens_to_ids(f"<|{args.prompt_role}|>")] if args.prompt_role != "user" else []) + token_ids
    records, cursor = {"train": [], "valid": []}, 0
    for row in read_rows(args.input):
        ids, offset = training_tokens(row, tokenizer)
        original = {**row, "assistant": {**row["assistant"], "reasoning_content": row["teacher_reasoning"]}}
        teacher_ids, teacher_offset = training_tokens(original, tokenizer)
        completion, teacher = ids[offset:], teacher_ids[teacher_offset:]
        closing, teacher_close = completion.index(close), teacher.index(close)
        if ids[offset - 1] != assistant or completion[0] != opening or set(token_ids) & set(ids):
            p.error(f"{row['id']}: unexpected prompt or reserved token collision")
        if len(ids) + len(control_ids) > args.max_tokens or ids[:offset] != teacher_ids[:teacher_offset]:
            p.error(f"{row['id']}: invalid context or mismatched teacher prompt")
        if completion[closing + 1:] != teacher[teacher_close + 1:]:
            p.error(f"{row['id']}: final reply changed")
        if cached_targets[cursor:cursor + len(teacher)].tolist() != teacher:
            p.error(f"{row['id']}: teacher feature order/token mismatch")
        answer_hidden = hidden[cursor + teacher_close + 1:cursor + len(teacher)]
        records[row["split"]].append((ids[:offset - 1], completion, closing, answer_hidden, row.get("loss_weight", 1)))
        cursor += len(teacher)
    if cursor != len(cached_targets) or not all(records.values()):
        p.error("Need aligned teacher features and nonempty train/validation splits")
    mx.random.seed(args.seed)
    embedding, initial = adapt_prompt(model, token_ids, initial_ids)
    initial_b = embedding.lora_b
    radius = args.max_change * mx.linalg.norm(initial, axis=-1, keepdims=True)
    model.train()
    from mlx_lm.models import flint
    flint.scaled_dot_product_attention = training_attention
    # Reuse the frozen OpenClaw instructions; keep the original chunk boundaries.
    long_prompts = [r[0] for split in records.values() for r in split if len(r[0]) >= 1024]
    shared_prefix = commonprefix(long_prompts) if long_prompts else []
    shared_prefix = shared_prefix[:len(shared_prefix) // 512 * 512]
    shared_cache = prefix_cache(model, shared_prefix) if shared_prefix else None
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    grad_fn = nn.value_and_grad(model, prompt_loss)
    schedule = random.Random(args.seed).choices(records["train"], weights=[r[-1] for r in records["train"]],
                                                k=args.steps * args.accumulate)

    def example(row, training):
        prompt, completion, closing, answer_hidden, _ = row
        if shared_cache is not None and prompt[:len(shared_prefix)] == shared_prefix:
            cache = prefix_cache(model, prompt[len(shared_prefix):], shared_cache)
        else:
            cache = prefix_cache(model, prompt)
        inputs = mx.array([control_ids + [assistant] + completion[:-1]])
        reference = mx.stop_gradient(nn.log_softmax(model.lm_head(answer_hidden[None]).astype(mx.float32), axis=-1))
        mx.eval(reference)
        fn = grad_fn if training else prompt_loss
        result = fn(model, inputs, mx.array([completion]), cache, len(control_ids), closing, reference, args.answer_kl, args.start_weight)
        mx.eval(result)
        return result

    best, best_b, best_step = float("inf"), embedding.lora_b, 0
    started = time.monotonic()
    for step in range(args.steps + 1):
        if step % 25 == 0 or step == args.steps:
            validation = sum(example(row, False).item() * row[-1] for row in records["valid"]) / sum(r[-1] for r in records["valid"])
            print(f"Step {step}: validation={validation:.5f}, elapsed={time.monotonic() - started:.0f}s", flush=True)
            if validation < best:
                best, best_b, best_step = validation, embedding.lora_b, step
        if step == args.steps:
            break
        accumulated, total = mx.zeros_like(embedding.lora_b), 0
        for row in schedule[step * args.accumulate:(step + 1) * args.accumulate]:
            value, grad = example(row, True)
            total += value.item()
            accumulated += grad["model"]["embed_tokens"]["lora_b"] / args.accumulate
        optimizer.update(embedding, {"lora_b": accumulated})
        delta = embedding.lora_b - initial_b
        embedding.lora_b = initial_b + delta * mx.minimum(1, radius / mx.maximum(mx.linalg.norm(delta, axis=-1, keepdims=True), 1e-12))
        mx.eval(embedding.lora_b, optimizer.state)
        if (step + 1) % 5 == 0:
            print(f"Update {step + 1}: loss={total / args.accumulate:.5f}, elapsed={time.monotonic() - started:.0f}s", flush=True)
    output.mkdir(parents=True)
    config = {**vars(args), "fine_tune_type": "lora", "num_layers": 1, "prompt_tokens": token_names,
              "lora_parameters": {"rank": len(token_ids), "scale": 1.0, "dropout": 0.0, "keys": ["model.embed_tokens"]}}
    (output / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    mx.save_safetensors(str(output / "adapters.safetensors"), {
        "model.embed_tokens.lora_a": embedding.lora_a,
        "model.embed_tokens.lora_b": initial_b + args.prompt_strength * (best_b - initial_b),
    })
    print(f"Saved step {best_step} to {output}; {len(token_ids)} input tokens, strength={args.prompt_strength}; evaluate before exporting")


if __name__ == "__main__":
    main()
