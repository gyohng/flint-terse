"""Teach the existing <think> embedding a style; freeze every other coefficient."""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.tuner.lora import LoRAEmbedding

from corpus import BASE, read_rows, training_tokens
from train import training_attention


def adapt_embedding(model, token):
    model.freeze()
    embedding = LoRAEmbedding.from_base(model.model.embed_tokens, r=1, scale=1, dropout=0)
    embedding.lora_a = (mx.arange(embedding.lora_a.shape[0]) == token).astype(mx.float32)[:, None]
    embedding.freeze(recurse=False, keys="lora_a")
    model.model.embed_tokens = embedding
    return embedding


def prefix_cache(model, prompt):
    cache = make_prompt_cache(model)
    for start in range(0, len(prompt), 512):
        model.model(mx.array([prompt[start:start + 512]]), cache=cache)
        mx.eval([c.state for c in cache])
    return cache


def token_loss(model, tokens, cache, closing, reference, reasoning_kl, answer_kl):
    logits = model(tokens[:, :-1], cache=copy.deepcopy(cache)).astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, tokens[:, 1:])
    thought = mx.arange(ce.shape[-1]) < closing
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    kl = (mx.exp(reference) * (reference - logp)).sum(axis=-1)
    return ((ce + reasoning_kl * kl) * thought).sum() / mx.maximum(thought.sum(), 1) + \
        answer_kl * (kl * ~thought).sum() / mx.maximum((~thought).sum(), 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("--model", default=BASE)
    p.add_argument("--output", default="adapters/think-token")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--accumulate", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--max-change", type=float, default=.5)
    p.add_argument("--reasoning-kl", type=float, default=.1)
    p.add_argument("--answer-kl", type=float, default=1)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        p.error(f"{output} exists; choose a new output")
    if min(args.steps, args.accumulate, args.learning_rate, args.max_change) <= 0 or min(args.reasoning_kl, args.answer_kl) < 0:
        p.error("Steps, accumulation, learning rate and max change must be positive; KL nonnegative")
    model, tokenizer = load(args.model)
    if model.model_type != "flint" or model.args.tie_word_embeddings:
        p.error("This recipe requires Flint's separate input embedding and output head")
    opening, close = [tokenizer.convert_tokens_to_ids(t) for t in ("<think>", "</think>")]
    rows = {s: [] for s in ("train", "valid")}
    for row in read_rows(args.input):
        if not row.get("approved"):
            p.error(f"{row['id']}: unapproved example")
        ids, offset = training_tokens(row, tokenizer)
        if len(ids) > args.max_tokens or opening in ids[:offset] or ids[offset] != opening:
            p.error(f"{row['id']}: requires an untruncated prompt without prior thinking tokens")
        completion = ids[offset:]
        rows[row["split"]].append((ids[:offset], completion, completion.index(close)))
    if not all(rows.values()):
        p.error("Need train and validation examples")
    random.seed(args.seed)
    mx.random.seed(args.seed)
    random.shuffle(rows["train"])
    embedding = adapt_embedding(model, opening)
    model.train()
    radius = args.max_change * mx.linalg.norm(embedding.embedding(mx.array([opening])).astype(mx.float32)).item()
    from mlx_lm.models import flint
    flint.scaled_dot_product_attention = training_attention
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    grad_fn = nn.value_and_grad(model, token_loss)

    def example(row, training):
        prompt, completion, closing = row
        cache = prefix_cache(model, prompt)
        tokens = mx.array([completion])
        embedding.scale = 0
        reference = model(tokens[:, :-1], cache=copy.deepcopy(cache)).astype(mx.float32)
        reference = mx.stop_gradient(reference - mx.logsumexp(reference, axis=-1, keepdims=True))
        mx.eval(reference)
        embedding.scale = 1
        fn = grad_fn if training else token_loss
        result = fn(model, tokens, cache, closing, reference, args.reasoning_kl, args.answer_kl)
        mx.eval(result)
        return result

    best, best_b = float("inf"), embedding.lora_b
    started = time.monotonic()
    for step in range(args.steps + 1):
        if step % 25 == 0 or step == args.steps:
            validation = sum(example(row, False).item() for row in rows["valid"]) / len(rows["valid"])
            change = mx.linalg.norm(embedding.lora_b).item() / (radius / args.max_change)
            print(f"Step {step}: validation={validation:.5f}, token change={change:.2%}, elapsed={time.monotonic() - started:.0f}s", flush=True)
            if validation < best:
                best, best_b = validation, embedding.lora_b
        if step == args.steps:
            break
        accumulated, total = mx.zeros_like(embedding.lora_b), 0
        for micro in range(args.accumulate):
            row = rows["train"][(step * args.accumulate + micro) % len(rows["train"])]
            value, grad = example(row, True)
            total += value.item()
            accumulated += grad["model"]["embed_tokens"]["lora_b"] / args.accumulate
        optimizer.update(embedding, {"lora_b": accumulated})
        embedding.lora_b *= mx.minimum(1, radius / mx.maximum(mx.linalg.norm(embedding.lora_b), 1e-12))
        mx.eval(embedding.lora_b, optimizer.state)
        if (step + 1) % 5 == 0:
            print(f"Update {step + 1}: loss={total / args.accumulate:.5f}, elapsed={time.monotonic() - started:.0f}s", flush=True)
    output.mkdir(parents=True)
    config = {**vars(args), "fine_tune_type": "lora", "num_layers": 1,
              "lora_parameters": {"rank": 1, "scale": 1.0, "dropout": 0.0, "keys": ["model.embed_tokens"]}}
    (output / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    mx.save_safetensors(str(output / "adapters.safetensors"), {
        "model.embed_tokens.lora_a": embedding.lora_a,
        "model.embed_tokens.lora_b": best_b,
    })
    print(f"Saved {output}; evaluate before exporting")


if __name__ == "__main__":
    main()
