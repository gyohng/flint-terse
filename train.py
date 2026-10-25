"""QLoRA SFT on Flint with frozen-base probability regularization."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, iterate_batches, train
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

from corpus import BASE, read_rows


def training_attention(queries, keys, values, cache, scale, mask, sinks=None):
    """Encode sinks as a zero-value token so ordinary SDPA supplies the VJP."""
    dtype = queries.dtype
    queries, keys, values = (x.astype(mx.float32) for x in (queries, keys, values))
    if sinks is None:
        return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask).astype(dtype)
    batch, heads, length, dim = queries.shape
    kv_heads, value_dim = keys.shape[1], values.shape[-1]
    padded_dim = next(d for d in (64, 80, 128, 256) if d > dim)
    # One extra coordinate makes q·k_sink*scale equal the learned sink logit.
    sink_q = mx.broadcast_to(sinks.astype(mx.float32).reshape(1, heads, 1, 1) / scale,
                             (batch, heads, length, 1))
    queries = mx.pad(mx.concatenate((queries, sink_q), axis=-1),
                     [(0, 0), (0, 0), (0, 0), (0, padded_dim - dim - 1)])
    keys = mx.pad(keys, [(0, 0), (0, 0), (0, 0), (0, padded_dim - dim)])
    sink_k = mx.pad(mx.ones((batch, kv_heads, 1, 1)),
                    [(0, 0), (0, 0), (0, 0), (dim, padded_dim - dim - 1)])
    keys = mx.concatenate((sink_k, keys), axis=-2)
    values = mx.pad(values, [(0, 0), (0, 0), (1, 0), (0, padded_dim - value_dim)])
    if mask is not None and not isinstance(mask, str):
        allow_sink = mx.full(mask.shape[:-1] + (1,), True if mask.dtype == mx.bool_ else 0.0, dtype=mask.dtype)
        mask = mx.concatenate((allow_sink, mask), axis=-1)
    if length <= 16:
        # MLX's short-query SDPA kernel has no VJP. These tiny slices are cheap explicitly.
        q = queries.reshape(batch, kv_heads, heads // kv_heads, length, padded_dim)
        logits = (q * scale) @ keys[:, :, None].swapaxes(-1, -2)
        if isinstance(mask, str):
            mask = mx.arange(keys.shape[-2])[None, :] <= mx.arange(length)[:, None] + keys.shape[-2] - length
        if mask is not None:
            logits = mx.where(mask, logits, -float("inf")) if mask.dtype == mx.bool_ else logits + mask
        out = (mx.softmax(logits, axis=-1) @ values[:, :, None]).reshape(batch, heads, length, padded_dim)
    else:
        out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask)
    return out[..., :value_dim].astype(dtype)


def loss_masks(batch, lengths, close_think):
    positions = mx.arange(batch.shape[1])
    supervised = (positions[1:] >= lengths[:, :1]) & (positions[1:] < lengths[:, 1:])
    closing = (batch == close_think) & (positions >= lengths[:, :1])
    answer = (mx.cumsum(closing, axis=-1)[:, :-1] > 0) & supervised
    return supervised, answer


def make_loss(model, close_think, kl_weight, kl_scope="answer"):
    adapters = [m for _, m in model.named_modules() if hasattr(m, "lora_a")]

    def logits(model, inputs, start):
        hidden = model.model(inputs)[:, start:]
        if model.args.tie_word_embeddings:
            return model.model.embed_tokens.as_linear(hidden).astype(mx.float32)
        return model.lm_head(hidden).astype(mx.float32)

    def loss(model, batch, lengths, start=0):
        inputs, targets = batch[:, :-1], batch[:, start + 1:]
        supervised, answer = loss_masks(batch, lengths, close_think)
        supervised, answer = supervised[:, start:], answer[:, start:]
        anchor = supervised if kl_scope == "all" else answer
        reference = None
        if kl_weight:
            # Share the frozen quantized weights; no second 40B model in memory.
            scales = [m.scale for m in adapters]
            try:
                for m in adapters:
                    m.scale = 0.0
                reference = mx.stop_gradient(logits(model, inputs, start))
                reference = reference - mx.logsumexp(reference, axis=-1, keepdims=True)
            finally:
                for m, scale in zip(adapters, scales):
                    m.scale = scale
        predictions = logits(model, inputs, start)
        nll = nn.losses.cross_entropy(predictions, targets)
        ntokens = supervised.sum()
        result = (nll * supervised).sum() / ntokens
        if reference is not None:
            logp = predictions - mx.logsumexp(predictions, axis=-1, keepdims=True)
            kl = (mx.exp(reference) * (reference - logp)).sum(axis=-1)
            result += kl_weight * (kl * anchor).sum() / mx.maximum(anchor.sum(), 1)
        return result, ntokens

    return loss


def completion_batches(*args, **kwargs):
    for batch, lengths in iterate_batches(*args, **kwargs):
        # Keep full attention context, but never project unused prompt vocabulary logits.
        start = max(0, int(lengths[:, 0].min().item()) - 1)
        yield batch, lengths, start


def relative_weight_change(model):
    delta_squared = base_squared = 0.0
    for _, module in model.named_modules():
        if not hasattr(module, "lora_a"):
            continue
        base = module.linear
        weight = base.weight
        if isinstance(base, nn.QuantizedLinear):
            weight = mx.dequantize(base.weight, base.scales, base.biases,
                                   group_size=base.group_size, bits=base.bits, mode=base.mode)
        delta = module.scale * (module.lora_a @ module.lora_b)
        delta_squared += mx.square(delta.astype(mx.float32)).sum().item()
        base_squared += mx.square(weight.astype(mx.float32)).sum().item()
    return (delta_squared / base_squared) ** 0.5


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=BASE)
    p.add_argument("--data", default="data/prepared")
    p.add_argument("--output", default="adapters/terse")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--targets", nargs="+", default=["self_attn.q_proj", "self_attn.v_proj"])
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--kl", type=float, default=0.1)
    p.add_argument("--kl-scope", choices=("answer", "all"), default="answer")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulate", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.iters < 1 or args.accumulate < 1 or args.iters % args.accumulate:
        p.error("--iters must be positive and divisible by --accumulate")
    if args.rank < 1 or args.layers < 1 or args.kl < 0 or args.learning_rate <= 0:
        p.error("rank/layers/learning-rate must be positive; KL must be nonnegative")
    output = Path(args.output)
    if output.exists():
        p.error(f"{output} exists; choose a new --output")
    datasets = []
    for split in ("train", "valid"):
        rows = read_rows(Path(args.data) / f"{split}.jsonl")
        if len(rows) < args.batch_size:
            p.error(f"{split}: fewer examples than batch size")
        for row in rows:
            if len(row["tokens"]) > args.max_tokens:
                p.error(f"{row['id']}: exceeds --max-tokens; refusing silent truncation")
            if not 0 < row["offset"] < len(row["tokens"]):
                p.error(f"{row['id']}: invalid prompt boundary")
        datasets.append([(r["tokens"], r["offset"]) for r in rows])

    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    model, tokenizer = load(args.model)
    if model.model_type != "flint":
        p.error("This recipe expects Flint")
    if args.layers > len(model.layers):
        p.error(f"Flint has only {len(model.layers)} layers")
    from mlx_lm.models import flint

    flint.scaled_dot_product_attention = training_attention
    model.freeze()
    lora = {
        "rank": args.rank, "scale": 2.0, "dropout": 0.0,
        "keys": args.targets,
    }
    linear_to_lora_layers(model, args.layers, lora)
    print_trainable_parameters(model)
    output.mkdir(parents=True)
    config = {
        **vars(args), "fine_tune_type": "lora", "num_layers": args.layers,
        "lora_parameters": lora,
    }
    (output / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    training_args = TrainingArgs(
        iters=args.iters, batch_size=args.batch_size,
        grad_accumulation_steps=args.accumulate, max_seq_length=args.max_tokens,
        grad_checkpoint=True, val_batches=-1,
        steps_per_report=10, steps_per_eval=max(args.accumulate, args.iters // 2),
        steps_per_save=args.iters + 1, adapter_file=output / "adapters.safetensors",
    )
    train(
        model, optim.Adam(learning_rate=args.learning_rate),
        train_dataset=datasets[0], val_dataset=datasets[1], args=training_args,
        loss=make_loss(model, tokenizer.convert_tokens_to_ids("</think>"), args.kl, args.kl_scope),
        iterate_batches=completion_batches,
    )
    weights = dict(tree_flatten(model.trainable_parameters()))
    if not all(bool(mx.isfinite(w).all()) for w in weights.values()):
        raise RuntimeError("Non-finite adapter weights; do not export this run")
    print(f"Relative weight change in adapted projections: {relative_weight_change(model):.6g}")
    print(f"Saved adapter: {output}; evaluate before exporting")


if __name__ == "__main__":
    main()
