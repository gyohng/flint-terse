"""Learn only Flint's </think> output row; export a standard rank-one LoRA."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from corpus import BASE, read_rows, training_tokens


def stop_loss(delta, hidden, odds, labels, weights, kl):
    """Exact variable part of vocabulary CE and KL when only one logit changes."""
    logits = odds + hidden.astype(mx.float32) @ delta
    softplus = mx.logaddexp(0, logits)
    ce = softplus - labels * logits
    divergence = softplus - mx.logaddexp(0, odds) - mx.sigmoid(odds) * (logits - odds)
    return ((ce + kl * divergence) * weights).sum() / weights.sum()


def collect(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"{output} exists; choose a new output")
    rows = read_rows(args.input)
    model, tokenizer = load(args.model)
    model.eval()
    close = tokenizer.convert_tokens_to_ids("</think>")
    head = model.lm_head
    base_row = head.weight[close:close + 1]
    if isinstance(head, nn.QuantizedLinear):
        base_row = mx.dequantize(base_row, head.scales[close:close + 1], head.biases[close:close + 1],
                                group_size=head.group_size, bits=head.bits, mode=head.mode)
    features = {k: [] for k in ("hidden", "odds", "labels", "targets", "answer", "weights", "valid")}
    for i, row in enumerate(rows):
        tokens, offset = training_tokens(row, tokenizer)
        closing = tokens.index(close, offset)
        if len(tokens) > args.max_tokens:
            raise ValueError(f"{row['id']}: exceeds --max-tokens")
        cache = make_prompt_cache(model)
        for start in range(0, offset - 1, 512):
            model.model(mx.array([tokens[start:min(start + 512, offset - 1)]]), cache=cache)
            mx.eval([c.state for c in cache])
        for start in range(offset - 1, len(tokens) - 1, 256):
            end = min(start + 256, len(tokens) - 1)
            hidden = model.model(mx.array([tokens[start:end]]), cache=cache)[0]
            logits = head(hidden).astype(mx.float32)
            stop = logits[:, close]
            others = mx.where(mx.arange(logits.shape[-1]) == close, -float("inf"), logits)
            values = {
                "hidden": hidden.astype(mx.float16),
                "odds": stop - mx.logsumexp(others, axis=-1),
                "labels": (mx.array(tokens[start + 1:end + 1]) == close).astype(mx.float32),
                "targets": mx.array(tokens[start + 1:end + 1]),
                "answer": mx.arange(start + 1, end + 1) > closing,
                "weights": mx.full((end - start,), 1 / (len(tokens) - offset)),
                "valid": mx.full((end - start,), row.get("split") == "valid"),
            }
            mx.eval(values)
            for key, value in values.items():
                features[key].append(value)
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            print(f"Features {i + 1}/{len(rows)}: {row['id']}", flush=True)
    arrays = {key: mx.concatenate(value) for key, value in features.items()}
    arrays["base_row"] = base_row.astype(mx.float32).reshape(-1)
    output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output), arrays, metadata={
        "model": args.model, "close": str(close), "vocab_size": str(model.args.vocab_size),
    })
    print(f"Saved {len(arrays['odds'])} positions to {output}")


def save_adapter(output, delta, metadata):
    output = Path(output)
    output.mkdir(parents=True)
    config = {
        "model": metadata["model"], "fine_tune_type": "lora", "num_layers": 1,
        "lora_parameters": {"rank": 1, "scale": 1.0, "dropout": 0.0, "keys": ["lm_head"]},
    }
    (output / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    selector = (mx.arange(int(metadata["vocab_size"])) == int(metadata["close"])).astype(mx.float32)
    mx.save_safetensors(str(output / "adapters.safetensors"), {
        "lm_head.lora_a": delta[:, None], "lm_head.lora_b": selector[None, :],
    })


def fit(args):
    if Path(args.output).exists():
        raise ValueError(f"{args.output} exists; choose a new output")
    data, metadata = mx.load(args.input, return_metadata=True)
    train_indices = mx.array([i for i, valid in enumerate(data["valid"].tolist()) if not valid])
    valid_indices = mx.array([i for i, valid in enumerate(data["valid"].tolist()) if valid])
    if not len(train_indices) or not len(valid_indices):
        raise ValueError("Need both training and validation examples")
    keys = ("hidden", "odds", "labels", "weights")
    validation = [data[key][valid_indices] for key in keys]
    delta = mx.zeros_like(data["base_row"])
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    mx.random.seed(args.seed)
    gradient = mx.value_and_grad(stop_loss)
    radius = args.max_change * mx.linalg.norm(data["base_row"])
    best_loss, best_delta, best_step = float("inf"), delta, 0
    for step in range(args.steps + 1):
        if step % 100 == 0 or step == args.steps:
            valid = stop_loss(delta, *validation, args.kl).item()
            change = (mx.linalg.norm(delta) / mx.linalg.norm(data["base_row"])).item()
            print(f"Step {step}: validation={valid:.6f}, closing-row change={change:.3%}", flush=True)
            if valid < best_loss:
                best_loss, best_delta, best_step = valid, delta, step
        if step == args.steps:
            break
        indices = train_indices[mx.random.randint(0, len(train_indices), (args.batch_size,))]
        loss, grad = gradient(delta, *(data[key][indices] for key in keys), args.kl)
        delta = optimizer.apply_gradients({"delta": grad}, {"delta": delta})["delta"]
        delta = delta * mx.minimum(1, radius / mx.maximum(mx.linalg.norm(delta), 1e-12))
        mx.eval(delta, optimizer.state)
    save_adapter(args.output, best_delta, metadata)
    print(f"Saved step {best_step} to {args.output}; evaluate before exporting")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect")
    p.add_argument("input")
    p.add_argument("--model", default=BASE)
    p.add_argument("--output", default="runs/stop-features.safetensors")
    p.add_argument("--max-tokens", type=int, default=6144)
    p = sub.add_parser("fit")
    p.add_argument("input")
    p.add_argument("--output", default="adapters/stop")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--kl", type=float, default=1)
    p.add_argument("--max-change", type=float, default=.25)
    p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "fit" and (min(args.steps, args.batch_size, args.learning_rate, args.max_change) <= 0 or args.kl < 0):
        parser.error("steps, batch-size, learning-rate and max-change must be positive; KL must be nonnegative")
    (collect if args.command == "collect" else fit)(args)


if __name__ == "__main__":
    main()
