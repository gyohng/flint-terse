"""Train a small output-head LoRA on cached native Flint hidden states."""

import argparse
import json
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear


def delta_norm(head):
    a, b = head.lora_a, head.lora_b
    return head.scale * mx.sqrt(mx.maximum(((a.T @ a) * (b @ b.T)).sum(), 0))


def loss(head, batch, reasoning_kl, answer_kl, replay_kl):
    hidden = batch["hidden"]
    original = head.linear(hidden)
    delta = (head.scale * ((hidden @ head.lora_a) @ head.lora_b)).astype(hidden.dtype)
    logits = (original + delta).astype(mx.float32)
    reference = original.astype(mx.float32)
    reference = reference - mx.logsumexp(reference, axis=-1, keepdims=True)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    kl = (mx.exp(reference) * (reference - logp)).sum(axis=-1)
    if "teacher_hidden" in batch:
        teacher = head.linear(batch["teacher_hidden"]).astype(mx.float32)
        teacher = teacher - mx.logsumexp(teacher, axis=-1, keepdims=True)
        imitation = (mx.exp(teacher) * (teacher - logp)).sum(axis=-1)
        supervised = batch["thought_weights"]
        penalty = reasoning_kl * batch["thought_weights"] + answer_kl * batch["answer_weights"]
    else:
        imitation = nn.losses.cross_entropy(logits, batch["targets"])
        supervised = mx.where(batch["replay"], 0, batch["thought_weights"])
        penalty = mx.where(batch["replay"], replay_kl * batch["weights"],
                           reasoning_kl * batch["thought_weights"] + answer_kl * batch["answer_weights"])
    return (imitation * supervised + kl * penalty).sum() / batch["weights"].sum()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("features")
    p.add_argument("--replay", nargs="+", help="Cached original traces for probability preservation")
    p.add_argument("--teacher-features", nargs="+", help="Demonstration-conditioned features, in the same order as features then replay")
    p.add_argument("--output", default="adapters/head")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--reasoning-kl", type=float, default=.1)
    p.add_argument("--answer-kl", type=float, default=2)
    p.add_argument("--replay-kl", type=float, default=1)
    p.add_argument("--max-change", type=float, default=.01)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if min(args.steps, args.rank, args.batch_size, args.learning_rate, args.max_change) <= 0 or min(args.reasoning_kl, args.answer_kl, args.replay_kl) < 0:
        p.error("steps, rank, batch size, learning rate and max change must be positive; KL nonnegative")
    output = Path(args.output)
    if output.exists():
        p.error(f"{output} exists; choose a new output")
    data, metadata = mx.load(args.features, return_metadata=True)
    keys = ("hidden", "targets", "thought_weights", "answer_weights", "weights", "valid")
    data = {k: data[k] for k in keys}
    data["replay"] = mx.zeros_like(data["valid"])
    for path in args.replay or []:
        replay, replay_metadata = mx.load(path, return_metadata=True)
        if replay_metadata["model"] != metadata["model"]:
            p.error("Style and replay features must use the same base model")
        replay["replay"] = mx.ones_like(replay["valid"])
        data = {k: mx.concatenate((v, replay[k])) for k, v in data.items()}
        del replay
    if args.teacher_features:
        parts = [mx.load(path, return_metadata=True) for path in args.teacher_features]
        if any(meta["model"] != metadata["model"] for _, meta in parts):
            p.error("Teacher features must use the same frozen model")
        teacher = {k: mx.concatenate([part[k] for part, _ in parts])
                   for k in ("hidden", "targets", "valid", "thought_weights", "answer_weights")}
        del parts
        if any(
            teacher[k].shape != data[k].shape or not bool((teacher[k] == data[k]).all())
            for k in ("targets", "valid", "thought_weights", "answer_weights")
        ):
            p.error("Teacher and student features must describe the same completions and splits")
        data["teacher_hidden"] = teacher["hidden"]
        del teacher
    model, _ = load(metadata["model"])
    model.freeze()
    mx.random.seed(args.seed)
    head = LoRALinear.from_base(model.lm_head, r=args.rank, scale=2, dropout=0)
    del model
    base = head.linear
    weight = base.weight
    if isinstance(base, nn.QuantizedLinear):
        weight = mx.dequantize(weight, base.scales, base.biases, group_size=base.group_size, bits=base.bits, mode=base.mode)
    base_norm = mx.linalg.norm(weight.astype(mx.float32)).item()
    del weight
    train = mx.array([i for i, v in enumerate(data["valid"].tolist()) if not v])
    valid = mx.array([i for i, v in enumerate(data["valid"].tolist()) if v])
    if not len(train) or not len(valid):
        p.error("Need training and validation positions")
    keys = ("hidden", "targets", "thought_weights", "answer_weights", "weights", "replay")
    if args.teacher_features:
        keys += ("teacher_hidden",)
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    optimizer.init(head.trainable_parameters())
    value_grad = nn.value_and_grad(head, loss)

    @partial(mx.compile, inputs=[head.state, optimizer.state], outputs=[head.state, optimizer.state])
    def step(batch):
        value, grad = value_grad(head, batch, args.reasoning_kl, args.answer_kl, args.replay_kl)
        optimizer.update(head, grad)
        head.lora_b = head.lora_b * mx.minimum(1, args.max_change * base_norm / mx.maximum(delta_norm(head), 1e-12))
        return value

    best_loss, best_weights, best_step = float("inf"), None, 0
    for i in range(args.steps + 1):
        if i % 100 == 0 or i == args.steps:
            total = mass = 0
            for start in range(0, len(valid), args.batch_size):
                indices = valid[start:start + args.batch_size]
                values = {key: data[key][indices] for key in keys}
                amount = values["weights"].sum().item()
                total += loss(head, values, args.reasoning_kl, args.answer_kl, args.replay_kl).item() * amount
                mass += amount
            validation = total / mass
            print(f"Step {i}: validation={validation:.5f}, head change={delta_norm(head).item() / base_norm:.3%}", flush=True)
            if validation < best_loss:
                best_loss, best_step = validation, i
                best_weights = {"lm_head.lora_a": head.lora_a, "lm_head.lora_b": head.lora_b}
        if i == args.steps:
            break
        indices = train[mx.random.randint(0, len(train), (args.batch_size,))]
        value = step({key: data[key][indices] for key in keys})
        mx.eval(value, head.state, optimizer.state)
    output.mkdir(parents=True)
    config = {**vars(args), "model": metadata["model"], "fine_tune_type": "lora", "num_layers": 1,
              "lora_parameters": {"rank": args.rank, "scale": 2.0, "dropout": 0.0, "keys": ["lm_head"]}}
    (output / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    mx.save_safetensors(str(output / "adapters.safetensors"), best_weights)
    print(f"Saved step {best_step} to {output}; evaluate before exporting")


if __name__ == "__main__":
    main()
