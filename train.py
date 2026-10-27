"""QLoRA SFT on Flint with frozen-base probability regularization."""

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from gguf_model import GGUFLinear, load_model as load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.tuner.trainer import TrainingArgs, grad_checkpoint, iterate_batches, train
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

from corpus import BASE, read_rows, training_tokens


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
        if not hasattr(module, "lora_a") or not hasattr(module, "linear"):
            continue
        base = module.linear
        weight = base.weight
        if isinstance(base, GGUFLinear):
            weight = mx.dequantize(base.weight, base.scales, base.biases, group_size=32, bits=base.bits)
            if base.padded:
                weight = weight.reshape(*weight.shape[:-1], -1, 32)[..., :16].reshape(*weight.shape[:-1], -1)
        elif isinstance(base, nn.QuantizedLinear):
            weight = mx.dequantize(base.weight, base.scales, base.biases,
                                   group_size=base.group_size, bits=base.bits, mode=base.mode)
        delta = module.scale * (module.lora_a @ module.lora_b)
        delta_squared += mx.square(delta.astype(mx.float32)).sum().item()
        base_squared += mx.square(weight.astype(mx.float32)).sum().item()
    return (delta_squared / base_squared) ** 0.5


def freeze_thinking_embedding(embedding, weights, scale):
    """Carry a native-token adapter into a larger-rank attention checkpoint."""
    prefix = "model.embed_tokens."
    a, b = (weights[prefix + key] for key in ("lora_a", "lora_b"))
    padding = embedding.lora_a.shape[1] - a.shape[1]
    if padding < 0:
        raise ValueError("Attention rank must be at least the thinking adapter rank")
    embedding.lora_a = mx.pad(a, [(0, 0), (0, padding)])
    embedding.lora_b = mx.pad(b, [(0, padding), (0, 0)]) * (scale / embedding.scale)
    embedding.freeze()
    return {prefix + key: getattr(embedding, key) for key in ("lora_a", "lora_b")}


def answer_loss(logits, targets, closing, reference, answer_kl, answer_weight):
    answer = nn.log_softmax(logits[:, closing + 1:], axis=-1)
    divergence = (mx.exp(reference) * (reference - answer)).sum(axis=-1)
    # Language and reply format are chosen early; long answers must not dilute that signal.
    preservation = .5 * (divergence.mean() + divergence[:, :8].mean())
    # Numeric replies can be one token long; do not give their terminator equal reward.
    answer_ce = 0
    if answer_weight:
        nll = nn.losses.cross_entropy(logits[:, closing + 1:-1], targets[:, closing + 1:-1])
        answer_ce = .5 * (nll.mean() + nll[:, :8].mean())
    return answer_kl * preservation + answer_weight * answer_ce


def reasoning_loss(logits, targets, closing, reference, answer_kl, start_weight=1, answer_weight=0, opening_only=False, answer_only=False, entry_only=False, entry_weight=0):
    if answer_only:
        # Sampled notes may be wrong; don't imitate them or encourage earlier closing.
        thought = 0
    elif entry_only:
        # Teach the first content token, without rewarding an earlier </think>.
        thought = nn.losses.cross_entropy(logits[:, 1:2], targets[:, 1:2]).mean()
    elif opening_only:
        count = max(1, min(16, closing - 1))
        # Include <think>; fixed normalization avoids over-weighting empty thoughts.
        thought = nn.losses.cross_entropy(logits[:, :count + 1], targets[:, :count + 1]).sum() / 17
    else:
        losses = nn.losses.cross_entropy(logits[:, :closing + 1], targets[:, :closing + 1])
        positions = mx.arange(closing + 1)
        weights = mx.where((positions > 0) & (positions <= min(8, closing - 1)), start_weight, 1)
        thought = (losses * weights).sum() / weights.sum()
    if entry_weight:
        thought += entry_weight * nn.losses.cross_entropy(logits[:, 1:2], targets[:, 1:2]).mean()
    return thought + answer_loss(logits, targets, closing, reference, answer_kl, answer_weight)


def run_layers(model, hidden, start=0, stop=None):
    full = create_attention_mask(hidden)
    sliding = create_attention_mask(hidden, window_size=model.args.sliding_window)
    for layer in model.layers[start:stop]:
        hidden = layer(hidden, sliding if layer.use_sliding else full)
    return hidden


def frozen_hidden(model, ids, first_layer):
    hidden = model.model.embed_tokens(mx.array([ids[:-1]]))
    hidden = mx.stop_gradient(run_layers(model, hidden, stop=first_layer))
    mx.eval(hidden)
    return hidden


def completion_hidden(model, ids, offset, cached=None, first_layer=0):
    hidden = (model.model(mx.array([ids[:-1]])) if cached is None
              else model.model.norm(run_layers(model, cached, start=first_layer)))
    return hidden[:, offset - 1:]


def completion_logp(model, ids, offset, cached=None, first_layer=0):
    hidden = completion_hidden(model, ids, offset, cached, first_layer)
    logits = model.lm_head(hidden).astype(mx.float32)
    return -nn.losses.cross_entropy(logits, mx.array([ids[offset:]]), reduction="sum")


def preference_loss(chosen, rejected, reference_ratio, beta):
    # DPO over complete responses; subtract the frozen starting policy's log-ratio.
    return mx.logaddexp(0, -beta * (chosen - rejected - reference_ratio))


def narration_loss(logits, discouraged):
    # Only the first word after <think>; later notes and final replies stay free.
    entry = logits[:, 1]
    allowed = ~(mx.arange(entry.shape[-1])[:, None] == mx.array(discouraged)[None, :]).any(axis=-1)
    return (mx.logsumexp(entry, axis=-1) - mx.logsumexp(mx.where(allowed, entry, -mx.inf), axis=-1)).mean()


def train_paired(model, tokenizer, args, frozen_weights=None):
    trainable = [name for name, _ in tree_flatten(model.trainable_parameters())]
    if not trainable or any(not name.startswith("model.layers.") for name in trainable):
        raise ValueError("Paired training requires frozen embeddings, final norm and output head")
    first_layer = min(int(name.split(".")[2]) for name in trainable)
    print(f"Caching the frozen first {first_layer} layers; training the remaining layers", flush=True)
    parts = [mx.load(path, return_metadata=True) for path in args.teacher_features]
    if any(meta["model"] != args.model for _, meta in parts):
        raise ValueError("Teacher features must use the same frozen model")
    same_prefix = getattr(args, 'same_prefix_teacher', False)
    teacher_adapter = getattr(args, 'teacher_adapter', None) or (getattr(args, 'resume', None) if same_prefix else None)
    if teacher_adapter and any(meta.get('adapter') != teacher_adapter for _, meta in parts):
        raise ValueError('Teacher features must use the selected attention adapter')
    teacher = {k: mx.concatenate([part[k] for part, _ in parts]) for k in ("hidden", "targets")}
    del parts
    records, cursor = {"train": [], "valid": []}, 0
    preferences = 0
    model.train()
    close = tokenizer.convert_tokens_to_ids("</think>")
    narration_penalty = getattr(args, 'narration_penalty', 0)
    discouraged = []
    if narration_penalty:
        for word in ('We', 'The', 'Okay', 'Hmm', 'I'):
            tokens = tokenizer.encode(word, add_special_tokens=False)
            if len(tokens) != 1:
                raise ValueError(f'Expected one token for narration start {word!r}')
            discouraged.extend(tokens)
    for row in (r for path in args.paired for r in read_rows(path)):
        if row.get("answer_only") and not args.answer_only:
            raise ValueError(f"{row['id']}: sampled notes require --answer-only")
        ids, offset = training_tokens(row, tokenizer)
        completion = ids[offset:]
        closing = completion.index(close)
        if len(ids) > args.max_tokens:
            raise ValueError(f"{row['id']}: context exceeds --max-tokens")
        cached = frozen_hidden(model, ids, first_layer)
        preserve = row.get("preserve_only", False)
        if same_prefix:
            if teacher['targets'][cursor:cursor + len(completion)].tolist() != completion:
                raise ValueError(f"{row['id']}: same-prefix teacher feature order/token mismatch")
            hidden = teacher['hidden'][cursor if preserve else cursor + closing + 1:cursor + len(completion)]
            cursor += len(completion)
        elif preserve:
            # Preserve the starting terse policy at these same prefixes, including its thinking.
            hidden = mx.stop_gradient(completion_hidden(model, ids, offset, cached, first_layer)[0])
            mx.eval(hidden)
        else:
            original = {**row, "assistant": {**row["assistant"], "reasoning_content": row["teacher_reasoning"]}}
            original_ids, original_offset = training_tokens(original, tokenizer)
            native = original_ids[original_offset:]
            native_close = native.index(close)
            if ids[:offset] != original_ids[:original_offset] or completion[closing + 1:] != native[native_close + 1:]:
                raise ValueError(f"{row['id']}: teacher prompt or final reply changed")
            if teacher["targets"][cursor:cursor + len(native)].tolist() != native:
                raise ValueError(f"{row['id']}: teacher feature order/token mismatch")
            hidden = teacher["hidden"][cursor + native_close + 1:cursor + len(native)]
            cursor += len(native)
        rejected_responses = []
        if args.preference_weight and row.get("rejected_assistant"):
            rejected = {**row, "assistant": row["rejected_assistant"], "ending": row["rejected_ending"]}
            rejected_ids, rejected_offset = training_tokens(rejected, tokenizer, require_approved=False)
            if (len(rejected_ids) > args.max_tokens or rejected_offset != offset
                or rejected_ids[:offset] != ids[:offset]):
                raise ValueError(f"{row['id']}: invalid rejected response or mismatched prompt")
            rejected_responses.append((rejected_ids, args.preference_weight))
        if args.compact_preference_weight and not preserve and len(ids) < len(original_ids):
            if len(original_ids) > args.max_tokens:
                raise ValueError(f"{row['id']}: original reasoning exceeds --max-tokens")
            rejected_responses.append((original_ids, args.compact_preference_weight))
        preference = []
        if rejected_responses:
            chosen_logp = completion_logp(model, ids, offset, cached, first_layer).item()
        for rejected_ids, weight in rejected_responses:
            rejected_cached = frozen_hidden(model, rejected_ids, first_layer)
            rejected_logp = completion_logp(model, rejected_ids, offset, rejected_cached, first_layer).item()
            preference.append((rejected_ids, rejected_cached, chosen_logp - rejected_logp, weight))
            preferences += 1
            if preferences % 25 == 0:
                print(f"Cached {preferences} frozen preference ratios", flush=True)
        records[row["split"]].append((ids, offset, closing, hidden, cached, preserve, preference, row.get("loss_weight", 1)))
        if sum(map(len, records.values())) % 50 == 0:
            print(f"Cached {sum(map(len, records.values()))} training/validation responses", flush=True)
    if cursor != len(teacher["targets"]) or not all(records.values()):
        raise ValueError("Need aligned teacher features and nonempty train/validation splits")
    if (args.preference_weight or args.compact_preference_weight) and not any(row[-2] for row in records["train"]):
        raise ValueError("Preference training needs incorrect responses or shorter reviewed reasoning")
    del teacher
    grad_checkpoint(model.layers[0])

    def loss(model, ids, offset, closing, reference, cached, preserve, preference):
        hidden = completion_hidden(model, ids, offset, cached, first_layer)
        logits = model.lm_head(hidden).astype(mx.float32)
        targets = mx.array([ids[offset:]])
        if preserve:
            divergence = (mx.exp(reference) * (reference - nn.log_softmax(logits, axis=-1))).sum(axis=-1)
            weight = getattr(args, 'retention_kl', None)
            result = (args.kl if weight is None else weight) * .5 * (divergence.mean() + divergence[:, :8].mean())
        else:
            result = reasoning_loss(logits, targets, closing, reference,
                                    args.kl, args.start_weight, args.answer_weight, args.opening_only, args.answer_only,
                                    getattr(args, 'entry_only', False), getattr(args, 'entry_weight', 0))
            if narration_penalty and ids[offset + 1] not in discouraged:
                result += narration_penalty * narration_loss(logits, discouraged)
        chosen_logp = -nn.losses.cross_entropy(logits, targets, reduction="sum") if preference else None
        for rejected_ids, rejected_cached, reference_ratio, weight in preference:
            rejected_logp = completion_logp(model, rejected_ids, offset, rejected_cached, first_layer)
            result += weight * preference_loss(chosen_logp, rejected_logp, reference_ratio, args.preference_beta)
        return result

    gradient = nn.value_and_grad(model, loss)
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    schedule = random.Random(args.seed).choices(records["train"], weights=[r[-1] for r in records["train"]], k=args.iters)

    def example(row, training):
        ids, offset, closing, hidden, cached, preserve, preference, _ = row
        reference = nn.log_softmax(model.lm_head(hidden[None]).astype(mx.float32), axis=-1)
        mx.eval(reference)
        result = (gradient if training else loss)(model, ids, offset, closing, reference, cached, preserve, preference)
        mx.eval(result)
        return result

    best, best_weights, best_step = float("inf"), None, 0
    started = time.monotonic()
    steps = args.iters // args.accumulate
    for step in range(steps + 1):
        if step % max(25, steps // 4) == 0 or step == steps:
            model.eval()
            validation = sum(example(row, False).item() * row[-1] for row in records["valid"]) / sum(r[-1] for r in records["valid"])
            print(f"Update {step}: validation={validation:.5f}, elapsed={time.monotonic() - started:.0f}s", flush=True)
            if validation < best:
                best, best_step = validation, step
                best_weights = {**(frozen_weights or {}), **dict(tree_flatten(model.trainable_parameters()))}
                mx.save_safetensors(str(Path(args.output) / "adapters.safetensors"), best_weights)
        if step == steps:
            break
        model.train()
        accumulated, total = None, 0
        for row in schedule[step * args.accumulate:(step + 1) * args.accumulate]:
            value, grad = example(row, True)
            total += value.item()
            accumulated = grad if accumulated is None else tree_map(lambda a, b: a + b, accumulated, grad)
        optimizer.update(model, tree_map(lambda g: g / args.accumulate, accumulated))
        mx.eval(model.trainable_parameters(), optimizer.state)
        if (step + 1) % 5 == 0:
            print(f"Update {step + 1}: loss={total / args.accumulate:.5f}, elapsed={time.monotonic() - started:.0f}s", flush=True)
    model.load_weights(list(best_weights.items()), strict=False)
    print(f"Selected update {best_step}", flush=True)


def main(argv=None, loaded=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=BASE)
    p.add_argument("--data", default="data/prepared")
    p.add_argument("--paired", nargs="+", help="Compact thinking paired with unchanged native replies, in feature-file order")
    p.add_argument("--teacher-features", nargs="+", help="Original-reasoning features in paired corpus order")
    p.add_argument("--resume", help="Continue compatible adapters, optionally adding layers or targets, with a fresh optimizer")
    p.add_argument('--same-prefix-teacher', action='store_true',
                   help='Use the starting adapter at compact prefixes for answer and policy retention')
    p.add_argument('--teacher-adapter', help='Keep this fixed probability reference while resuming another adapter')
    p.add_argument('--narration-penalty', type=float, default=0,
                   help='Penalize We/The/Okay/Hmm/I only at the start of supervised thinking')
    p.add_argument("--think-adapter", help="Keep a native thinking-token adapter frozen while learning attention updates")
    p.add_argument("--opening-only", action="store_true", help="Paired mode: supervise only the first 16 reasoning tokens, retaining the full-thinking answer teacher")
    p.add_argument('--entry-only', action='store_true', help='Paired mode: supervise only the first substantive thinking token')
    p.add_argument('--entry-weight', type=float, default=0, help='Emphasize the first token after <think>, including an immediate </think>')
    p.add_argument("--answer-only", action="store_true", help="Paired mode: learn verified replies without supervising sampled notes or the closing boundary")
    p.add_argument("--start-weight", type=float, default=1)
    p.add_argument("--answer-weight", type=float, default=0, help="Paired mode: supervise verified replies without rewarding their terminator")
    p.add_argument("--preference-weight", type=float, default=0, help="Paired mode: add DPO loss against labeled incorrect responses")
    p.add_argument("--compact-preference-weight", type=float, default=0,
                   help="Paired mode: prefer reviewed shorter thinking over the original thinking, with identical final replies")
    p.add_argument("--preference-beta", type=float, default=.1)
    p.add_argument("--output", default="adapters/terse")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--train-layers", type=int, help="When resuming paired training, update only the last N layers and retain the rest")
    p.add_argument("--targets", nargs="+", default=["self_attn.q_proj", "self_attn.v_proj"])
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--kl", type=float, default=0.1)
    p.add_argument('--retention-kl', type=float, help='Separate KL weight for preserve_only rows; defaults to --kl')
    p.add_argument("--kl-scope", choices=("answer", "all"), default="answer")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulate", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.iters < 1 or args.accumulate < 1 or args.iters % args.accumulate:
        p.error("--iters must be positive and divisible by --accumulate")
    if args.same_prefix_teacher and (not args.resume or not args.paired or args.compact_preference_weight):
        p.error('--same-prefix-teacher requires paired --resume without compact preferences')
    if args.teacher_adapter and not (args.resume and args.paired):
        p.error('--teacher-adapter requires paired --resume')
    if args.narration_penalty < 0 or (args.narration_penalty and (not args.paired or args.answer_only)):
        p.error('--narration-penalty requires paired reasoning supervision and a nonnegative weight')
    if args.rank < 1 or args.layers < 1 or args.kl < 0 or args.learning_rate <= 0:
        p.error("rank/layers/learning-rate must be positive; KL must be nonnegative")
    if args.retention_kl is not None and (args.retention_kl < 0 or not args.paired):
        p.error('--retention-kl requires paired training and a nonnegative weight')
    if args.start_weight <= 0 or args.answer_weight < 0 or bool(args.paired) != bool(args.teacher_features):
        p.error("Start weight must be positive and answer weight nonnegative; --paired requires --teacher-features")
    if (min(args.preference_weight, args.compact_preference_weight) < 0 or args.preference_beta <= 0
        or ((args.preference_weight or args.compact_preference_weight) and not args.paired)):
        p.error("Preference weight must be nonnegative, beta positive, and preferences require paired training")
    if args.opening_only and not args.paired:
        p.error("--opening-only requires --paired")
    if args.entry_only and (not args.paired or args.opening_only or args.answer_only or args.narration_penalty):
        p.error('--entry-only requires paired training without other thinking-loss overrides')
    if args.entry_weight < 0 or (args.entry_weight and (not args.paired or args.entry_only or args.answer_only)):
        p.error('--entry-weight requires paired reasoning supervision and a nonnegative weight')
    if args.answer_only and (not args.paired or args.opening_only or args.preference_weight or args.compact_preference_weight):
        p.error("--answer-only requires paired training without reasoning supervision or preferences")
    if args.think_adapter and (not args.paired or args.resume):
        p.error("--think-adapter requires paired training and cannot be combined with --resume")
    if args.train_layers is not None and (not args.resume or not args.paired or not 1 <= args.train_layers <= args.layers):
        p.error('--train-layers requires paired --resume and must be between 1 and --layers')
    if args.paired and (args.batch_size != 1 or any(not t.startswith(("self_attn.", "mlp.")) for t in args.targets)):
        p.error("Paired training uses batch size 1 and transformer adapters, keeping the output head frozen")
    output = Path(args.output)
    if output.exists():
        p.error(f"{output} exists; choose a new --output")
    datasets = []
    for split in (() if args.paired else ("train", "valid")):
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
    model, tokenizer = loaded if loaded is not None else load(args.model)
    if model.model_type != "flint":
        p.error("This recipe expects Flint")
    if args.layers > len(model.layers):
        p.error(f"Flint has only {len(model.layers)} layers")
    from mlx_lm.models import flint

    flint.scaled_dot_product_attention = training_attention
    model.freeze()
    previous = json.loads((Path(args.resume) / "adapter_config.json").read_text()) if args.resume else None
    lora = {
        "rank": args.rank, "scale": previous["lora_parameters"]["scale"] if previous else 2.0, "dropout": 0.0,
        "keys": args.targets + (["model.embed_tokens"] if args.think_adapter else []),
    }
    linear_to_lora_layers(model, args.layers, lora)
    frozen_weights = None
    if args.think_adapter:
        previous = json.loads((Path(args.think_adapter) / "adapter_config.json").read_text())
        same_base = previous.get('model') == args.model
        if not same_base and Path(args.model).suffix.lower() == '.gguf' and previous.get('base_model', {}).get('sha256'):
            from checkpoint import sha256
            same_base = sha256(args.model) == previous['base_model']['sha256']
        if (not same_base or previous.get("think_tokens") is not True
            or previous.get("prompt_tokens") or previous["lora_parameters"]["keys"] != ["model.embed_tokens"]):
            p.error("--think-adapter must tune only the same model's native thinking tokens")
        frozen_weights = freeze_thinking_embedding(model.model.embed_tokens,
            mx.load(str(Path(args.think_adapter) / "adapters.safetensors")), previous["lora_parameters"]["scale"])
    if args.resume:
        previous_lora = previous["lora_parameters"]
        same_base = Path(previous['model']).resolve() == Path(args.model).resolve()
        if not same_base and Path(args.model).suffix.lower() == '.gguf' and previous.get('base_model', {}).get('sha256'):
            from checkpoint import sha256
            same_base = sha256(args.model) == previous['base_model']['sha256']
        if (not same_base or previous["num_layers"] > args.layers
            or not set(previous_lora["keys"]) <= set(args.targets)
            or any(previous_lora[k] != lora[k] for k in ("rank", "scale", "dropout"))):
            p.error("--resume must match model/rank/scale and retain every existing adapter layer and target")
        model.load_weights(str(Path(args.resume) / "adapters.safetensors"), strict=False)
    if args.train_layers is not None:
        for layer in model.layers[:-args.train_layers]:
            layer.freeze()
        trainable = dict(tree_flatten(model.trainable_parameters()))
        frozen_weights = {name: value for name, value in tree_flatten(model.parameters())
                          if name.endswith(('.lora_a', '.lora_b')) and name not in trainable}
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
    if args.paired:
        train_paired(model, tokenizer, args, frozen_weights)
    else:
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
