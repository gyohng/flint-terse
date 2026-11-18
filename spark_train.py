"""CUDA LoRA training on the frozen int4 GGUF base (the weights LM Studio runs).

Rows hold a prompt, (short) thinking and a final reply. Loss per row:
  cross-entropy on the thinking tokens through </think>        (skipped for preserve_only rows)
+ kl * KL(base || adapted) on the final-reply tokens, where "base" is the same frozen model with the
  adapter switched off, conditioned on the row's original thinking (`teacher_reasoning`) when its
  reply tokens line up, else on the same short thinking.
Rows are packed into sequences of up to --pack tokens; packed rows cannot attend to each other.

  python3 spark_train.py --data data/spark-r0.jsonl --output adapters/spark-r0
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

import flint_torch
from corpus import read_rows, training_tokens


def encode(rows, tokenizer, max_tokens, close):
    out, skipped = [], 0
    for row in rows:
        try:
            ids, offset = training_tokens(row, tokenizer, require_approved=False)
        except ValueError as e:
            print(f"skip {row.get('id')}: {e}")
            skipped += 1
            continue
        if len(ids) > max_tokens or close not in ids[offset:]:
            skipped += 1
            continue
        closing = ids.index(close, offset)
        item = dict(id=row["id"], ids=ids, offset=offset, closing=closing, weight=row.get("loss_weight", 1.0),
                    preserve_only=bool(row.get("preserve_only")), family=row.get("family"),
                    advantage=row.get("advantage"))
        item["teacher_ids"], item["teacher_closing"] = ids, closing
        teacher = row.get("teacher_reasoning")
        if item["advantage"] is not None:
            teacher = None
        if teacher and teacher != row["assistant"].get("reasoning_content"):
            original = {**row, "assistant": {**row["assistant"], "reasoning_content": teacher}}
            try:
                tids, toffset = training_tokens(original, tokenizer, require_approved=False)
                tclosing = tids.index(close, toffset)
                if len(tids) <= max_tokens and tids[tclosing:] == ids[closing:]:
                    item["teacher_ids"], item["teacher_closing"] = tids, tclosing
            except ValueError:
                pass
        out.append(item)
    return out, skipped


def pack(items, limit, key):
    """Greedy first-fit packing by length of items[key]."""
    bins = []
    for it in sorted(items, key=lambda i: -len(i[key])):
        for b in bins:
            if b["n"] + len(it[key]) <= limit:
                b["items"].append(it)
                b["n"] += len(it[key])
                break
        else:
            bins.append({"items": [it], "n": len(it[key])})
    return [b["items"] for b in bins]


def packed_tensors(items, key, device):
    ids, pos, seg, starts = [], [], [], []
    for j, it in enumerate(items):
        starts.append(len(ids))
        ids += it[key]
        pos += range(len(it[key]))
        seg += [j] * len(it[key])
    t = lambda x: torch.tensor(x, device=device)
    return t([ids]), (t(pos), t(seg)), starts


def answer_kl(logits, reference):
    """Per-row KL(base || adapted) over reply tokens; the first eight tokens get extra weight."""
    logp = F.log_softmax(logits.float(), dim=-1)
    kl = (reference.exp() * (reference - logp)).sum(-1)
    return 0.5 * (kl.mean() + kl[:8].mean())


def batch_loss(model, items, args, device):
    sft = [it for it in items if it["advantage"] is None]
    refs = {}
    if sft:
        # Base reference over the reply tokens (adapter off), from each row's teacher sequence.
        with torch.no_grad():
            model.set_lora(False)
            tids, tseg, tstarts = packed_tensors(sft, "teacher_ids", device)
            hidden = model.hidden(tids, seg=tseg)[0]
            for it, s in zip(sft, tstarts):
                a, b = s + it["teacher_closing"], s + len(it["teacher_ids"]) - 1
                refs[id(it)] = F.log_softmax(model.logits(hidden[a:b]).float(), -1)
            del hidden
            model.set_lora(True)
    ids, seg, starts = packed_tensors(items, "ids", device)
    hidden = model.hidden(ids, checkpoint=True, seg=seg)[0]
    total, stats = 0.0, {"think": [], "kl": [], "pg": []}
    for it, s in zip(items, starts):
        # Logit at position p predicts token p+1.
        c, n = s + it["closing"], s + len(it["ids"]) - 1
        o = s + it["offset"] - 1
        if it["advantage"] is not None:
            # Policy gradient on the whole completion (thinking, </think>, reply, ending).
            targets = ids[0, o + 1:n + 1]
            logp = -F.cross_entropy(model.logits(hidden[o:n]).float(), targets, reduction="none")
            loss = -it["advantage"] * logp.mean() * args.pg_weight
            stats["pg"].append(loss.item())
            total = total + loss
            continue
        kl = answer_kl(model.logits(hidden[c:n]), refs[id(it)])
        loss = args.kl * kl
        stats["kl"].append(kl.item())
        if not it["preserve_only"]:
            targets = ids[0, o + 1:c + 1]
            think = F.cross_entropy(model.logits(hidden[o:c]).float(), targets)
            loss = loss + think
            stats["think"].append(think.item())
        total = total + loss * it["weight"]
    return total / len(items), stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=flint_torch.GGUF_DEFAULT)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", help="Adapter directory to continue from")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=float, default=32.0, help="LoRA scale = alpha / rank")
    p.add_argument("--targets", nargs="+", default=list(flint_torch.ATTN + flint_torch.MLP))
    p.add_argument("--expert-rank", type=int, default=0, help="Per-expert LoRA rank on down_proj (multiple of 8)")
    p.add_argument("--expert-layers", type=int, default=16, help="Apply expert LoRA to the last N MoE layers")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--pack", type=int, default=8192, help="Tokens per packed sequence")
    p.add_argument("--accumulate", type=int, default=1, help="Packed sequences per optimizer update")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--kl", type=float, default=2.0)
    p.add_argument("--pg-weight", type=float, default=1.0, help="Scale of the policy-gradient loss for rows with an advantage")
    p.add_argument("--max-tokens", type=int, default=6144)
    p.add_argument("--valid", type=int, default=24, help="Rows held out for validation")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        p.error(f"{output} exists")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("models/flint")
    close = tokenizer.convert_tokens_to_ids("</think>")
    rows = [r for path in args.data for r in read_rows(path)]
    items, skipped = encode(rows, tokenizer, args.max_tokens, close)
    random.shuffle(items)
    valid, train = items[:args.valid], items[args.valid:]
    print(f"{len(train)} train rows, {len(valid)} valid, {skipped} skipped; "
          f"{sum(i['teacher_ids'] is not i['ids'] for i in items)} with original-thinking references", flush=True)

    start = time.perf_counter()
    model = flint_torch.load(args.model, device)
    print(f"Loaded int4 base in {time.perf_counter() - start:.0f}s", flush=True)
    scale = args.alpha / args.rank
    params = model.add_lora(args.rank, scale, tuple(args.targets))
    if args.resume:
        model.load_lora(load_file(Path(args.resume) / "adapters.safetensors"), scale)
        params = [q for q in model.parameters() if q.requires_grad]
    if args.expert_rank and not any(m.down_a is not None for _, m in model.moe_layers()):
        params = model.add_expert_lora(args.expert_rank, 1.0, args.expert_layers)
    print(f"LoRA rank {args.rank} on {args.targets}: {sum(q.numel() for q in params) / 1e6:.1f}M params", flush=True)

    def batches(rows):
        key = lambda it: it["ids"] if len(it["ids"]) >= len(it["teacher_ids"]) else it["teacher_ids"]
        for it in rows:
            it["_len"] = key(it)
        return pack(rows, args.pack, "_len")

    per_epoch = len(batches(train))
    steps = max(1, math.ceil(per_epoch * args.epochs / args.accumulate))
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.0, betas=(0.9, 0.99))
    warm = max(1, steps // 20)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / steps)))))
    print(f"{per_epoch} packed sequences per epoch -> {steps} updates", flush=True)

    def validate():
        with torch.no_grad():
            th, kl = [], []
            for b in batches(valid):
                _, st = batch_loss(model, b, args, device)
                th += st["think"]
                kl += st["kl"]
        return (sum(th) / max(1, len(th)), sum(kl) / max(1, len(kl)))

    vt, vk = validate()
    print(f"valid before: think {vt:.3f} kl {vk:.4f}", flush=True)
    log, queue = [], []
    for step in range(steps):
        stats = {"think": [], "kl": [], "pg": []}
        for _ in range(args.accumulate):
            if not queue:
                queue = batches(random.sample(train, len(train)))
                random.shuffle(queue)
            loss, st = batch_loss(model, queue.pop(), args, device)
            (loss / args.accumulate).backward()
            for k in stats:
                stats[k] += st.get(k, [])
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        rec = dict(step=step + 1, pg=sum(stats["pg"]) / max(1, len(stats["pg"])),
                   think=sum(stats["think"]) / max(1, len(stats["think"])),
                   kl=sum(stats["kl"]) / max(1, len(stats["kl"])), lr=schedule.get_last_lr()[0])
        log.append(rec)
        if (step + 1) % 10 == 0 or step == 0 or step + 1 == steps:
            recent = log[-10:]
            line = (f"step {step + 1}/{steps} pg {sum(r['pg'] for r in recent) / len(recent):+.4f} "
                    f"think {sum(r['think'] for r in recent) / len(recent):.3f} "
                    f"kl {sum(r['kl'] for r in recent) / len(recent):.4f} {time.perf_counter() - start:.0f}s")
            if (step + 1) % 50 == 0 or step + 1 == steps:
                vt, vk = validate()
                rec.update(valid_think=vt, valid_kl=vk)
                line += f" | valid think {vt:.3f} kl {vk:.4f}"
            print(line, flush=True)

    output.mkdir(parents=True)
    save_file(model.lora_state(), output / "adapters.safetensors")
    config = {"fine_tune_type": "lora", "num_layers": len(model.layers),
              "lora_parameters": {"rank": args.rank, "scale": scale, "dropout": 0.0,
                                  "keys": sorted({k.rsplit(".", 2)[-2] for k in model.lora_state()})},
              "base_model": {"repo": "RecursiveSI/flint-int4-gguf",
                             "revision": "9c47151036bc2c34331142dda04488bceab2bc9f",
                             "filename": "flint-int4-q4_1.gguf",
                             "sha256": "43c07baccdba42eb7528a9630ee15699b1942c297291dc1b39a061f508ee1a4b"},
              "training": {k: v for k, v in vars(args).items() if k != "output"}}
    (output / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
