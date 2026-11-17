"""Sample groups from the served adapter for GRPO-style training (llama-server /completion).

For each prompt: G samples at temperature 1 with the adapter applied, checked by the task's tests,
number or tool call. Reward = 0 if wrong or incomplete or malformed, else 1 - alpha * min(1, L / L_ref),
where L counts thinking tokens and L_ref is the base model's thinking length for the prompt (from a
round-0 log), so correctness always dominates and shorter correct thinking wins ties. Advantages are
normalized per group (DLER-style batch std); groups with no reward variance are dropped.

  python3 rl_sample.py data/spark-pool.jsonl --output runs/rl-it1.jsonl --ref-log data/spark-r0.log.jsonl
"""

import argparse
import json
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from transformers import AutoTokenizer

from corpus import read_rows
from gen_corpus import END_TOKENS, Server, grade, split


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pool")
    p.add_argument("--output", required=True)
    p.add_argument("--ref-log", required=True)
    p.add_argument("--default-ref", type=float, default=400, help="L_ref for prompts without a base sample")
    p.add_argument("--api", default="http://127.0.0.1:8080")
    p.add_argument("--lora-scale", type=float, help="Apply the server's adapter 0 at this scale (omit to sample the plain base)")
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--families", nargs="+", default=["code", "math"])
    p.add_argument("--limit", type=int)
    p.add_argument("--max-tokens", type=int, default=4096, help="Per-sample generation cap (thinking + reply)")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained("models/flint")
    server = Server(args.api, tok, None if args.lora_scale is None else [{"id": 0, "scale": args.lora_scale}])
    ref = {}
    for line in Path(args.ref_log).read_text().splitlines():
        r = json.loads(line)
        lens = [n["think_tokens"] for n in r["natives"] if n.get("complete")]
        if lens:
            ref[r["id"]] = max(1, statistics.mean(lens))
    held = {r["id"] for f in ("data/spark-code-chat-eval.jsonl", "data/spark-prose-eval.jsonl") for r in read_rows(f)}
    rows = [r for path in args.pool.split(",") for r in read_rows(path) if r["family"] in args.families and r["id"] not in held]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.limit]
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False)) if s else 0
    stats = {"groups": 0, "kept": 0, "correct": 0, "samples": 0, "think": 0}
    lock = threading.Lock()

    def one(row):
        rng = random.Random(f"{args.seed}-{row['id']}")
        out = []
        for _ in range(args.group):
            text, complete, _ = server.generate(server.prompt(row), args.max_tokens, rng.randrange(1 << 30))
            thinking, reply, ending, clean = split(text)
            ok, assistant = grade(row, reply, complete and clean and thinking is not None)
            L = ntok(thinking or "")
            reward = (1 - args.alpha * min(1.0, L / ref.get(row["id"], args.default_ref))) if ok is True else 0.0
            out.append(dict(text=text, thinking=thinking, reply=reply, ending=ending, assistant=assistant,
                            correct=ok is True, reward=reward, think_tokens=L))
        return row, out

    start = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as pool, open(args.output, "w") as f:
        for i, fut in enumerate(as_completed([pool.submit(one, r) for r in rows])):
            row, samples = fut.result()
            rewards = [s["reward"] for s in samples]
            with lock:
                stats["groups"] += 1
                stats["samples"] += len(samples)
                stats["correct"] += sum(s["correct"] for s in samples)
                stats["think"] += sum(s["think_tokens"] for s in samples)
            sd = statistics.pstdev(rewards)
            if sd < 1e-6:
                continue
            mean = statistics.mean(rewards)
            stats["kept"] += 1
            for s in samples:
                if s["assistant"] is None or s["thinking"] is None:
                    # malformed: train it down with its raw text
                    assistant = {"reasoning_content": s["thinking"] or "", "content": s["reply"]}
                else:
                    assistant = {"reasoning_content": s["thinking"],
                                 **{k: v for k, v in s["assistant"].items() if k != "reasoning_content"}}
                f.write(json.dumps({**{k: row[k] for k in ("id", "family", "messages") if k in row},
                                    **({"tools": row["tools"]} if row.get("tools") else {}),
                                    "assistant": assistant, "ending": s["ending"] or "<|user|>",
                                    "advantage": (s["reward"] - mean) / (sd + 1e-4), "reward": s["reward"],
                                    "correct": s["correct"], "think_tokens": s["think_tokens"]},
                                   ensure_ascii=False) + "\n")
            if (i + 1) % 25 == 0:
                el = time.perf_counter() - start
                print(f"{i + 1}/{len(rows)} groups kept {stats['kept']} | pass {stats['correct'] / stats['samples']:.3f} "
                      f"| think {stats['think'] / stats['samples']:.0f} | {server.tokens / el:.0f} tok/s", flush=True)
    print(f"done: {stats['kept']}/{stats['groups']} groups with signal; pass {stats['correct'] / max(1, stats['samples']):.3f}; "
          f"mean think {stats['think'] / max(1, stats['samples']):.0f}", flush=True)


if __name__ == "__main__":
    main()
