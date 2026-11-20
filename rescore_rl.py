"""Re-score RL samples (truncation reward with an absolute or per-task relative budget, or a length floor): reward = correct * (1 - alpha * max(0, L/L_ref - floor)), advantages
centred per group and scaled by the batch-wide std (DLER-style), dropping groups with no variance.
Below floor * L_ref thinking there is no further length reward, so correctness dominates.

  python3 rescore_rl.py runs/rl-it1.jsonl --ref-log data/spark-r0.log.jsonl --output runs/rl-it1b.jsonl
  python3 rescore_rl.py S.jsonl --ref-log data/spark-*.log.jsonl --rel-budget 0.3 --output S-scored.jsonl
"""

import argparse
import collections
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("samples")
    p.add_argument("--ref-log", nargs="+", required=True, help="Logs with base samples (reference thinking lengths)")
    p.add_argument("--output", required=True)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--floor", type=float, default=0.25)
    p.add_argument("--default-ref", type=float, default=400)
    p.add_argument("--budget", type=int, help="Truncation reward: 1 if correct and thinking <= budget tokens, else 0")
    p.add_argument("--rel-budget", type=float,
                   help="Relative truncation reward: budget = max(--budget-floor, R * the base's thinking length)")
    p.add_argument("--budget-floor", type=int, default=64)
    p.add_argument("--soft-budget", type=int,
                   help="Soft length reward (Kimi 1.5 / L1 style): correct * (1 - alpha * min(1, L / budget)), "
                        "so a long correct answer still beats a wrong one on hard tasks")
    args = p.parse_args()
    ref = {}
    for path in args.ref_log:
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            natives = r.get("natives", [])
            lens = ([n["think_tokens"] for n in natives if n.get("think_tokens") and n.get("correct") is True]
                    or [n["think_tokens"] for n in natives if n.get("think_tokens") and n.get("complete")])
            if lens and r["id"] not in ref:
                ref[r["id"]] = max(1, statistics.mean(lens))
    groups = collections.defaultdict(list)
    for line in Path(args.samples).read_text().splitlines():
        r = json.loads(line)
        L, Lr = r["think_tokens"], ref.get(r["id"], args.default_ref)
        if args.rel_budget:
            r["reward"] = 1.0 if r["correct"] and L <= max(args.budget_floor, args.rel_budget * Lr) else 0.0
        elif args.soft_budget:
            r["reward"] = (1 - args.alpha * min(1.0, L / args.soft_budget)) if r["correct"] else 0.0
        elif args.budget:
            r["reward"] = 1.0 if r["correct"] and L <= args.budget else 0.0
        else:
            r["reward"] = (1 - args.alpha * min(1.0, max(0.0, L / Lr - args.floor))) if r["correct"] else 0.0
        groups[r["id"]].append(r)
    centred, kept = [], 0
    for rows in groups.values():
        rewards = [r["reward"] for r in rows]
        if statistics.pstdev(rewards) < 1e-6:
            continue
        kept += 1
        mean = statistics.mean(rewards)
        for r in rows:
            r["advantage"] = r["reward"] - mean
            centred.append(r)
    sd = statistics.pstdev([r["advantage"] for r in centred]) or 1.0
    with open(args.output, "w") as f:
        for r in centred:
            r["advantage"] /= sd
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pos = [r for r in centred if r["advantage"] > 0]
    print(f"{kept}/{len(groups)} groups with signal, {len(centred)} samples; batch sd {sd:.3f}; "
          f"positive-advantage samples correct {sum(r['correct'] for r in pos)}/{len(pos)}")


if __name__ == "__main__":
    main()
