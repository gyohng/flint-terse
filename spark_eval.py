"""Evaluate a Flint llama-server (OpenAI-compatible chat API) on the repo's eval sets.

Same settings as the Mac evaluations: temperature 1, top-p .95, top-k 20, seed 42 per request,
4,096 output tokens. Requests run concurrently; rows are saved in input order.

  python3 spark_eval.py --label base                        # base, all eval sets -> runs/spark-base-*.jsonl
  python3 spark_eval.py --label terse --lora-scale 1        # with the server's adapter applied
  python3 spark_eval.py --compare base terse                # accuracy and thinking reduction by family
"""

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tokenizers import Tokenizer

from corpus import chat_messages, check_answer, read_rows, write_rows

DEFAULT_SETS = ["data/release-eval.jsonl", "data/confirmation-eval.jsonl", "data/direct-eval.jsonl",
                "data/code-prose-eval.jsonl", "data/retention-eval.jsonl", "data/release-chat.jsonl",
                "data/spark-code-chat-eval.jsonl", "data/spark-prose-eval.jsonl"]


def request(api, body, timeout=3600):
    req = urllib.request.Request(api.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def sample(api, model, row, tokenizer, max_tokens=4096, seed=42, temperature=1.0, lora=None):
    body = {"model": model, "messages": chat_messages(row["messages"]), "max_tokens": max_tokens,
            "temperature": temperature, "top_p": 0.95, "top_k": 20, "seed": seed}
    if row.get("tools"):
        body["tools"] = row["tools"]
    if lora is not None:
        body["lora"] = lora
    start = time.perf_counter()
    result = request(api, body)
    choice = result["choices"][0]
    message = choice["message"]
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    calls = []
    for call in message.get("tool_calls") or []:
        arguments = call["function"]["arguments"]
        calls.append({"function": {"name": call["function"]["name"],
                                   "arguments": json.loads(arguments) if isinstance(arguments, str) else arguments}})
    complete = choice["finish_reason"] in ("stop", "tool_calls")
    assistant = {"reasoning_content": reasoning, "content": content}
    if calls:
        assistant["tool_calls"] = calls
    out = {k: v for k, v in row.items() if k not in ("assistant", "tools")}
    out.update(assistant=assistant, complete=complete, finish_reason=choice["finish_reason"],
               reasoning_tokens=len(tokenizer.encode(reasoning, add_special_tokens=False).ids) if reasoning else 0,
               answer_tokens=len(tokenizer.encode(content, add_special_tokens=False).ids) if content else 0,
               output_tokens=result["usage"]["completion_tokens"], seconds=round(time.perf_counter() - start, 2),
               seed=seed)
    if "expected_tool" in row:
        expected = row["expected_tool"]
        out["correct"] = (complete and len(calls) == 1 and calls[0]["function"]["name"] == expected["name"]
                          and all(calls[0]["function"]["arguments"].get(k) == v
                                  for k, v in expected["arguments"].items()))
    else:
        out["correct"] = check_answer(row, content, complete)
    out["leak"] = any(tag in content for tag in ("<think>", "</think>"))
    return out


def run_set(api, model, rows, tokenizer, workers, **kwargs):
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda row: sample(api, model, row, tokenizer, **kwargs), rows))


def group(row, dataset):
    return row.get("family") or {"direct-eval": "direct", "openclaw-eval": "tool"}.get(dataset, dataset)


def summarize(rows):
    graded = [r for r in rows if r["correct"] is not None]
    return (f"n={len(rows)} complete={sum(r['complete'] for r in rows)} "
            f"correct={sum(r['correct'] is True for r in graded)}/{len(graded)} "
            f"think mean={statistics.mean(r['reasoning_tokens'] for r in rows):.0f} "
            f"median={statistics.median(r['reasoning_tokens'] for r in rows):g} "
            f"leaks={sum(r['leak'] for r in rows)}")


def compare(base_label, new_label, sets):
    total = {}
    for dataset in sets:
        stem = Path(dataset).stem
        base, new = (read_rows(f"runs/spark-{label}-{stem}.jsonl") for label in (base_label, new_label))
        assert [r["id"] for r in base] == [r["id"] for r in new], stem
        for b, n in zip(base, new):
            key = stem if stem in ("code-prose-eval",) else group(b, stem)
            total.setdefault(key, []).append((b, n))
    print(f"{'set':18} {'n':>4} {'base ok':>8} {'new ok':>7} {'done b/n':>9} {'think b':>8} {'think n':>8} {'cut':>6}  leaks")
    for key, pairs in total.items():
        graded = [(b, n) for b, n in pairs if b["correct"] is not None]
        # Graded tasks: thinking on tasks both solve. Ungraded: on replies both complete.
        shared = ([(b, n) for b, n in graded if b["correct"] is True and n["correct"] is True]
                  + [(b, n) for b, n in pairs if b["correct"] is None and b["complete"] and n["complete"]])
        before = sum(b["reasoning_tokens"] for b, _ in shared)
        after = sum(n["reasoning_tokens"] for _, n in shared)
        ok = lambda i: f"{sum(p[i]['correct'] is True for p in graded)}/{len(graded)}" if graded else "-"
        done = f"{sum(b['complete'] for b, _ in pairs)}/{sum(n['complete'] for _, n in pairs)}"
        cut = f"{100 * (1 - after / before):.1f}%" if before else "-"
        print(f"{key:18} {len(pairs):>4} {ok(0):>8} {ok(1):>7} {done:>9} {before:>8} {after:>8} {cut:>6}  "
              f"{sum(n['leak'] for _, n in pairs)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", default="http://127.0.0.1:8080/v1")
    p.add_argument("--model", default="flint")
    p.add_argument("--label", help="Name for this run; rows go to runs/spark-LABEL-SET.jsonl")
    p.add_argument("--sets", nargs="+", default=DEFAULT_SETS)
    p.add_argument("--tokenizer", default="models/flint/tokenizer.json")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compare", nargs=2, metavar=("BASE", "NEW"))
    p.add_argument("--lora-scale", type=float, help="llama-server: apply adapter 0 at this scale (0 = base)")
    args = p.parse_args()
    if args.compare:
        return compare(*args.compare, args.sets)
    if not args.label:
        p.error("--label is required")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    for dataset in args.sets:
        output = Path("runs") / f"spark-{args.label}-{Path(dataset).stem}.jsonl"
        if output.exists():
            print(f"{output}: exists, skipping")
            continue
        start = time.perf_counter()
        rows = run_set(args.api, args.model, read_rows(dataset), tokenizer, args.workers,
                       max_tokens=args.max_tokens, seed=args.seed,
                       lora=None if args.lora_scale is None else [{"id": 0, "scale": args.lora_scale}])
        write_rows(output, rows)
        elapsed = time.perf_counter() - start
        print(f"{Path(dataset).stem}: {summarize(rows)} | {sum(r['output_tokens'] for r in rows) / elapsed:.0f} tok/s, "
              f"{elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
