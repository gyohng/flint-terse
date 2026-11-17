"""Insight-only thinking for code tasks, verified by regeneration (served base, llama-server).

For each code prompt with logged base samples:
  1. empty test: G answers after "<think></think>". If all pass, the target thinking is empty.
  2. otherwise: the base extracts from its own long thinking ONLY the non-obvious bits (traps the
     tests reveal, exact meaning of ambiguous wording, return type/format, edge cases that change
     the code), as terse shorthand. K candidates.
  3. each candidate is kept only if G answers regenerated after it pass at least as often as the
     base's own samples did; the shortest such candidate wins.

  python3 gen_insight.py data/spark-pool.jsonl,data/spark-code-pool.jsonl --reuse data/spark-r0.log.jsonl \
      --output data/spark-insight.jsonl
"""

import argparse
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from transformers import AutoTokenizer

from corpus import read_rows
from gen_corpus import END_TOKENS, Server, grade, split, strip_code

PROMPT = """Below is a coding request and an assistant's long private reasoning about it. Most of that reasoning is restating the request, describing the obvious approach, and going in circles. Somewhere in it are the few NON-OBVIOUS facts that actually decide whether the code passes: what the example tests reveal that the wording does not (exact meaning of odd/even, 0- or 1-based, inclusive ranges, return type such as str vs list vs tuple, rounding, sort order, which elements count), traps (built-in names shadowed by the function name, mutation of inputs, empty input), and the one key trick if the solution is not standard.

Write ONLY those non-obvious facts, as terse shorthand fragments: symbols, abbreviations, test values. Examples:
odd idx (1,3,5): "python"→"yhn" ✓
returns str(list): '[(2,), (9,)]' not list
fn named sum → shadows builtin; use loop/math, not sum()
re.search not fullmatch: "abbc" → True

Never restate the request. Never describe the standard approach. No code lines, no def/return, no code blocks. If nothing is non-obvious, output one short line naming the return type and confirming the tests are plain, e.g.: returns int; tests plain

Coding request:
{request}

Long reasoning:
{thinking}

Non-obvious facts:"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pools")
    p.add_argument("--reuse", required=True, help="Round-0 .log.jsonl with base native samples")
    p.add_argument("--output", required=True)
    p.add_argument("--api", default="http://127.0.0.1:8080")
    p.add_argument("--group", type=int, default=4, help="Answers per check")
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--limit", type=int)
    p.add_argument("--ids", help="Only prompts whose id is in this jsonl (field id)")
    p.add_argument("--no-empty", action="store_true", help="Always write a short scan note, never empty thinking")
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained("models/flint")
    server = Server(args.api, tok)
    natives = {}
    for line in Path(args.reuse).read_text().splitlines():
        r = json.loads(line)
        natives[r["id"]] = r["natives"]
    held = {r["id"] for f in ("data/spark-code-chat-eval.jsonl",) for r in read_rows(f)}
    rows = [r for path in args.pools.split(",") for r in read_rows(path)
            if r["family"] == "code" and r["id"] not in held]
    if args.ids:
        wanted = {json.loads(l)["id"] for l in open(args.ids)}
        rows = [r for r in rows if r["id"] in wanted]
    rows = rows[:args.limit]
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False)) if s else 0
    out_path, log_path = Path(args.output), Path(args.output).with_suffix(".log.jsonl")
    done = {json.loads(l)["id"] for l in log_path.read_text().splitlines()} if log_path.exists() else set()
    rows = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} code prompts", flush=True)

    def check(row, notes, rng):
        """Pass count of G answers written after <think>notes</think>, plus one passing reply."""
        passed, keep = 0, None
        for _ in range(args.group):
            text, complete, _ = server.generate(server.prompt(row, "<think>" + notes + "</think>"),
                                                args.max_tokens, rng.randrange(1 << 30))
            ending = next((t for t in END_TOKENS if text.endswith(t)), None)
            reply = (text[: -len(ending)] if ending else text).strip()
            if not complete or "<think>" in reply or "</think>" in reply:
                continue
            ok, assistant = grade(row, reply, True)
            if ok is True:
                passed += 1
                keep = keep or (reply, ending, assistant)
        return passed, keep

    def one(row):
        rng = random.Random(f"insight-{row['id']}")
        if row["id"] not in natives:  # sample the base's own reasoning first
            samples = []
            for _ in range(2):
                text, complete, _ = server.generate(server.prompt(row), 6144, rng.randrange(1 << 30))
                thinking, reply, ending, clean = split(text)
                ok, _ = grade(row, reply, complete and clean and thinking is not None)
                samples.append(dict(thinking=thinking, complete=complete and clean, correct=ok,
                                    think_tokens=ntok(thinking or "")))
            natives[row["id"]] = samples
        nat = [n for n in natives[row["id"]] if n.get("complete") and n.get("thinking")]
        record = {"id": row["id"], "tries": [], "natives": [{k: n.get(k) for k in ("think_tokens", "correct")}
                                                              for n in natives[row["id"]]]}
        if not nat:
            return record, None
        base_rate = sum(n["correct"] is True for n in natives[row["id"]]) / len(natives[row["id"]])
        need = max(1, round(base_rate * args.group))
        best = None
        if not args.no_empty:
            passed, keep = check(row, "", rng)
            record["tries"].append({"notes": "", "passed": passed})
            best = ("", keep) if passed == args.group else None
        if best is None:
            source = min(nat, key=lambda n: n["think_tokens"] if n["correct"] else 1 << 30)
            request = "\n\n".join(m.get("content") or "" for m in row["messages"] if m["role"] == "user")[-4000:]
            msg = [{"role": "user", "content": PROMPT.format(request=request, thinking=source["thinking"][-12000:])}]
            for k in range(args.candidates):
                text, complete, _ = server.generate(
                    tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False) + "<think></think>",
                    400, rng.randrange(1 << 30), temperature=0.6 + 0.2 * k)
                notes = text
                for t in END_TOKENS:
                    notes = notes.removesuffix(t)
                notes = strip_code(notes.strip().strip("`").strip())
                if not complete or "<|" in notes or "<think>" in notes or "</think>" in notes:
                    continue
                if notes.lower().strip(" .()") == "none":
                    if not args.no_empty:
                        continue  # the empty check already failed
                    notes = "tests plain; no traps"
                passed, keep = check(row, notes, rng)
                record["tries"].append({"notes": notes, "passed": passed})
                if passed >= max(need, args.group - 1) and keep and (best is None or ntok(notes) < ntok(best[0])):
                    best = (notes, keep)
        if best is None:
            return record, None
        notes, (reply, ending, assistant) = best
        out = {k: row[k] for k in ("id", "family", "messages", "tests", "test_imports") if k in row}
        out.update(assistant={"reasoning_content": notes, **{k: v for k, v in assistant.items() if k != "reasoning_content"}},
                   ending=ending or "<|user|>", approved=True, think_tokens=ntok(notes),
                   native_think_tokens=min(n["think_tokens"] for n in nat))
        return record, out

    start, n, kept = time.perf_counter(), 0, 0
    with ThreadPoolExecutor(args.workers) as pool, out_path.open("a") as out_f, log_path.open("a") as log_f:
        for fut in as_completed([pool.submit(one, r) for r in rows]):
            try:
                record, row = fut.result()
            except Exception as e:
                print(f"error: {type(e).__name__}: {e}", flush=True)
                continue
            n += 1
            log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if row:
                kept += 1
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if n % 25 == 0:
                print(f"{n}/{len(rows)} kept {kept} | {server.tokens / (time.perf_counter() - start):.0f} tok/s", flush=True)
    print(f"done: kept {kept}/{n}", flush=True)


if __name__ == "__main__":
    main()
