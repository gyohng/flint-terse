"""On-policy compressed-thinking corpus from a Flint llama-server (/completion, raw template text).

For each pool prompt:
  1. native:     sample the served model's own thinking + reply (N samples).
  2. compress:   the served model rewrites a native thinking trace as terse shorthand (K candidates,
                 style from example-vomit.txt), or decides no thinking is needed.
  3. regenerate: the served model writes its reply after <think>SHORT</think>.
  4. accept:     checks pass (tests / number / tool call), no delimiter leaks, reply complete; for
                 replies without a checker, the server judges the new reply equivalent to the native one.
  5. keep the shortest accepted candidate per prompt.

Round k>0: point --api at a server running the current adapter and pass --policy-only to keep the
shortest checked native sample instead of rewriting it ("shortest acceptable" expert iteration).

  python3 gen_corpus.py data/spark-pool.jsonl --output data/spark-r0.jsonl
"""

import argparse
import json
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from transformers import AutoTokenizer

from corpus import check_answer, parse_assistant, read_rows

STYLE = """You compress an AI assistant's private reasoning notes ("thinking") into terse shorthand. The assistant will write its full reply AFTER these notes, so the notes are a plan and the key working, never the reply itself.

Rules:
- Keep every fact, number, decision, constraint, intermediate result, edge case and check the reply depends on. Drop everything else.
- Telegraphic style: no "We need to", "Let's", "The user asks", "Maybe", "Hmm", "OK so", no restating the prompt, no repetition, no dithering, no recap.
- Use math notation and symbols (=, ≠, →, ⇒, ∴, ×, ≤, ∈), abbreviations and fragments. Caveman phrasing is fine ("need X; Y bad → use Z").
- Math: keep the actual calculation chain in symbols (each step on the way to the answer), not just the result.
- Code: approach, data structure, edge cases, complexity as fragments. NEVER write the code itself: no def/return/import lines, no code blocks.
- Writing/chat: compact plan only (tone, structure, key points).
- Tool use: which tool, which arguments, why, in one or two fragments.
- If the reply needs no thought at all (greeting, trivial request), output exactly: (none)
- Aim for about a fifth of the original length; never drop a check or a step the reply depends on.
- Output only the compressed notes. No preamble, no section labels or tags, no <think> tags.

Examples of good compressed notes:

Math task notes: 4 boxes × 12 = 48 pens; gave away 48/3 = 16 → 32 left; +10 bought → 42.
Code task notes: dedupe keep order: seen set + out list; O(n); [] → [].
Code task notes: merge intervals: sort by start; extend last if s ≤ last_end else append; touching [1,2],[2,3] → merge; empty → [].
Greeting notes: (none)
Writing task notes: apology: own mistake (wrong time, 20 min wait), no excuses, offer reschedule w/ 2 options, sincere, ~80 words.

Longer example original:
{before}

Longer example compressed:
{after}"""


FAMILY_HINT = {
    "math": ("\n\nThis is a math task: write the full calculation chain as short equations, one step per "
             "quantity, e.g. 'pens = 2.5 × 12 = 30; cost = 30 × $2 = $60 → 60'. Keep any unit conversion "
             "and a quick sanity check. Never output only the final number."),
    "code": ("\n\nThis is a code task. In words and symbols (no Python code lines, no def/return, no code "
             "blocks) keep: the approach; any trap the tests reveal (exact meaning of odd/even index, inclusive "
             "ranges, return type such as str vs list, rounding, name clashes with builtins); and a check of the "
             "approach against one or two given tests, e.g. 'idx 1,3,5 of \"python\" → y,h,n ✓'."),
}


LOSSLESS_CODE = ("\n\nThis is a code task. Rewrite EVERY step of the reasoning in shorthand: keep all of its "
                 "working — each check of an idea against the given tests (with the values), each trace through "
                 "an example, each correction or change of mind and why, edge cases, the final plan. Delete only "
                 "restatement of the request, filler words, hedging and repeated text. No Python code lines, "
                 "no def/return, no code blocks; describe code ideas in words/symbols.")


def strip_code(notes):
    """Drop code lines (def/return/import/fences and indented bodies) from notes; keep prose lines."""
    out, fence = [], False
    for line in notes.splitlines():
        if line.strip().startswith("```"):
            fence = not fence
            continue
        if fence or re.match(r"\s*(def |class |return |import |from \S+ import )", line) or re.match(r"(    |\t)\S", line):
            continue
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def load_style():
    text = Path("example-vomit.txt").read_text()
    before = text.split("Example before\n--------\n")[1].split("\n--------")[0].strip()
    after = text.split("Example after (as a sample)\n--------\n")[1].split("\n--------")[0].strip()
    return STYLE.format(before=before, after=after)


class Server:
    def __init__(self, api, tokenizer, lora=None):
        self.api = api.rstrip("/")
        self.lora = lora
        self.tok = tokenizer
        self.lock = threading.Lock()
        self.tokens = 0

    def generate(self, text, max_new_tokens, seed, temperature=1.0):
        """llama-server /completion on raw template text. Returns (text + ending token, complete, n)."""
        body = {"prompt": text, "n_predict": max_new_tokens, "temperature": temperature, "top_p": 0.95,
                "top_k": 20, "min_p": 0.0, "seed": seed, "cache_prompt": True}
        if self.lora is not None:
            body["lora"] = self.lora
        req = urllib.request.Request(self.api + "/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=3600) as r:
                    out = json.load(r)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
        n = out.get("tokens_predicted", 0)
        with self.lock:
            self.tokens += n
        content = out["content"]
        complete = out.get("stop_type") in ("eos", "word")
        if complete and not any(content.endswith(t) for t in END_TOKENS):
            # llama-server drops the end-of-generation token; tool calls yield to an observation.
            content += "<|observation|>" if "</tool_call>" in content[-200:] else "<|user|>"
        return content, bool(complete), n

    def prompt(self, row, extra=""):
        return self.tok.apply_chat_template(row["messages"], tools=row.get("tools"), add_generation_prompt=True,
                                            tokenize=False) + extra


END_TOKENS = ("<|user|>", "<|observation|>", "<|end_of_text|>", "<|developer|>")


def split(text):
    """-> (thinking or None, reply, ending, clean)"""
    ending = next((t for t in END_TOKENS if text.endswith(t)), None)
    body = text[: -len(ending)] if ending else text
    m = re.fullmatch(r"<think>(.*?)</think>(.*)", body, re.DOTALL)
    if not m:
        if body.startswith("<think>"):
            return None, "", ending, False
        return "", body, ending, "<think>" not in body and "</think>" not in body
    thinking, reply = m[1], m[2]
    clean = "<think>" not in thinking and "<think>" not in reply and "</think>" not in reply
    return thinking.strip(), reply.strip(), ending, clean


def grade(row, reply, complete):
    tools = row.get("tools") or []
    try:
        assistant = parse_assistant("", reply, tools) if tools else {"content": reply}
    except (KeyError, ValueError, TypeError):
        return False, None
    if "expected_tool" in row:
        calls = assistant.get("tool_calls", [])
        exp = row["expected_tool"]
        ok = complete and len(calls) == 1 and calls[0]["function"]["name"] == exp["name"] and all(
            calls[0]["function"]["arguments"].get(k) == v for k, v in exp["arguments"].items())
        return ok, assistant
    return check_answer(row, assistant.get("content", ""), complete), assistant


JUDGE = """Two replies to the same user request are shown. Decide whether reply B is as good as reply A for the user: same essential content, correctness, completeness, format and tone. Wording may differ. Minor differences are fine; missing requested parts, errors, or a clearly worse reply are not.

User request:
{request}

Reply A:
{a}

Reply B:
{b}

Answer with exactly one word: YES or NO."""


def judge(server, row, a, b, seed):
    request = "\n\n".join(f"[{m['role']}] {m.get('content') or ''}" for m in row["messages"])[-6000:]
    msg = [{"role": "user", "content": JUDGE.format(request=request, a=a[:6000], b=b[:6000])}]
    text, complete, _ = server.generate(server.tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
                                        + "<think></think>", 16, seed, temperature=0.3)
    verdict = text
    return verdict.strip().upper().startswith("YES")


def same_tools(a, b):
    def norm(x):
        return json.dumps(x.get("tool_calls", []), sort_keys=True)
    return norm(a) == norm(b)


def process(server, row, args, style, rng):
    ntok = lambda s: len(server.tok.encode(s, add_special_tokens=False)) if s else 0
    record = {"id": row["id"], "family": row["family"], "natives": [], "candidates": []}
    natives = []
    reused = args.reuse.get(row["id"]) if args.reuse else None
    for n in reused or []:
        complete = n["complete"]
        ok, assistant = grade(row, n["reply"], complete and n["thinking"] is not None)
        natives.append(dict(n, correct=ok, assistant=assistant))
    for i in range(0 if reused else args.samples):
        seed = rng.randrange(1 << 30)
        text, complete, n = server.generate(server.prompt(row), args.max_tokens, seed)
        thinking, reply, ending, clean = split(text)
        ok, assistant = grade(row, reply, complete and clean and thinking is not None)
        natives.append(dict(thinking=thinking, reply=reply, ending=ending, complete=complete and clean,
                            correct=ok, assistant=assistant, think_tokens=ntok(thinking)))
    record["natives"] = [{k: v for k, v in n.items() if k != "assistant"} for n in natives]
    usable = [n for n in natives if n["complete"] and n["thinking"] is not None and n["correct"] is not False]
    if not usable:
        return record, None
    candidates = []
    if args.policy_only:
        # Expert iteration: verified samples of the current policy. Code notes must exist.
        ok = [n for n in usable if n["correct"] is True
              and not (row["family"] == "code" and len(n["thinking"].strip()) < 30)]
        if ok:
            rate = len([n for n in natives if n["correct"] is True]) / len(natives)
            # Hard prompts (low pass rate) keep the longest verified notes; easy ones the shortest.
            pick = max(ok, key=lambda n: n["think_tokens"]) if rate < 0.75 else min(ok, key=lambda n: n["think_tokens"])
            candidates.append(dict(thinking=pick["thinking"], reply=pick["reply"], ending=pick["ending"],
                                   assistant=pick["assistant"], native=pick))
        record["pass_rate"] = len([n for n in natives if n["correct"] is True]) / len(natives)
    else:
        source = min(usable, key=lambda n: n["think_tokens"])
        if source["think_tokens"] == 0:
            candidates.append(dict(thinking="", reply=source["reply"], ending=source["ending"],
                                   assistant=source["assistant"], native=source))
        ratio = {"code": args.code_ratio, "math": 0.2}.get(row["family"])
        budget = max(25, int(ratio * source["think_tokens"] * 0.7)) if ratio else None  # ~0.7 words/token
        for k in range(args.candidates if source["think_tokens"] else 0):
            msg = [{"role": "system", "content": style},
                   {"role": "user", "content": "User request the notes were written for:\n"
                    + "\n\n".join(m.get("content") or "" for m in row["messages"] if m["role"] == "user")[-4000:]
                    + "\n\nOriginal notes:\n" + source["thinking"]
                    + (LOSSLESS_CODE if args.lossless and row["family"] == "code" else FAMILY_HINT.get(row["family"], ""))
                    + (f"\n\nLength: about {budget} words (the checks and steps matter more than brevity here)."
                       if budget else "")
                    + "\n\nCompressed notes:"}]
            text, complete, _ = server.generate(
                server.tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False) + "<think></think>",
                1024, rng.randrange(1 << 30), temperature=0.7 + 0.15 * k)
            short = text
            for t in END_TOKENS:
                short = short.removesuffix(t)
            short = short.strip().strip("`").strip()
            if not complete or "<think>" in short or "</think>" in short or "<|" in short:
                record["candidates"].append(dict(think_tokens=ntok(short), accepted=False, why="compressor incomplete"))
                continue
            if short.lower().strip(" .()") == "none":
                short = ""
            if row["family"] == "code":
                short = strip_code(short)
            why = None
            if row["family"] in ("code", "math") and len(short) < 30:
                why = "empty"  # code/math keep their checks
            elif ntok(short) >= (min(0.8, ratio + 0.15) if ratio else 0.6) * source["think_tokens"]:
                why = "not shorter"
            elif re.search(r"(?m)^\s*(def |class |return |import |from \S+ import |```)", short):
                why = "code in notes"  # the notes must not contain the code itself
            elif short and row["family"] == "math" and not re.search(r"[=×*/+\-−→]", short):
                why = "bare answer"  # keep the calculation chain, not just the result
            if why:
                record["candidates"].append(dict(think_tokens=ntok(short), accepted=False, why=why))
                continue
            if source["correct"] is None and not row.get("tools"):
                # Unchecked reply: keep the base's own reply; training holds it with answer KL.
                candidates.append(dict(thinking=short, reply=source["reply"], ending=source["ending"],
                                       assistant=source["assistant"], native=source, correct=None, keep_native=True))
                continue
            prefix = server.prompt(row, "<think>" + short + "</think>")
            text, complete, _ = server.generate(prefix, args.max_tokens, rng.randrange(1 << 30))
            ending = next((t for t in END_TOKENS if text.endswith(t)), None)
            reply = (text[: -len(ending)] if ending else text).strip()
            if not complete or "<think>" in reply or "</think>" in reply:
                record["candidates"].append(dict(think_tokens=ntok(short), accepted=False, why="incomplete/leak"))
                continue
            ok, assistant = grade(row, reply, True)
            candidates.append(dict(thinking=short, reply=reply, ending=ending, assistant=assistant, native=source,
                                   correct=ok))
    accepted = []
    # Code/math: prefer the shortest notes above a floor, so checks are not traded for length.
    floor = min(0.15 * (source_tokens := max((n["think_tokens"] for n in usable), default=0)), 120) \
        if row["family"] in ("code", "math") and not args.policy_only else 0
    target = {"code": args.code_ratio, "math": 0.2}.get(row["family"])
    if target and not args.policy_only:
        goal = target * source_tokens
        order = sorted(candidates, key=lambda c: abs(ntok(c["thinking"]) - goal))  # closest to the budget
    else:
        order = sorted(candidates, key=lambda c: (ntok(c["thinking"]) < floor, ntok(c["thinking"])))
    for c in order:
        if c["assistant"] is None:
            why = "unparseable"
        elif c.get("correct", c["native"]["correct"]) is True:
            why = None
        elif c.get("correct", c["native"]["correct"]) is False:
            why = "check failed"
        elif c["native"]["assistant"] and c["assistant"].get("tool_calls") is not None:
            why = None if same_tools(c["native"]["assistant"], c["assistant"]) else "different tool call"
        elif c["reply"] == c["native"]["reply"] or c.get("keep_native"):
            why = None
        else:
            why = None if judge(server, row, c["native"]["reply"], c["reply"], rng.randrange(1 << 30)) else "judge"
        record["candidates"].append(dict(think_tokens=ntok(c["thinking"]), accepted=why is None, why=why))
        if why is None:
            accepted.append(c)
            break  # sorted by length: the first accepted is the shortest
    if not accepted and args.full_fallback:
        correct = [n for n in usable if n["correct"] is True]
        if correct:  # hard task: keep the base's own full working
            n = min(correct, key=lambda n: n["think_tokens"])
            accepted = [dict(thinking=n["thinking"], reply=n["reply"], ending=n["ending"],
                             assistant=n["assistant"], native=n)]
            record["fallback"] = True
    if not accepted:
        return record, None
    best = accepted[0]
    assistant = {"reasoning_content": best["thinking"], **{k: v for k, v in best["assistant"].items()
                                                             if k != "reasoning_content"}}
    out = {k: v for k, v in row.items()}
    out.update(assistant=assistant, ending=best["ending"] or "<|user|>", approved=True,
               teacher_reasoning=best["native"]["thinking"],
               native_think_tokens=best["native"]["think_tokens"], think_tokens=ntok(best["thinking"]))
    return record, out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pool")
    p.add_argument("--output", required=True)
    p.add_argument("--api", default="http://127.0.0.1:8080")
    p.add_argument("--lora-scale", type=float, help="Apply the server's first loaded adapter at this scale")
    p.add_argument("--tokenizer", default="models/flint")
    p.add_argument("--samples", type=int, default=2, help="Native samples per prompt")
    p.add_argument("--candidates", type=int, default=3, help="Compression candidates per prompt")
    p.add_argument("--policy-only", action="store_true", help="Keep the shortest acceptable native sample")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--reuse", help="Reuse native samples from a previous .log.jsonl (repair pass)")
    p.add_argument("--families", nargs="+", help="Only these families")
    p.add_argument("--full-fallback", action="store_true", help="Keep the shortest correct native thinking if no compression passes")
    p.add_argument("--lossless", action="store_true", help="Code: rewrite all steps in shorthand instead of extracting essentials")
    p.add_argument("--code-ratio", type=float, default=0.3, help="Target note length for code, as a share of the original")
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    server = Server(args.api, tok, None if args.lora_scale is None else [{"id": 0, "scale": args.lora_scale}])
    style = load_style()
    rows = read_rows(args.pool)[: args.limit]
    if args.families:
        rows = [r for r in rows if r["family"] in args.families]
    if args.reuse:
        args.reuse = {r["id"]: r["natives"] for r in map(json.loads, Path(args.reuse).read_text().splitlines())}
        rows = [r for r in rows if r["id"] in args.reuse]
    output, log = Path(args.output), Path(args.output).with_suffix(".log.jsonl")
    done = {json.loads(l)["id"] for l in log.read_text().splitlines()} if log.exists() else set()
    rows = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} prompts to process ({len(done)} done)", flush=True)
    start, kept, n = time.perf_counter(), 0, 0
    lock = threading.Lock()
    with ThreadPoolExecutor(args.workers) as pool, output.open("a") as out_f, log.open("a") as log_f:
        futures = {pool.submit(process, server, r, args, style, random.Random(f"{args.seed}-{r['id']}")): r
                   for r in rows}
        for fut in as_completed(futures):
            try:
                record, row = fut.result()
            except Exception as e:  # keep going; the prompt is retried on resume
                print(f"{futures[fut]['id']}: {type(e).__name__}: {e}", flush=True)
                continue
            with lock:
                n += 1
                log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_f.flush()
                if row:
                    kept += 1
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                if n % 20 == 0:
                    el = time.perf_counter() - start
                    print(f"{n}/{len(rows)} kept {kept} | {server.tokens / el:.0f} tok/s | {el:.0f}s", flush=True)
    print(f"done: kept {kept}/{n}", flush=True)


if __name__ == "__main__":
    main()
