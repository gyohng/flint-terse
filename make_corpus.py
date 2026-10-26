"""Build reference-checked candidates and disjoint math/code/agent holdouts."""

import ast
import argparse
import json
import operator
import random
import re
import shlex
import urllib.parse
import urllib.request
from fractions import Fraction
from pathlib import Path

from code_checks import check_python
from corpus import read_rows, write_rows


def arithmetic(text):
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}

    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(str(node.value))
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](visit(node.operand))
        raise ValueError("Unsupported arithmetic")

    return visit(ast.parse(text.replace(",", ""), mode="eval").body)


def math_rows(source, limit, split):
    rows = list(enumerate(read_rows(source)))
    random.Random(351).shuffle(rows)
    result = []
    for index, row in rows:
        proof, answer = row["answer"].rsplit("####", 1)
        answer = answer.strip().replace(",", "")
        equations = re.findall(r"<<([^<>]+)>>", proof)
        if len(equations) < 2:
            continue
        try:
            for equation in equations:
                lhs, rhs = equation.split("=")
                if abs(arithmetic(lhs) - arithmetic(rhs)) > Fraction(1, 100000):
                    raise ValueError("Incorrect annotated equation")
            if arithmetic(equations[-1].split("=")[1]) != Fraction(answer):
                continue
        except (ValueError, SyntaxError, ZeroDivisionError):
            continue
        terse = "; ".join(equations) + "."
        item = {
            "id": f"gsm8k-{split}-{index}", "family": "math",
            "messages": [{"role": "user", "content": row["question"] + "\nReturn only the number in your final answer."}],
            "expected_number": answer,
        }
        if split != "test":
            item.update(
                split="valid" if len(result) % 11 == 0 else "train",
                assistant={"reasoning_content": terse, "content": answer},
                original_reasoning=re.sub(r"<<[^<>]+>>", "", proof).strip(),
                candidates=[terse, "Compute: " + terse], approved=True,
            )
        result.append(item)
        if len(result) == limit:
            break
    return result


def code_rows(limit=200):
    rows = json.loads(Path("data/sources/mbpp.json").read_text())
    random.Random(711).shuffle(rows)
    accepted = []
    for row in rows:
        tests = list(row["test_list"])
        if row["task_id"] == 454:
            # The supplied cases miss implementations that inspect only the first word.
            tests.append('assert text_match_wordz("alpha zebra") is True')
        if not check_python(row["code"], tests, row["test_imports"]):
            continue
        functions = [n for n in ast.parse(row["code"]).body if isinstance(n, ast.FunctionDef)]
        signatures = ", ".join(f"{n.name}({ast.unparse(n.args)})" for n in functions)
        accepted.append({
            "id": f"mbpp-{row['task_id']}", "family": "code",
            "messages": [{"role": "user", "content": row["prompt"] + "\nFunctions: " + signatures
                          + "\nExamples/tests:\n" + "\n".join(row["test_list"])
                          + "\nYour final answer must contain only Python code, including necessary imports."}],
            "tests": tests, "test_imports": row["test_imports"],
        })
        if len(accepted) == limit:
            break
    holdout, prompts = accepted[:40], accepted[40:]
    for i, row in enumerate(prompts):
        row["split"] = "valid" if i < 20 else "train"
    return prompts, holdout


def chat_rows(subset, count, rng, used_ids, used_prompts):
    """Import prompts only; Flint supplies its own reasoning and final reply."""
    def page(offset, length=100):
        query = urllib.parse.urlencode({"dataset": "HuggingFaceTB/smoltalk", "config": subset,
                                       "split": "train", "offset": offset, "length": length})
        with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + query, timeout=60) as response:
            return json.load(response)

    total = page(0, 1)["num_rows_total"]
    offsets = list(range(0, total, 100))
    rng.shuffle(offsets)
    result = []
    for offset in offsets:
        for item in page(offset)["rows"]:
            row = item["row"]
            if item["truncated_cells"]:
                continue
            if subset == "smol-magpie-ultra" and row["category"] not in ("creative-writing", "planning"):
                continue
            messages = row["messages"]
            end = next(i for i, m in enumerate(messages) if m["role"] == "assistant")
            messages = messages[:end]
            text = "\n".join(m["content"] for m in messages)
            if not messages or messages[-1]["role"] != "user" or not 20 <= len(text) <= 3500:
                continue
            word_counts = re.findall(r"\b(\d[\d,]*)[ -]+words?\b", text, re.I)
            if any(int(n.replace(",", "")) > 350 for n in word_counts):
                continue
            identity = f"{subset}-{item['row_idx']}"
            if identity in used_ids or json.dumps(messages, sort_keys=True) in used_prompts:
                continue
            result.append({"id": identity, "family": "prose",
                           "source": "HuggingFaceTB/smoltalk/" + subset, "messages": messages})
            if len(result) == count:
                return result
    raise ValueError(f"Only found {len(result)} suitable {subset} prompts")


def expand_corpus(output):
    output = Path(output)
    if output.exists():
        raise ValueError(f"{output} exists; choose a new output")
    existing = [r for path in Path("data").glob("*.jsonl") for r in read_rows(path)]
    used_ids = {r["id"] for r in existing if "id" in r} | {"gsm8k-train-6327", "mbpp-643"}
    used_prompts = {json.dumps(r["messages"], sort_keys=True) for r in existing if "messages" in r}
    math = [r for r in math_rows("data/sources/gsm8k-train.jsonl", 7473, "train") if r["id"] not in used_ids][:300]
    code, _ = code_rows(10000)
    code = [r for r in code if r["id"] not in used_ids][:80]
    rng = random.Random(20261026)
    chat = [r for subset, count in (("smol-magpie-ultra", 200), ("smol-rewrite", 150), ("smol-summarize", 150))
            for r in chat_rows(subset, count, rng, used_ids, used_prompts)]
    rows = []
    for row in math + code + chat:
        prompt = json.dumps(row["messages"], sort_keys=True)
        if row["id"] in used_ids or prompt in used_prompts:
            continue
        used_ids.add(row["id"])
        used_prompts.add(prompt)
        rows.append({k: v for k, v in row.items() if k in (
            "id", "family", "source", "messages", "tools", "expected_number", "tests", "test_imports")})
    rng.shuffle(rows)
    for i, row in enumerate(rows):
        row["split"] = "valid" if i % 10 == 0 else "train"
    write_rows(output, rows)
    print(f"Saved {len(rows)} new prompts: {len(math)} math, {len(code)} code, {len(chat)} chat before deduplication")


def agent_rows(context, count, start=0, holdout=False):
    system = (
        "You are an assistant using OpenClaw tools. Never claim an action before its result. "
        "Screenshot: bash /home/user/.openclaw/macshot.sh 'AppName' returns a local image path; "
        "deliver it as MEDIA:<path>. Calculator: bash /home/user/.openclaw/maccalc.sh 'expression' "
        "returns value=<display> expected=<number> img=<path>. Compare display with expected before delivery. "
        "A missing/mismatched display is not verified. Use process.poll only for a sessionId reported as running."
    )
    rows = []
    apps = ["Calculator", "Safari", "Notes", "Mail", "Terminal", "Preview"]
    for i in range(start, start + count):
        kind, n = i % 12, i // 12
        a, b = 31 + n * 7, 19 + n * 3
        image = f"/tmp/capture_{i}.jpg"
        file = f"/tmp/project_{n}/{'parse config' if n % 2 else 'decoder'}.py"
        row = {"id": f"agent-{'test' if holdout else 'source'}-{i}", "family": f"agent-{kind}",
               "tools": context["tools"], "split": "valid" if kind == 11 else "train"}
        previous = None
        if kind == 0:
            app = apps[n % len(apps)]
            prompt = f"Send a screenshot of {app} on macstudio."
            call = {"name": "exec", "arguments": {"command": f"bash /home/user/.openclaw/macshot.sh {shlex.quote(app)}"}}
            short = f"Shot {app} -> macshot; await path -> MEDIA."
            original = f"The user wants a screenshot of {app}. The documented tool is macshot.sh. I should call that script for the requested app and wait for the resulting path before delivering the image."
        elif kind == 1:
            prompt = f"Calculate {a}+{b} in macstudio Calculator and send its verified screenshot."
            call = {"name": "exec", "arguments": {"command": f"bash /home/user/.openclaw/maccalc.sh '{a}+{b}'"}}
            short = f"{a}+{b}={a+b}; maccalc -> compare readback -> MEDIA."
            original = f"The requested calculation is {a} plus {b}, which is {a+b}. The action still needs to happen in Calculator. I should use maccalc.sh, then compare its displayed and expected values before sending its screenshot."
        elif kind in (2, 3, 4):
            prompt = (f"Calculate {a}+{b} in Calculator. Deliver MEDIA:<path> only if the readback matches. "
                      "If it does not, reply exactly 'Verification failed.'; do not retry.")
            display = a+b if kind == 2 else (a+b+1 if kind == 3 else "")
            previous = (f"bash /home/user/.openclaw/maccalc.sh '{a}+{b}'", f"value={display} expected={a+b} img={image}")
            answer = f"MEDIA:{image}" if kind == 2 else "Verification failed."
            short = f"display={display or 'missing'}; expected={a+b}; {'match -> MEDIA' if kind==2 else 'unverified -> no delivery/retry'}."
            original = f"The helper returned displayed value {display or 'missing'} and expected value {a+b}. " + (
                "They match, so the task permits delivering the returned image path." if kind == 2 else
                "The display does not establish the expected result. The user explicitly forbids delivery or retry on a mismatch, so I must report verification failure.")
        elif kind == 5:
            prompt = f"Find the bug in {file}. You have not read it. First inspect it with cat."
            call = {"name": "exec", "arguments": {"command": f"cat {shlex.quote(file)}"}}
            short = "File unread -> cat first; no invented diagnosis."
            original = "The user asks for a bug diagnosis, but I have not seen the file. The required first step is to inspect its contents with cat. I cannot infer the bug from the filename."
        elif kind == 6:
            folder = f"/tmp/project_{n}"
            prompt = f"The edit is saved. Run python -m pytest tests/test_parser.py -q in {folder} before claiming it works."
            call = {"name": "exec", "arguments": {"command": "python -m pytest tests/test_parser.py -q", "workdir": folder}}
            short = "Edit != verified fix; run requested pytest in project."
            original = "Saving the edit does not establish that the bug is fixed. I must run the specified pytest command from the project directory and inspect its result before claiming success."
        elif kind == 7:
            session = f"tests-{i}"
            prompt = "Run the test suite and wait for its actual completion."
            previous = ("python -m pytest -q", f"Command still running (sessionId={session}).")
            call = {"name": "process", "arguments": {"action": "poll", "sessionId": session}}
            short = f"{session} running -> poll same handle; no restart/success claim."
            original = "The test command is still running and returned a process handle. I should poll that existing session rather than start another run or assume the tests passed."
        elif kind == 8:
            passed, failed = a, n % 4
            prompt = "Report the completed test results as JSON with integer passed/failed and boolean all_passed."
            previous = ("python -m pytest -q", f"{passed} passed, {failed} failed. Process exited with code {int(failed>0)}.")
            answer = json.dumps({"passed": passed, "failed": failed, "all_passed": failed == 0})
            short = f"passed={passed}; failed={failed}; all_passed={str(failed==0).lower()}."
            original = f"The completed test output reports {passed} passing and {failed} failing tests. All tests passed exactly when the failing count is zero. I should report those observed counts without running another test command."
        elif kind == 9:
            prompt = "A deployment has finished. Return JSON with boolean release_ready; require all tests passing and the health check returning 200."
            failed, status = (0 if n % 2 else 2), (200 if n % 3 else 503)
            previous = ("cat /tmp/deploy-result.txt", f"Build exit=0; test failures={failed}; health HTTP={status}.")
            ready = failed == 0 and status == 200
            answer = json.dumps({"release_ready": ready})
            short = f"failures={failed}, health={status}; ready=(failures=0 & HTTP=200)={str(ready).lower()}."
            original = "A successful build alone is insufficient. The release requires both zero test failures and a healthy HTTP 200 response. I should evaluate both observed conditions and return the requested boolean."
        elif kind == 10:
            prompt = "Return JSON with boolean fresh and verified. A fresh artifact must have mtime at or after task start, and capture must succeed."
            start_time, mtime, exit_code = 1000+i, 1000+i+(1 if n%2 else -20), n%3 == 0
            previous = ("cat /tmp/capture-result.txt", f"task_start={start_time}; artifact_mtime={mtime}; capture_exit={int(exit_code)}")
            fresh = mtime >= start_time
            answer = json.dumps({"fresh": fresh, "verified": fresh and not exit_code})
            short = f"mtime {mtime}{'>=' if fresh else '<'}start {start_time}; exit={int(exit_code)}; fresh={str(fresh).lower()}, verified={str(fresh and not exit_code).lower()}."
            original = "I must compare the artifact modification time to the start of the task, then separately check whether the capture command succeeded. An old artifact or failed capture cannot verify this task."
        else:
            prompt = "The file capture failed. Return JSON with boolean success and artifact set to null when no file was created."
            previous = (f"bash /home/user/.openclaw/macshot.sh {apps[n % len(apps)]}", "capture failed; exit code 1; no artifact created")
            answer = json.dumps({"success": False, "artifact": None})
            short = "Capture failed; no artifact -> success=false, artifact=null."
            original = "The capture command explicitly failed and created no file. I must report failure and a null artifact rather than inventing a path or returning an old screenshot."
        row["messages"] = [{"role": "system", "content": context["system"] if n % 8 == 0 else system},
                           {"role": "user", "content": prompt}]
        if previous:
            command, output = previous
            row["messages"] += [{"role": "assistant", "tool_calls": [{"function": {"name": "exec", "arguments": {"command": command}}}]},
                                {"role": "tool", "content": output}]
        if kind in (0, 1, 5, 6, 7):
            row["expected_tool"] = call
            assistant = {"reasoning_content": short, "tool_calls": [{"function": call}]}
        else:
            row["expected_json" if kind >= 8 else "expected"] = json.loads(answer) if kind >= 8 else answer
            assistant = {"reasoning_content": short, "content": answer}
        if not holdout:
            row.update(assistant=assistant, original_reasoning=original, candidates=[short], approved=True)
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--final-eval", action="store_true", help="Create 100 further math/code tasks, disjoint from the initial sets")
    p.add_argument("--expand", metavar="OUTPUT", help="Collect new math/code and SmolTalk prompts for native-trace training")
    p.add_argument("--fresh-eval", metavar="OUTPUT", help="Reserve 60 math and 40 code tasks absent from data and prior captures")
    args = p.parse_args()
    if args.fresh_eval and Path(args.fresh_eval).exists():
        p.error(f"{args.fresh_eval} already exists")
    sources = Path("data/sources")
    sources.mkdir(exist_ok=True)
    urls = {
        "gsm8k-train.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
        "gsm8k-test.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
        "mbpp.json": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json",
    }
    for name, url in urls.items():
        if not (sources / name).exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                (sources / name).write_bytes(response.read())
    if args.fresh_eval:
        seen_ids, seen_prompts = set(), set()
        def prompt_key(row):
            return json.dumps(row.get("messages"), sort_keys=True)
        for path in [*Path("data").glob("*.jsonl"), *Path("runs").glob("*.jsonl")]:
            for row in read_rows(path):
                seen_ids.add(row.get("id"))
                if row.get("messages"):
                    seen_prompts.add(prompt_key(row))
        code, code_test = code_rows(10000)
        selected = []
        for candidates, count in [(math_rows("data/sources/gsm8k-test.jsonl", 2000, "test"), 60),
                                  (code + code_test, 40)]:
            fresh = []
            for row in candidates:
                key = prompt_key(row)
                if row["id"] not in seen_ids and key not in seen_prompts:
                    fresh.append({**row, "split": "test"})
                    seen_ids.add(row["id"])
                    seen_prompts.add(key)
                    if len(fresh) == count:
                        break
            if len(fresh) < count:
                p.error(f"Only {len(fresh)} unused {candidates[0]['family']} tasks remain; need {count}")
            selected.extend(fresh)
        write_rows(args.fresh_eval, selected)
        print(f"Reserved {len(selected)} fresh evaluation tasks in {args.fresh_eval}")
        return
    if args.expand:
        expand_corpus(args.expand)
        return
    if args.final_eval:
        code, _ = code_rows(240)
        rows = math_rows("data/sources/gsm8k-test.jsonl", 120, "test")[60:] + code[-40:]
        for row in rows:
            row["split"] = "test"
        write_rows("data/final-eval.jsonl", rows)
        print(f"{len(rows)} final evaluation tasks")
        return
    context = json.loads(Path("runs/openclaw-context.json").read_text())
    references = math_rows("data/sources/gsm8k-train.jsonl", 600, "train") + agent_rows(context, 300)
    code, code_test = code_rows()
    holdout = math_rows("data/sources/gsm8k-test.jsonl", 60, "test") + code_test + agent_rows(context, 60, 1200, True)
    seen = set()

    def unique(rows):
        result = []
        for row in rows:
            key = json.dumps([row["messages"], row.get("tools")], sort_keys=True)
            if key not in seen:
                seen.add(key)
                result.append(row)
        return result

    # The holdout owns any duplicate; no task is allowed into both sets.
    holdout = unique(holdout)
    for row in holdout:
        row["split"] = "test"
    references, code = unique(references), unique(code)
    write_rows("data/reference-candidates.jsonl", references)
    write_rows("data/code-prompts.jsonl", code)
    write_rows("data/holdout.jsonl", holdout)
    print(f"{len(references)} reference candidates; {len(code)} coding prompts; {len(holdout)} holdout tasks")


if __name__ == "__main__":
    main()
