"""Extra code prompts for RL from MBPP (data/sources/mbpp-full.jsonl), excluding every eval task.

Each task appears once, alternating the two prompt styles used in the eval sets: code-only and
conversational ("Show the Python function first, then explain..."). Tests are the MBPP asserts.

  python3 build_code_pool.py --output data/spark-code-pool.jsonl
"""

import argparse
import glob
import json
import re

from corpus import read_rows, write_rows

STYLES = {
    "mbpp": "Your final answer must contain only Python code, including necessary imports.",
    "prose-code-mbpp": "Show the Python function first, then explain the approach and give a usage example.",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/spark-code-pool.jsonl")
    args = p.parse_args()
    banned = set()
    for path in glob.glob("data/*eval*.jsonl") + ["data/holdout.jsonl", "data/release-chat.jsonl"]:
        for row in read_rows(path):
            m = re.search(r"mbpp-(\d+)$", row["id"])
            if m:
                banned.add(int(m[1]))
    out = []
    for i, task in enumerate(read_rows("data/sources/mbpp-full.jsonl")):
        if task["task_id"] in banned or task.get("test_setup_code"):
            continue
        tests = task["test_list"]
        names = re.findall(r"^def (\w+\([^)]*\))", task["code"], re.M)
        called = re.findall(r"assert\s+\(?\s*(?:math\.isclose\()?\s*(\w+)\(", tests[0])
        sig = next((n for n in names if called and n.startswith(called[0] + "(")), names[0] if names else None)
        if not sig:
            continue
        imports = sorted({f"import {m}" for m in re.findall(r"\b(math|re|collections|itertools|heapq)\.", " ".join(tests))})
        prefix = "mbpp" if i % 2 else "prose-code-mbpp"
        text = (f"{task['text']}\nFunctions: {sig}\nExamples/tests:\n" + "\n".join(tests) + "\n" + STYLES[prefix])
        out.append({"id": f"rl-{prefix}-{task['task_id']}", "family": "code",
                    "messages": [{"role": "user", "content": text}], "tests": tests, "test_imports": imports})
    write_rows(args.output, out)
    print(f"{args.output}: {len(out)} code prompts ({len(banned)} eval tasks excluded)")


if __name__ == "__main__":
    main()
