"""Base-model code accuracy under thinking budgets (llama-server /completion; serve the base alone).

  scripts/serve.sh 32 &   python3 budget_probe.py --budgets 0 128 256 512

budget 0: '<think></think>' prefilled. budget N: thinking sampled up to N tokens, then '</think>'
is forced (if not already closed) and the reply is sampled. Same sampling as the evals.
"""

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

from corpus import check_answer, read_rows


def gen(prompt, n, seed):
    body = {"prompt": prompt, "n_predict": n, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": seed}
    req = urllib.request.Request("http://127.0.0.1:8080/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def run(row, budget, tok, seed=42):
    base = tok.apply_chat_template(row["messages"], add_generation_prompt=True, tokenize=False)
    if budget == 0:
        think = "<think></think>"
    else:
        out = gen(base + "<think>", budget, seed)
        text = out["content"]
        think = "<think>" + (text.split("</think>")[0] if "</think>" in text else text) + "</think>"
    out = gen(base + think, 3072, seed + 1)
    return check_answer(row, out["content"].strip(), out["stop_type"] in ("eos", "word"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", nargs="+", type=int, default=[0, 128, 256, 512])
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained("models/flint")
    rows = [r for f in ("data/release-eval.jsonl", "data/confirmation-eval.jsonl", "data/spark-code-chat-eval.jsonl")
            for r in read_rows(f) if r.get("family") == "code"]
    with ThreadPoolExecutor(32) as pool:
        for b in args.budgets:
            res = list(pool.map(lambda r: run(r, b, tok), rows))
            print(f"base, thinking budget {b}: code {sum(x is True for x in res)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
