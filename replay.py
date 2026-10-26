"""Replay OpenClaw tool episodes against LM Studio using fixed observations."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from corpus import BASE, check_answer, generate_api, parse_assistant, prompt_text, read_rows, split_thinking


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("--output", required=True)
    p.add_argument("--context", default="runs/openclaw-context.json")
    p.add_argument("--api", default="http://localhost:1234/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=BASE)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.api_chat = True
    context = json.loads(Path(args.context).read_text())
    if Path(args.tokenizer).suffix.lower() == '.gguf':
        from gguf import GGUFReader
        from gguf_model import tokenizer_from_gguf
        tokenizer = tokenizer_from_gguf(GGUFReader(args.tokenizer))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    episodes = read_rows(args.input)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    passed = thinking = 0
    with output.open("x") as f:
        for episode in episodes:
            messages = [{"role": "system", "content": context["system"]},
                        {"role": "user", "content": episode["prompt"]}]
            turns = []
            for step in episode["steps"]:
                row = {**step, "messages": messages, "tools": context["tools"]}
                text, tokens, finish, seconds, _ = generate_api(prompt_text(row, tokenizer), args, row, tokenizer)
                note, answer, closed = split_thinking(text)
                complete = closed and finish in ("stop", "tool_calls")
                try:
                    assistant = parse_assistant(note, answer, context["tools"])
                except (KeyError, ValueError):
                    assistant, complete = {"content": answer}, False
                correct = check_answer(row, answer, complete)
                turns.append({"assistant": assistant, "correct": correct,
                              "reasoning_tokens": len(tokenizer.encode(note, add_special_tokens=False)),
                              "output_tokens": tokens, "seconds": seconds})
                if correct is not True:
                    break
                messages.append({"role": "assistant", **assistant})
                if "observation" in step:
                    messages.append({"role": "tool", "content": step["observation"]})
            correct = len(turns) == len(episode["steps"]) and all(t["correct"] for t in turns)
            passed += correct
            thinking += sum(t["reasoning_tokens"] for t in turns)
            f.write(json.dumps({"id": episode["id"], "correct": correct, "turns": turns,
                                "model": args.model, "temperature": args.temperature,
                                "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0,
                                "max_tokens": args.max_tokens, "seed": args.seed}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{episode['id']}: {'passed' if correct else 'failed'}; {len(turns)} turns", flush=True)
    print(f"Episodes passed {passed}/{len(episodes)}; thinking tokens {thinking}")


if __name__ == "__main__":
    main()
