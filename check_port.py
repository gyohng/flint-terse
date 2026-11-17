"""Check the PyTorch Flint port (and an optional adapter) against llama-server on the same int4 GGUF.

llama-server greedily continues a few prompts and reports each generated token's log-probability;
the port scores the same tokens. Small differences come from llama.cpp's quantized activations.

  scripts/serve.sh 4 [--lora adapters/X/adapter.gguf --lora-init-without-apply] &
  python3.15 check_port.py [--adapter adapters/X] [--scales 0 1 3]
"""

import argparse
import json
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

import flint_torch

PROMPTS = ["Write a Python function that reverses the words in a sentence.",
           "Lulu has $6, Nora has five times as much. How much does Nora have?"]


def llama(api, text, scale, adapter, tokens=48):
    body = {"prompt": text, "n_predict": tokens, "temperature": 0, "n_probs": 1, "cache_prompt": False}
    if adapter:
        body["lora"] = [{"id": 0, "scale": scale}]
    req = urllib.request.Request(api.rstrip("/") + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    probs = json.load(urllib.request.urlopen(req))["completion_probabilities"]
    return [p["id"] for p in probs], [p["logprob"] for p in probs]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", default="http://127.0.0.1:8080")
    p.add_argument("--adapter", help="Adapter directory (adapters.safetensors + adapter_config.json) served as LoRA 0")
    p.add_argument("--scales", nargs="+", type=float, default=[1.0], help="LoRA scales to compare")
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained("models/flint")
    model = flint_torch.load()
    base_scale = 1.0
    if args.adapter:
        cfg = json.loads((Path(args.adapter) / "adapter_config.json").read_text())
        base_scale = cfg["lora_parameters"]["scale"]
        model.load_lora(load_file(Path(args.adapter) / "adapters.safetensors"), base_scale)
    for prompt in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        ids = tok.encode(text, add_special_tokens=False)
        for scale in (args.scales if args.adapter else [0.0]):
            toks, lps = llama(args.api, text, scale, args.adapter)
            for _, proj in model.projections():
                proj.scale = base_scale * scale
            with torch.no_grad():
                lp = torch.log_softmax(model(torch.tensor([ids + toks], device="cuda"))[0].float(), -1)
            ours = [lp[len(ids) - 1 + i, t].item() for i, t in enumerate(toks)]
            agree = sum(lp[len(ids) - 1 + i].argmax().item() == t for i, t in enumerate(toks)) / len(toks)
            d = [abs(a - b) for a, b in zip(ours, lps)]
            print(f"scale {scale}: mean |dlogp| {sum(d) / len(d):.3f}  max {max(d):.3f}  "
                  f"argmax agreement {agree:.2f}  | {tok.decode(toks)[:60]!r}", flush=True)


if __name__ == "__main__":
    main()
