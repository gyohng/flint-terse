"""Merge an MLX LoRA adapter into the MLX int4 Flint base as a complete model LM Studio can list and load.

Adapted layers are dequantized, merged and stored unquantized as float16; mlx-lm loads any layer without
`.scales` as a plain Linear, so every other tensor stays the original int4. (mlx_lm.fuse would re-quantize
the merged layers to 4 bits, which rounds away most of the LoRA delta; bfloat16 rounds away 5-18% of it.)
Verified against llama.cpp with the GGUF adapter: mean |dlogp| 0.010, 100% top-token agreement.

  python3 scripts/merge_mlx.py models/flint-int4-mlx merged/hf/flint-terse-lora/mlx out/flint-terse-mlx
"""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx

SKIP = {"flint-int4.json", "README.md", "assets", ".cache", "chat_template.jinja"}


def merge(base, adapter, output, chat_template, dtype=mx.float16):
    scale = json.loads((adapter / "adapter_config.json").read_text())["lora_parameters"]["scale"]
    lora = mx.load(str(adapter / "adapters.safetensors"))
    adapted = {k.rsplit(".", 1)[0] for k in lora}
    quant = json.loads((base / "config.json").read_text())["quantization"]
    output.mkdir(parents=True, exist_ok=True)
    merged = 0
    for shard in sorted(base.glob("model-*.safetensors")):
        weights, out = mx.load(str(shard)), {}
        for key, value in weights.items():
            module, leaf = key.rsplit(".", 1)
            if module not in adapted:
                out[key] = value
            elif leaf == "weight":
                packed = value.view(mx.uint32) if value.dtype == mx.uint8 else value   # stored as bytes
                w = mx.dequantize(packed, weights[f"{module}.scales"], weights[f"{module}.biases"],
                                  group_size=quant["group_size"], bits=quant["bits"]).astype(mx.float32)
                a, b = lora[f"{module}.lora_a"].astype(mx.float32), lora[f"{module}.lora_b"].astype(mx.float32)
                delta = b @ a if a.ndim == 3 else (a @ b).T   # experts [E,out,r]@[E,r,in]; else ([in,r]@[r,out])^T
                out[key] = (w + scale * delta).astype(dtype)
                mx.eval(out[key])
                merged += 1
            # the adapted layers' quantization scales and biases are dropped
        mx.save_safetensors(str(output / shard.name), out, metadata={"format": "mlx"})
        print(f"{shard.name}: {merged} layers merged so far", flush=True)
    assert merged == len(adapted), f"merged {merged} of {len(adapted)} adapted layers"
    for f in base.iterdir():
        if f.name not in SKIP and not f.name.startswith("model-"):
            shutil.copy(f, output / f.name)
    shutil.copy(chat_template, output / "chat_template.jinja")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base", type=Path, help="MLX int4 base directory (RecursiveSI/flint-int4)")
    p.add_argument("adapter", type=Path, help="MLX adapter directory (adapters.safetensors, adapter_config.json)")
    p.add_argument("output", type=Path)
    p.add_argument("--chat-template", type=Path, default=Path("release/flint-terse/chat_template.jinja"))
    args = p.parse_args()
    merge(args.base, args.adapter, args.output, args.chat_template)


if __name__ == "__main__":
    main()
