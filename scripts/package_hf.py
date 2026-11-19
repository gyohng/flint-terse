"""Package a trained adapter for Hugging Face: GGUF LoRA (llama.cpp / LM Studio) and MLX LoRA (mlx-lm).

  python3 scripts/package_hf.py adapters/spark-final flint-terse out/flint-terse

MLX conversion: same key names (mlx-lm module paths), lora_a [in, r] / lora_b [r, out]; per-expert factors
become mlp.switch_mlp.down_proj with [E, r, in] / [E, out, r], zero-padded to the adapter's main rank because
mlx-lm uses one rank for every module (exact), stored as float16.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        "mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj"]


def mlx_adapter(src, dst):
    cfg = json.loads((src / "adapter_config.json").read_text())
    rank, scale = cfg["lora_parameters"]["rank"], cfg["lora_parameters"]["scale"]
    weights = load_file(src / "adapters.safetensors")
    out, keys = {}, list(KEYS)
    for name, value in weights.items():
        if ".mlp.experts." not in name:
            out[name] = value
            continue
        prefix, leaf = name.replace(".mlp.experts.", ".mlp.switch_mlp.").rsplit(".", 1)
        value = np.swapaxes(value, 1, 2)                  # [E, in, r] -> [E, r, in]; [E, r, out] -> [E, out, r]
        r = value.shape[1] if leaf == "lora_a" else value.shape[2]
        pad = [(0, 0), (0, rank - r), (0, 0)] if leaf == "lora_a" else [(0, 0), (0, 0), (0, rank - r)]
        # Expert factors already carry their own scale; undo the global one mlx-lm multiplies in.
        if leaf == "lora_b":
            value = value / scale
        out[f"{prefix}.{leaf}"] = np.pad(value, pad).astype(np.float16)
        if "mlp.switch_mlp.down_proj" not in keys:
            keys.append("mlp.switch_mlp.down_proj")
    dst.mkdir(parents=True, exist_ok=True)
    save_file(out, dst / "adapters.safetensors")
    mlx_cfg = {"fine_tune_type": "lora", "num_layers": cfg["num_layers"],
               "lora_parameters": {"rank": rank, "scale": scale, "dropout": 0.0, "keys": keys}}
    (dst / "adapter_config.json").write_text(json.dumps(mlx_cfg, indent=2) + "\n")
    return keys


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("adapter", type=Path, help="Trained adapter directory (adapters.safetensors, adapter.gguf)")
    p.add_argument("name")
    p.add_argument("output", type=Path)
    p.add_argument("--readme", type=Path, help="Model card to copy as README.md")
    args = p.parse_args()
    out = args.output
    if out.exists():
        p.error(f"{out} exists")
    out.mkdir(parents=True)
    shutil.copy(args.adapter / "adapter.gguf", out / f"{args.name}.gguf")
    shutil.copy(args.adapter / "chat_template.jinja", out / "chat_template.jinja")
    shutil.copy("models/flint/LICENSE", out / "LICENSE")
    keys = mlx_adapter(args.adapter, out / "mlx")
    if args.readme:
        shutil.copy(args.readme, out / "README.md")
    print(f"{out}: {args.name}.gguf, mlx/ ({len(keys)} module types), chat_template.jinja, LICENSE")


if __name__ == "__main__":
    main()
