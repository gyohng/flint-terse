"""Package a trained adapter for Hugging Face as three repositories: NAME (the adapter merged into the
int4 GGUF, for LM Studio), NAME-lora (GGUF LoRA for llama.cpp) and NAME-mlx (MLX LoRA for mlx-lm), each
with Flint's licence and a model card filled from scripts/model_card_{merged,gguf,mlx}.md.

  llama-export-lora -m flint-int4-q4_1.gguf --lora adapters/spark-final/adapter.gguf -o merged.gguf
  python3 gguf-py/gguf/scripts/gguf_new_metadata.py merged.gguf flint-terse.gguf \
      --general-name "Flint Terse" --chat-template-file release/flint-terse/chat_template.jinja
  python3 scripts/package_hf.py adapters/spark-final flint-terse out --title "Flint Terse" --merged flint-terse.gguf

MLX conversion: same key names (mlx-lm module paths), lora_a [in, r] / lora_b [r, out]; per-expert factors
become mlp.switch_mlp.down_proj with [E, r, in] / [E, out, r], zero-padded to the adapter's main rank because
mlx-lm uses one rank for every module (exact), stored as float16.
"""

import argparse
import json
import os
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


def fill(template, values):
    text = Path(template).read_text()
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


def size(path):
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) if Path(path).is_dir() else Path(path).stat().st_size
    return f"{round(total / 1e6)} MB"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("adapter", type=Path, help="Trained adapter directory (adapters.safetensors, adapter.gguf)")
    p.add_argument("name")
    p.add_argument("output", type=Path, help="Parent directory for NAME/ and NAME-mlx/")
    p.add_argument("--title", required=True)
    p.add_argument("--user", default="gyohng")
    p.add_argument("--summary", default="", help="Markdown paragraph for both cards")
    p.add_argument("--gguf-note", default="", help="Extra usage note for the GGUF card")
    p.add_argument("--mlx-note", default="", help="Extra usage note for the MLX card")
    p.add_argument("--merged", type=Path, required=True, help="Merged GGUF (see above)")
    p.add_argument("--merged-note", default="The LoRA-adapted tensors are stored as F16.",
                   help="Sentence on which tensors the merge rewrote")
    p.add_argument("--expert-clause", default="", help="Appended to the card's LoRA description")
    args = p.parse_args()
    merged, gguf, mlx = (args.output / f"{args.name}{s}" for s in ("", "-lora", "-mlx"))
    for d in (merged, gguf, mlx):
        if d.exists():
            p.error(f"{d} exists")
        d.mkdir(parents=True)
        shutil.copy("models/flint/LICENSE", d / "LICENSE")
    shutil.copy(args.adapter / "adapter.gguf", gguf / f"{args.name}.gguf")
    shutil.copy(args.adapter / "chat_template.jinja", gguf / "chat_template.jinja")
    try:
        os.link(args.merged, merged / f"{args.name}.gguf")   # tens of GB: hard-link when on the same disk
    except OSError:
        shutil.copy(args.merged, merged / f"{args.name}.gguf")
    mlx_adapter(args.adapter, mlx)
    values = dict(TITLE=args.title, NAME=args.name, USER=args.user, SUMMARY=args.summary,
                  GGUF_SIZE=size(gguf / f"{args.name}.gguf"), MLX_SIZE=size(mlx / "adapters.safetensors"),
                  MERGED_SIZE=f"{args.merged.stat().st_size / 1e9:.1f} GB",
                  MERGED_NOTE=args.merged_note, EXPERT_CLAUSE=args.expert_clause)
    (merged / "README.md").write_text(fill("scripts/model_card_merged.md", values))
    (gguf / "README.md").write_text(fill("scripts/model_card_gguf.md", values | {"EXTRA": args.gguf_note}))
    (mlx / "README.md").write_text(fill("scripts/model_card_mlx.md", values | {"EXTRA": args.mlx_note}))
    print(f"{merged} ({values['MERGED_SIZE']}), {gguf} ({values['GGUF_SIZE']}), {mlx} ({values['MLX_SIZE']})")


if __name__ == "__main__":
    main()
