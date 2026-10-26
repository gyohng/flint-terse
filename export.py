"""Merge a Flint adapter and export GGUF for deployment, or an MLX checkpoint."""

import argparse
import gc
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load
from mlx_lm.utils import hf_repo_to_path, save

from corpus import BASE, configure_adapter_tokenizer


CONVERT_GGUF = """
import json, re, sys
sys.path.insert(0, sys.argv.pop(1))
preserve = set(json.loads(sys.argv.pop(1)))
type_file = sys.argv.pop(1)
preserved_names = set()
import convert_hf_to_gguf as converter
from conversion.flint import FlintModel
import gguf
original = FlintModel.tensor_force_quant
def tensor_type(self, name, new_name, bid, n_dims):
    if name in preserve:
        preserved_names.add(new_name)
        return gguf.GGMLQuantizationType.F16
    return original(self, name, new_name, bid, n_dims)
FlintModel.tensor_force_quant = tensor_type
converter.main()
with open(type_file, "w") as f:
    for name in sorted(preserved_names):
        print("^" + re.escape(name) + "$=f16", file=f)
"""


def hf_weights(model):
    """Dequantize one projection at a time and restore HF's individual experts."""
    modules = dict(model.named_modules())
    weights = dict(tree_flatten(model.parameters()))
    for name, value in weights.items():
        if name.endswith((".scales", ".biases")):
            continue
        if name.endswith(".weight"):
            module = modules[name.removesuffix(".weight")]
            if hasattr(module, "bits"):
                value = mx.dequantize(value, module.scales, module.biases,
                                      group_size=module.group_size, bits=module.bits, mode=module.mode)
        if ".switch_mlp." in name:
            for expert in range(value.shape[0]):
                yield name.replace(".switch_mlp.", f".experts.{expert}."), value[expert]
        else:
            yield name, value


def write_hf(model, config, source, output, shard_bytes=2_000_000_000, template=None):
    config = dict(config)
    for key in ("quantization", "quantization_config", "mlx_export"):
        config.pop(key, None)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    source = Path(source)
    if not source.is_dir():
        source = hf_repo_to_path(str(source))
    # Transformers 5 rewrites the tokenizer class to TokenizersBackend, which
    # the converter's Transformers 4 cannot load. Preserve the original files.
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "chat_template.jinja", "generation_config.json", "LICENSE"):
        path = source / name
        if path.exists():
            shutil.copyfile(path, output / name)
    if template is not None:
        tokenizer_config = output / "tokenizer_config.json"
        tokenizer_data = json.loads(tokenizer_config.read_text())
        tokenizer_data["chat_template"] = template
        tokenizer_config.write_text(json.dumps(tokenizer_data, ensure_ascii=False) + "\n")
        (output / "chat_template.jinja").write_text(template)

    shard, size, total, weight_map = {}, 0, 0, {}
    number = 0

    def flush():
        nonlocal number
        number += 1
        filename = f"model-{number:05d}.safetensors"
        mx.save_safetensors(str(output / filename), shard, metadata={"format": "pt"})
        weight_map.update({name: filename for name in shard})
        print(f"Wrote HF shard {number} ({sum(v.nbytes for v in shard.values()) / 1e9:.2f} GB)", flush=True)
        shard.clear()
        mx.clear_cache()

    for name, value in hf_weights(model):
        if shard and size + value.nbytes > shard_bytes:
            flush()
            size = 0
        shard[name] = value
        size += value.nbytes
        total += value.nbytes
    if shard:
        flush()
    (output / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total}, "weight_map": weight_map,
    }) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=BASE)
    p.add_argument("--adapter", default="adapters/terse")
    p.add_argument("--format", choices=("gguf", "mlx"), default="gguf")
    p.add_argument("--outtype", choices=("q4_k_m", "q8_0", "f16"), default="q8_0")
    p.add_argument("--output")
    p.add_argument("--install", action="store_true", help="Symlink the model into LM Studio's library")
    args = p.parse_args()
    default_output = f"models/flint-terse-{args.outtype.upper()}.gguf" if args.format == "gguf" else "models/flint-terse"
    output = Path(args.output or default_output).resolve()
    if output.exists():
        p.error(f"{output} exists; choose a new --output")
    converter = Path(".tools/llama.cpp/convert_hf_to_gguf.py").resolve()
    converter_python = Path(".tools/gguf-venv/bin/python").absolute()
    quantizer = converter.parent / "build/bin/llama-quantize"
    if args.format == "gguf":
        if output.suffix != ".gguf":
            p.error("GGUF output must end in .gguf")
        if not converter.exists() or not converter_python.exists():
            p.error("Run ./run.sh setup-gguf first")
        if args.outtype == "q4_k_m" and not quantizer.exists():
            p.error("Run ./run.sh setup-gguf to build llama-quantize first")
    model, tokenizer, config = load(args.model, adapter_path=args.adapter, return_config=True)
    template = configure_adapter_tokenizer(tokenizer, args.adapter)
    # Only adapted modules become BF16/FP16. Experts stay INT4.
    fused = [(name, module.fuse(dequantize=True)) for name, module in model.named_modules()
             if hasattr(module, "fuse")]
    if not fused:
        p.error("No adapter layers found")
    preserve = json.dumps([name + ".weight" for name, _ in fused])
    model.update_modules(tree_unflatten(fused))
    if args.format == "mlx":
        quant = dict(config["quantization"])
        for name, _ in fused:
            quant[name] = False
        config["quantization"] = quant
        save(output, args.model, model, tokenizer, config, donate_model=False)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        # HF and converter scratch files are temporary, not extra model copies to retain.
        with tempfile.TemporaryDirectory(prefix=".flint-gguf-", dir=output.parent) as tmp:
            tmp = Path(tmp)
            convert_type = "f16" if args.outtype == "q4_k_m" else args.outtype
            converted = tmp / ("f16.gguf" if args.outtype == "q4_k_m" else output.name)
            tensor_types = tmp / "preserve-f16.txt"
            write_hf(model, config, args.model, tmp / "hf", template=template)
            del model, fused
            gc.collect()
            mx.clear_cache()
            subprocess.run([
                str(converter_python), str(converter), str(tmp / "hf"),
                "--vocab-only", "--outfile", str(tmp / "vocab.gguf"), "--outtype", convert_type,
            ], check=True)
            subprocess.run([
                str(converter_python), "-c", CONVERT_GGUF, str(converter.parent), preserve,
                str(tensor_types), str(tmp / "hf"), "--outfile", str(converted),
                "--model-name", output.stem, "--outtype", convert_type, "--use-temp-file",
            ], check=True)
            if args.outtype == "q4_k_m":
                subprocess.run([
                    str(quantizer), "--tensor-type-file", str(tensor_types),
                    str(converted), str(tmp / output.name), "Q4_K_M",
                ], check=True)
            (tmp / output.name).rename(output)
    print(f"Exported model: {output}")
    if args.install:
        if args.format == "gguf":
            subprocess.run(["lms", "import", str(output), "--symbolic-link",
                            "--user-repo", f"local/{output.stem}", "--yes"], check=True)
        else:
            target = Path.home() / ".lmstudio/models/local" / output.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(output, target_is_directory=True)
            print(f"LM Studio library: {target} (requires MLX runtime support)")


if __name__ == "__main__":
    main()
