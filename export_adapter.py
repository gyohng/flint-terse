"""Export FP32 Flint LoRA tensors without rewriting the base weights."""

import argparse
import json
from pathlib import Path

import gguf
import numpy as np
from safetensors.numpy import load_file


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, default=Path("adapters/opening-pruned"))
    p.add_argument("--model", type=Path, default=Path("models/flint-int4"))
    args = p.parse_args(argv)
    config = json.loads((args.adapter / "adapter_config.json").read_text())
    weights = load_file(args.adapter / "adapters.safetensors")
    if config.get("prompt_tokens") or any(not name.endswith(('.lora_a', '.lora_b')) for name in weights):
        p.error("Expected native embedding or projection LoRA tensors")
    if args.model.suffix.lower() == '.gguf':
        reader = gguf.GGUFReader(args.model)
        blocks = reader.fields['flint.block_count'].contents()
        template = reader.fields['tokenizer.chat_template'].contents()
    else:
        blocks = json.loads((args.model / 'config.json').read_text())['num_hidden_layers']
        template = (args.model / "chat_template.jinja").read_text()
    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.FLINT, blocks)
    rank = config['lora_parameters']['rank']
    output = args.adapter / "adapter.gguf"
    writer = gguf.GGUFWriter(output, "flint")
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, rank * config["lora_parameters"]["scale"])
    expected = {}
    for prefix in sorted({name.rsplit('.', 1)[0] for name in weights}):
        a, b = (weights[prefix + '.' + key] for key in ('lora_a', 'lora_b'))
        if a.dtype != np.float32 or b.dtype != np.float32 or a.ndim != 2 or b.ndim != 2:
            p.error(f'{prefix}: expected the original FP32 matrices')
        if a.shape[1] != rank or b.shape[0] != rank:
            p.error(f'{prefix}: adapter rank differs from its configuration')
        mapped = names.get_name(prefix + '.weight', try_suffixes=('.weight',))
        if mapped is None:
            p.error(f'Unsupported adapter projection: {prefix}')
        expected[mapped + '.lora_a'] = np.ascontiguousarray(a if prefix == 'model.embed_tokens' else a.T)
        expected[mapped + '.lora_b'] = np.ascontiguousarray(b.T)
    for name, value in expected.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Verify the layout conversion preserves every adapter coefficient.
    exported = {t.name: t.data for t in gguf.GGUFReader(output).tensors}
    assert set(exported) == set(expected)
    assert all(exported[name].tobytes() == value.tobytes() for name, value in expected.items())
    override = args.adapter / "chat_template.jinja"
    override.write_text(template.replace(r' | replace("\x00", "")', ""))
    print(f"FP32 adapter: {output} ({output.stat().st_size:,} bytes)")
    print(f"Jinja override: {override}")


if __name__ == "__main__":
    main()
