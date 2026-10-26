"""Export the opening-only embedding adapter, without reading or rewriting base weights."""

import argparse
import json
from pathlib import Path

import gguf
import numpy as np
from safetensors.numpy import load_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, default=Path("adapters/opening-pruned"))
    p.add_argument("--model", type=Path, default=Path("models/flint-int4"))
    args = p.parse_args()
    config = json.loads((args.adapter / "adapter_config.json").read_text())
    weights = load_file(args.adapter / "adapters.safetensors")
    prefix = "model.embed_tokens."
    if set(weights) != {prefix + "lora_a", prefix + "lora_b"} or config.get("prompt_tokens"):
        p.error("This exporter supports native thinking-token embedding adapters only")
    a, b = weights[prefix + "lora_a"], weights[prefix + "lora_b"]
    if a.dtype != np.float32 or b.dtype != np.float32:
        p.error("Expected the original FP32 adapter tensors")
    output = args.adapter / "adapter.gguf"
    writer = gguf.GGUFWriter(output, "flint")
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, a.shape[1] * config["lora_parameters"]["scale"])
    writer.add_tensor("token_embd.weight.lora_a", a)
    writer.add_tensor("token_embd.weight.lora_b", np.ascontiguousarray(b.T))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Verify the layout conversion preserves every adapter coefficient.
    exported = {t.name: t.data for t in gguf.GGUFReader(output).tensors}
    assert exported["token_embd.weight.lora_a"].tobytes() == a.tobytes()
    assert exported["token_embd.weight.lora_b"].T.tobytes() == b.tobytes()
    template = (args.model / "chat_template.jinja").read_text()
    override = args.adapter / "chat_template.jinja"
    override.write_text(template.replace(r' | replace("\x00", "")', ""))
    print(f"FP32 adapter: {output} ({output.stat().st_size:,} bytes)")
    print(f"Jinja override: {override}")


if __name__ == "__main__":
    main()
