"""Combine attention and native thinking-token adapters for the same GGUF."""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attention', type=Path)
    p.add_argument('opening', type=Path)
    p.add_argument('--scale', type=float, required=True, help='Strength of the opening adapter; attention stays unchanged')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config, opening = [json.loads((path / 'adapter_config.json').read_text()) for path in (args.attention, args.opening)]
    if config['base_model']['sha256'] != opening['base_model']['sha256']:
        p.error('Adapters must use the same unchanged GGUF')
    if (args.scale <= 0 or opening.get('prompt_tokens')
        or opening['lora_parameters']['keys'] != ['model.embed_tokens']
        or 'model.embed_tokens' in config['lora_parameters']['keys']):
        p.error('Use a positive scale, an attention-only adapter and a native thinking-token adapter')
    weights = load_file(args.attention / 'adapters.safetensors')
    embedding = load_file(args.opening / 'adapters.safetensors')
    if any(value.dtype != np.float32 for value in [*weights.values(), *embedding.values()]):
        p.error('Keep both adapters in FP32')
    padding = config['lora_parameters']['rank'] - opening['lora_parameters']['rank']
    if padding < 0:
        p.error('The attention rank must be at least the opening rank')
    factor = np.float32(args.scale * opening['lora_parameters']['scale'] / config['lora_parameters']['scale'])
    weights['model.embed_tokens.lora_a'] = np.pad(embedding['model.embed_tokens.lora_a'], [(0, 0), (0, padding)])
    weights['model.embed_tokens.lora_b'] = np.pad(embedding['model.embed_tokens.lora_b'] * factor, [(0, padding), (0, 0)])
    config['lora_parameters']['keys'].append('model.embed_tokens')
    config.update(output=str(args.output), opening_adapter=str(args.opening), opening_scale=args.scale)
    args.output.mkdir(parents=True)
    save_file(weights, args.output / 'adapters.safetensors')
    (args.output / 'adapter_config.json').write_text(json.dumps(config, indent=2) + '\n')


if __name__ == '__main__':
    main()
