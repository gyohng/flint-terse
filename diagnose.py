"""Inspect same-prefix losses and thinking-start probabilities without generating replies."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from corpus import prompt_text, read_rows, training_tokens
from gguf_model import load_model
from train import completion_hidden


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--adapter', required=True)
    p.add_argument('--teacher-features', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--probes', required=True)
    args = p.parse_args()
    features, meta = mx.load(args.teacher_features, return_metadata=True)
    if meta['model'] != args.model:
        p.error('Use the same model as the cached teacher')
    model, tokenizer = load_model(args.model, adapter_path=args.adapter)
    model.eval()
    closing_id = tokenizer.convert_tokens_to_ids('</think>')
    cursor, groups = 0, {}
    for row in read_rows(args.data):
        ids, offset = training_tokens(row, tokenizer)
        target = mx.array([ids[offset:]])
        count = target.shape[1]
        if features['targets'][cursor:cursor + count].tolist() != ids[offset:]:
            p.error(f"{row['id']}: cached teacher tokens differ")
        reference_hidden = features['hidden'][None, cursor:cursor + count]
        cursor += count
        if row['split'] != 'valid':
            continue
        logits = model.lm_head(completion_hidden(model, ids, offset)).astype(mx.float32)
        reference = model.lm_head(reference_hidden).astype(mx.float32)
        q, r = nn.log_softmax(logits, axis=-1), nn.log_softmax(reference, axis=-1)
        kl = (mx.exp(r) * (r - q)).sum(axis=-1)
        ce, old_ce = [nn.losses.cross_entropy(x, target) for x in (logits, reference)]
        close = ids[offset:].index(closing_id)
        values = {'policy_kl': kl.mean(), 'answer_kl': kl[:, close + 1:].mean()}
        for name, a, b in [('start_ce', 1, min(9, close)), ('thought_ce', 1, close),
                           ('close_ce', close, close + 1), ('answer_ce', close + 1, count - 1)]:
            if b > a:
                values[name] = ce[:, a:b].mean()
                values['old_' + name] = old_ce[:, a:b].mean()
        result = {k: v.item() for k, v in values.items()}
        group = 'retention' if row.get('preserve_only') else 'style'
        groups.setdefault(group, []).append(result)
        print(json.dumps({'id': row['id'], 'group': group, **result}), flush=True)
    for group, rows in groups.items():
        print(group, {k: round(sum(r[k] for r in rows if k in r) / sum(k in r for r in rows), 4)
                      for k in rows[0]}, flush=True)
    starting = meta['adapter']
    for label, adapter in [('trained', args.adapter), ('starting', starting)]:
        config = json.loads((Path(adapter) / 'adapter_config.json').read_text())
        current = json.loads((Path(args.adapter) / 'adapter_config.json').read_text())
        if config['lora_parameters'] != current['lora_parameters']:
            p.error('Thinking-start probes require matching adapter layout and scale')
        model.load_weights(str(Path(adapter) / 'adapters.safetensors'), strict=False)
        for row in read_rows(args.probes):
            ids = tokenizer.encode(prompt_text(row, tokenizer) + '<think>', add_special_tokens=False)
            probabilities = mx.softmax(model(mx.array([ids]))[0, -1].astype(mx.float32))
            best = mx.argsort(probabilities)[-5:][::-1].tolist()
            print(json.dumps({'probe': row['id'], 'adapter': label,
                'top': [(tokenizer.decode([i]), round(probabilities[i].item(), 4)) for i in best]}), flush=True)


if __name__ == '__main__':
    main()
