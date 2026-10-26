"""Find a Flint GGUF, train a separate adapter, evaluate it, and load flint-terse."""

import argparse
import gc
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from checkpoint import DEFAULT_MODEL, resolve_checkpoint, sha256


def run(*args):
    subprocess.run([str(a) for a in args], check=True)


def lmstudio_key(model):
    script = """
import { LMStudioClient } from './.tools/lms-sdk/node_modules/@lmstudio/sdk/dist/index.mjs';
const client = new LMStudioClient();
try { console.log(JSON.stringify(await client.system.listDownloadedModels())); }
finally { await client[Symbol.asyncDispose](); }
"""
    listing = json.loads(subprocess.check_output(['node', '--input-type=module', '-e', script]))
    for entry in listing:
        path = Path.home() / '.lmstudio/models' / entry['indexedModelIdentifier'].split('@')[-1]
        if path.is_file() and path.samefile(model):
            return entry['modelKey']
    return None


def main(argv=None, loaded=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', nargs='?', default=DEFAULT_MODEL, help='Local GGUF or owner/repository:filename.gguf')
    p.add_argument('--revision', help='Hugging Face revision; the default checkpoint is pinned')
    p.add_argument('--model-dir', action='append', default=[], help='Additional local model search directory')
    p.add_argument('--data', nargs='+', default=['data/on-policy-pilot.jsonl', 'data/prose-extra-paired.jsonl', 'data/pruned-pilot.jsonl'])
    p.add_argument('--output', type=Path, default=Path('release/flint-terse'))
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--evaluate-only', action='store_true', help='Evaluate and load an already trained output directory')
    p.add_argument('--eval', nargs='+', default=['data/release-eval.jsonl', 'data/direct-eval.jsonl',
                                               'data/retention-eval.jsonl', 'data/release-chat.jsonl'])
    p.add_argument('--api', default='http://localhost:1234/v1')
    p.add_argument('--context', type=int, default=65536)
    args = p.parse_args(argv)
    if args.output.exists() and not args.evaluate_only:
        p.error(f'{args.output} exists; choose a new --output')
    for label in ('base', 'terse'):
        for dataset in args.eval:
            if (Path('runs') / f'{args.output.name}-{label}-{Path(dataset).stem}.jsonl').exists():
                p.error('Evaluation output already exists; choose a new --output name')
    run('lms', 'server', 'start')
    model, source = resolve_checkpoint(args.model, args.revision, args.model_dir)
    if args.evaluate_only:
        config = json.loads((args.output / 'adapter_config.json').read_text())
        if config['base_model']['sha256'] != source['sha256']:
            p.error('The adapter was trained against a different GGUF')
    else:
        from gguf import GGUFReader
        reader = GGUFReader(model)
        if reader.fields['general.architecture'].contents() != 'flint':
            p.error('This training recipe supports Flint checkpoints only')
        del reader
        Path('runs').mkdir(exist_ok=True)
        import cache_features
        import train_prompt
        from gguf_model import load_model
        import mlx.core as mx
        if loaded is None:
            loaded = load_model(str(model))
        with tempfile.TemporaryDirectory(prefix='flint-teacher-', dir='runs') as tmp:
            features = Path(tmp) / 'features.safetensors'
            cache_features.main([*args.data, '--model', str(model), '--original-reasoning',
                '--max-tokens', '16384', '--output', str(features)], loaded=loaded)
            train_prompt.main([*args.data, '--model', str(model), '--teacher-features', str(features),
                '--think-opening', '--answer-kl', '8', '--answer-weight', '1', '--start-weight', '3',
                '--steps', str(args.steps), '--max-tokens', '16384', '--output', str(args.output)], loaded=loaded)
        del loaded
        gc.collect()
        mx.clear_cache()
        config_file = args.output / 'adapter_config.json'
        config = json.loads(config_file.read_text())
        config['base_model'] = source
        config_file.write_text(json.dumps(config, indent=2) + '\n')
        run(sys.executable, 'export_adapter.py', '--adapter', args.output, '--model', model)
    if sha256(model) != source['sha256']:
        raise RuntimeError('Base GGUF changed during training')

    # Symlink into the library if this exact file is not already indexed there.
    key = lmstudio_key(model)
    if key is None:
        run('lms', 'import', model, '--symbolic-link', '--user-repo', 'local/flint-terse-base', '--yes')
        key = lmstudio_key(model)
        if key is None:
            raise RuntimeError('Imported GGUF is not indexed by LM Studio yet; retry with --evaluate-only')
    results = {}
    for label in ('base', 'terse'):
        identifier = 'flint-base-eval' if label == 'base' else 'flint-terse'
        command = ['node', 'load_lmstudio.mjs', '--model', key, '--adapter', args.output,
                   '--identifier', identifier, '--context', args.context]
        run(*command, *(['--base'] if label == 'base' else []))
        results[label] = []
        for dataset in args.eval:
            output = Path('runs') / f'{args.output.name}-{label}-{Path(dataset).stem}.jsonl'
            run(sys.executable, 'corpus.py', 'capture', dataset, '--api', args.api, '--api-chat',
                '--model', identifier, '--tokenizer', model, '--temperature', 1, '--seed', 42,
                '--max-tokens', 4096, '--output', output)
            results[label].extend(json.loads(line) for line in output.read_text().splitlines())
        if label == 'base':
            run('lms', 'unload', identifier)
    base, terse = results['base'], results['terse']
    baseline = sum(r['correct'] is True for r in base)
    adapted = sum(r['correct'] is True for r in terse)
    paired = [(b, t) for b, t in zip(base, terse) if b['correct'] is True and t['correct'] is True]
    before = sum(b['reasoning_tokens'] for b, _ in paired)
    after = sum(t['reasoning_tokens'] for _, t in paired)
    graded = sum(r['correct'] is not None for r in base)
    print(f'Checks: base {baseline}/{graded}, terse {adapted}/{graded}. '
          f'Thinking on shared successes: {before} -> {after} tokens.', flush=True)
    if adapted < baseline:
        raise SystemExit('Quality regression detected. The adapter is experimental; inspect the saved answers.')
    if sum(r['complete'] for r in terse) < sum(r['complete'] for r in base):
        raise SystemExit('Completion regression detected. Inspect the saved answers.')
    if not paired or after >= before:
        raise SystemExit('No thinking reduction on shared correct answers. Inspect the saved traces.')
    print(f'Adapter saved in {args.output}; loaded in LM Studio as flint-terse. '
          'Review the prose captures too; these checks do not establish unchanged quality on every task.')


if __name__ == '__main__':
    main()
