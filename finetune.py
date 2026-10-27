"""Find a Flint GGUF, train a separate adapter, evaluate it, and load flint-terse."""

import argparse
import gc
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from checkpoint import DEFAULT_MODEL, DEFAULT_SHA256, resolve_checkpoint, sha256


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


def read_baseline(prefix, datasets, source, context):
    settings = dict(base_sha256=source['sha256'], context=context, temperature=1,
                    seed=42, max_tokens=4096, top_p=.95, top_k=20, min_p=0,
                    repeat_penalty=1, api_mode='chat')
    rows = []
    for dataset in datasets:
        path = Path(f'{prefix}-{Path(dataset).stem}.jsonl')
        saved = [json.loads(line) for line in path.read_text().splitlines()]
        tasks = [json.loads(line) for line in Path(dataset).read_text().splitlines()]
        if len(saved) != len(tasks) or any(
            any(row.get(k) != v for k, v in {**task, **settings}.items())
            for row, task in zip(saved, tasks)
        ):
            raise ValueError(f'{path}: baseline tasks, checkpoint or evaluation settings differ')
        rows.extend(saved)
    return rows


def accuracy_can_pass(base, candidate, max_drop):
    for family in [None, *sorted({r['family'] for r in base if r.get('family')})]:
        checked = baseline = possible = 0
        for i, row in enumerate(base):
            if row['correct'] is None or (family is not None and row.get('family') != family):
                continue
            checked += 1
            baseline += row['correct'] is True
            possible += i >= len(candidate) or candidate[i]['correct'] is True
        if (baseline - possible) * 100 > max_drop * checked:
            return False
    return True


def main(argv=None, loaded=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', nargs='?', default=DEFAULT_MODEL, help='Local GGUF or owner/repository:filename.gguf')
    p.add_argument('--revision', help='Hugging Face revision; the default checkpoint is pinned')
    p.add_argument('--model-dir', action='append', default=[], help='Additional local model search directory')
    p.add_argument('--data', nargs='+', default=['data/on-policy-pilot.jsonl', 'data/prose-extra-paired.jsonl', 'data/pruned-pilot.jsonl'])
    p.add_argument('--teacher-features', type=Path, help='Reuse cached probability features for this exact teacher and corpus')
    p.add_argument('--output', type=Path, default=Path('adapters/flint-terse'))
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--method', choices=('opening', 'tokens', 'attention'), default='opening',
                   help='Adapt the opening token, both thinking boundaries, or attention projections')
    p.add_argument('--layers', type=int, help='Attention adapter depth; defaults to 8, or the resumed adapter depth')
    p.add_argument('--train-layers', type=int, help='When resuming an attention adapter, update only the last N layers')
    p.add_argument('--preference-weight', type=float, default=0,
                   help='Attention training: prefer checked correct responses over labeled mistakes')
    p.add_argument('--compact-preference-weight', type=float, default=0,
                   help='Attention training: prefer reviewed short notes over original notes with the same final reply')
    p.add_argument('--answer-only', action='store_true',
                   help='Attention training: repair final answers without imitating the sampled reasoning')
    p.add_argument('--adapter-scale', type=float,
                   help='Multiply the learned update at export without changing its FP32 matrices')
    start = p.add_mutually_exclusive_group()
    start.add_argument('--initial-adapter', help='Initialize from a compatible native thinking-token adapter')
    start.add_argument('--resume-adapter', help='Continue an attention adapter at its saved scale')
    start.add_argument('--from-scratch', action='store_true', help='Skip the packaged opening-token initializer')
    p.add_argument('--attention-adapter', help='Train the opening token alongside this frozen attention adapter')
    p.add_argument('--same-prefix-teacher', action='store_true',
                   help='When resuming attention, retain probabilities at the compact prefixes')
    p.add_argument('--teacher-adapter', help='Fixed attention adapter for answer-probability references')
    p.add_argument('--narration-penalty', type=float, default=0,
                   help='Attention training: discourage stock narration at the start of supervised thinking')
    p.add_argument('--entry-only', action='store_true',
                   help='Attention training: supervise the first thinking token instead of the full notes')
    p.add_argument('--entry-weight', type=float, default=0,
                   help='Attention training: emphasize the first token of thinking and of the final reply')
    p.add_argument('--retention-kl', type=float, default=8, help='KL weight for math/code policy retention')
    p.add_argument('--learning-rate', type=float, help='Defaults to .0001 for the packaged recipe, .001 from scratch, .000005 for attention')
    p.add_argument('--max-change', type=float,
                   help='Limit the embedding update norm relative to the original embedding')
    p.add_argument('--max-accuracy-drop', type=float, default=2,
                   help='Allowed accuracy drop overall and per task family, in percentage points (default: 2)')
    p.add_argument('--evaluate-only', action='store_true', help='Evaluate and load an already trained output directory')
    p.add_argument('--resume-eval', action='store_true', help='Continue an interrupted evaluation without repeating saved tasks')
    p.add_argument('--no-promote', action='store_true', help='Leave the candidate as flint-terse-eval for further checks')
    p.add_argument('--baseline', type=Path, help='Reuse saved baseline prefix; matching runs/flint-terse-base is reused automatically')
    p.add_argument('--eval', nargs='+', default=['data/release-eval.jsonl', 'data/confirmation-eval.jsonl', 'data/direct-eval.jsonl',
                                               'data/retention-eval.jsonl', 'data/release-chat.jsonl'])
    p.add_argument('--api', default='http://localhost:1234/v1')
    p.add_argument('--context', type=int, default=65536)
    args = p.parse_args(argv)
    if args.same_prefix_teacher and (not args.resume_adapter or args.compact_preference_weight):
        p.error('--same-prefix-teacher requires --resume-adapter without compact preferences')
    if args.teacher_adapter and not args.resume_adapter:
        p.error('--teacher-adapter requires --resume-adapter')
    if args.retention_kl < 0:
        p.error('--retention-kl must be nonnegative')
    if args.narration_penalty < 0 or (args.narration_penalty and (args.method != 'attention' or args.answer_only)):
        p.error('--narration-penalty requires attention reasoning supervision and a nonnegative weight')
    if args.entry_only and (args.method != 'attention' or args.answer_only or args.narration_penalty):
        p.error('--entry-only requires attention training without other thinking-loss overrides')
    if args.entry_weight < 0 or (args.entry_weight and (args.method != 'attention' or args.entry_only or args.answer_only)):
        p.error('--entry-weight requires attention reasoning supervision and a nonnegative weight')
    if args.attention_adapter and (args.method != 'opening' or args.initial_adapter or args.resume_adapter):
        p.error('--attention-adapter requires --method opening without another initializer')
    if args.resume_adapter and args.method != 'attention':
        p.error('--resume-adapter requires --method attention')
    if args.train_layers is not None and not args.resume_adapter:
        p.error('--train-layers requires --resume-adapter')
    if args.layers is None:
        args.layers = json.loads((Path(args.resume_adapter) / 'adapter_config.json').read_text())['num_layers'] if args.resume_adapter else 8
    if min(args.preference_weight, args.compact_preference_weight) < 0 or (
        (args.preference_weight or args.compact_preference_weight) and args.method != 'attention'):
        p.error('Preference weights must be nonnegative and require --method attention')
    if args.answer_only and (args.method != 'attention' or args.preference_weight or args.compact_preference_weight):
        p.error('--answer-only requires --method attention without preferences')
    if not args.evaluate_only and args.method == 'attention' and args.max_change not in (None, 1):
        p.error('--max-change applies only to opening training')
    if args.max_change is not None and args.max_change <= 0:
        p.error('--max-change must be positive')
    if not 0 <= args.max_accuracy_drop <= 100:
        p.error('--max-accuracy-drop must be between 0 and 100')
    if args.adapter_scale is not None and (args.adapter_scale <= 0 or (args.evaluate_only and args.adapter_scale != 1)):
        p.error('--adapter-scale must be positive and applies only when training; evaluation uses the saved scale')
    if args.resume_eval and not args.evaluate_only:
        p.error('--resume-eval requires --evaluate-only')
    if args.output.exists() and not args.evaluate_only:
        p.error(f'{args.output} exists; choose a new --output')
    if not args.evaluate_only and not args.answer_only:
        from corpus import read_rows
        if any(row.get('answer_only') for path in args.data for row in read_rows(path)):
            p.error('This corpus contains sampled notes; use --method attention --answer-only')
    run('lms', 'server', 'start')
    model, source = resolve_checkpoint(args.model, args.revision, args.model_dir)
    packaged = (not args.evaluate_only and args.method == 'opening' and not args.from_scratch
                and args.initial_adapter is None and not args.attention_adapter and source['sha256'] == DEFAULT_SHA256)
    if packaged:
        args.initial_adapter = 'data/opening-init'
    if args.learning_rate is None:
        args.learning_rate = .0001 if packaged else .000005 if args.method == 'attention' else .001
    if args.max_change is None:
        args.max_change = .1 if packaged else 1
    if args.adapter_scale is None:
        args.adapter_scale = .9 if packaged else 1
    results = {}
    prefix = args.baseline or Path('runs/flint-terse-base')
    try:
        results['base'] = read_baseline(prefix, args.eval, source, args.context)
    except (FileNotFoundError, ValueError):
        if args.baseline:
            raise
        print('No matching saved baseline; evaluating the unchanged base.', flush=True)
    else:
        args.baseline = prefix
    for label in (('terse',) if args.baseline else ('base', 'terse')):
        for dataset in args.eval:
            if not args.resume_eval and (Path('runs') / f'{args.output.name}-{label}-{Path(dataset).stem}.jsonl').exists():
                p.error('Evaluation output already exists; choose a new --output name')
    if args.evaluate_only:
        config = json.loads((args.output / 'adapter_config.json').read_text())
        if config['base_model']['sha256'] != source['sha256']:
            p.error('The adapter was trained against a different GGUF')
    else:
        for entry in json.loads(subprocess.check_output(['lms', 'ps', '--json'])):
            if entry['identifier'] in {'flint-terse', 'flint-terse-eval', 'flint-base-eval'}:
                run('lms', 'unload', entry['identifier'])
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
            initial = args.attention_adapter or args.resume_adapter
            for initial in filter(None, (initial, args.teacher_adapter)):
                previous = json.loads((Path(initial) / 'adapter_config.json').read_text())
                if previous['base_model']['sha256'] != source['sha256']:
                    p.error('The attention adapter must use this exact GGUF')
            loaded = load_model(str(model), adapter_path=args.attention_adapter)
        with tempfile.TemporaryDirectory(prefix='flint-teacher-', dir='runs') as tmp:
            features = args.teacher_features or Path(tmp) / 'features.safetensors'
            if not args.teacher_features:
                if args.same_prefix_teacher or args.teacher_adapter:
                    from mlx_lm.tuner.utils import load_adapters, remove_lora_layers
                    load_adapters(loaded[0], args.teacher_adapter or args.resume_adapter)
                teacher_adapter = args.teacher_adapter or (args.resume_adapter if args.same_prefix_teacher else args.attention_adapter)
                cache_features.main([*args.data, '--model', str(model),
                    *(['--adapter', teacher_adapter] if teacher_adapter else []),
                    *([] if args.same_prefix_teacher or args.attention_adapter else ['--original-reasoning']),
                    '--max-tokens', '16384', '--output', str(features)], loaded=loaded)
                if args.same_prefix_teacher or args.teacher_adapter:
                    remove_lora_layers(loaded[0])
            if args.method == 'attention':
                import train
                train.main(['--paired', *args.data, '--model', str(model), '--teacher-features', str(features),
                    *(['--think-adapter', args.initial_adapter] if args.initial_adapter else []),
                    *(['--resume', args.resume_adapter] if args.resume_adapter else []),
                    *(['--same-prefix-teacher'] if args.same_prefix_teacher else []),
                    *(['--teacher-adapter', args.teacher_adapter] if args.teacher_adapter else []),
                    '--narration-penalty', str(args.narration_penalty),
                    *(['--entry-only'] if args.entry_only else []),
                    '--entry-weight', str(args.entry_weight),
                    '--retention-kl', str(args.retention_kl),
                    *(['--train-layers', str(args.train_layers)] if args.train_layers is not None else []),
                    '--kl', '8', '--answer-weight', '1', '--start-weight', '3',
                    '--preference-weight', str(args.preference_weight),
                    '--compact-preference-weight', str(args.compact_preference_weight),
                    *(['--answer-only'] if args.answer_only else []),
                    '--layers', str(args.layers), '--rank', '8', '--iters', str(args.steps * 4),
                    '--learning-rate', str(args.learning_rate), '--max-tokens', '16384',
                    '--output', str(args.output)], loaded=loaded)
            else:
                train_prompt.main([*args.data, '--model', str(model), '--teacher-features', str(features),
                    '--think-tokens' if args.method == 'tokens' else '--think-opening',
                    '--answer-kl', '8', '--answer-weight', '1', '--start-weight', '3',
                    '--steps', str(args.steps), '--max-change', str(args.max_change),
                    '--learning-rate', str(args.learning_rate),
                    '--max-tokens', '16384', '--output', str(args.output),
                    *(['--attention-adapter', args.attention_adapter] if args.attention_adapter else []),
                    *(['--initial-adapter', args.initial_adapter] if args.initial_adapter else [])], loaded=loaded)
        del loaded
        gc.collect()
        mx.clear_cache()
        config_file = args.output / 'adapter_config.json'
        config = json.loads(config_file.read_text())
        config['base_model'] = source
        config['adapter_scale'] = args.adapter_scale
        config['lora_parameters']['scale'] *= args.adapter_scale
        config_file.write_text(json.dumps(config, indent=2) + '\n')
        run(sys.executable, 'export_adapter.py', '--adapter', args.output, '--model', model)
    if not args.evaluate_only and sha256(model) != source['sha256']:
        raise RuntimeError('Base GGUF changed during training')

    # Symlink into the library if this exact file is not already indexed there.
    key = lmstudio_key(model)
    if key is None:
        run('lms', 'import', model, '--symbolic-link', '--user-repo', 'local/flint-terse-base', '--yes')
        key = lmstudio_key(model)
        if key is None:
            raise RuntimeError('Imported GGUF is not indexed by LM Studio yet; retry with --evaluate-only')
    if args.baseline:
        print(f'Reusing baseline: {args.baseline}', flush=True)
    for label in (('terse',) if args.baseline else ('base', 'terse')):
        identifier = 'flint-base-eval' if label == 'base' else 'flint-terse-eval'
        command = ['node', 'load_lmstudio.mjs', '--model', key, '--adapter', args.output,
                   '--identifier', identifier, '--context', args.context]
        run(*command, *(['--base'] if label == 'base' else []))
        results[label] = []
        for dataset in args.eval:
            output = Path('runs') / f'{args.output.name}-{label}-{Path(dataset).stem}.jsonl'
            run(sys.executable, 'corpus.py', 'capture', dataset, '--api', args.api, '--api-chat',
                '--model', identifier, '--tokenizer', model, '--temperature', 1, '--seed', 42,
                '--max-tokens', 4096, '--output', output, *(['--resume'] if args.resume_eval else []))
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            for row in rows:
                row.update(base_sha256=source['sha256'], context=args.context)
            output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
            results[label].extend(rows)
            if label == 'terse' and not accuracy_can_pass(results['base'], results[label], args.max_accuracy_drop):
                raise SystemExit(f'{dataset}: remaining tasks cannot keep overall and per-family accuracy within '
                                 f'{args.max_accuracy_drop:g} points; stopping evaluation.')
        if label == 'base':
            run('lms', 'unload', identifier)
    base, terse = results['base'], results['terse']
    if any(tag in (r['assistant'].get('content') or '') for r in terse for tag in ('<think>', '</think>')):
        raise SystemExit('Thinking delimiters leaked into a final reply. Inspect the saved answers.')
    baseline = sum(r['correct'] is True for r in base)
    adapted = sum(r['correct'] is True for r in terse)
    paired = [(b, t) for b, t in zip(base, terse) if b['correct'] is True and t['correct'] is True]
    before = sum(b['reasoning_tokens'] for b, _ in paired)
    after = sum(t['reasoning_tokens'] for _, t in paired)
    graded = sum(r['correct'] is not None for r in base)
    print(f'Checks: base {baseline}/{graded}, terse {adapted}/{graded}. '
          f'Thinking on shared successes: {before} -> {after} tokens.', flush=True)
    for family in sorted({r['family'] for r in base if r.get('family') and r['correct'] is not None}):
        tasks = [(b, t) for b, t in zip(base, terse) if b.get('family') == family and b['correct'] is not None]
        print(f'{family}: base {sum(b["correct"] is True for b, _ in tasks)}/{len(tasks)}, '
              f'terse {sum(t["correct"] is True for _, t in tasks)}/{len(tasks)}.', flush=True)
    if not accuracy_can_pass(base, terse, args.max_accuracy_drop):
        raise SystemExit('Accuracy drop exceeds the allowed margin. Inspect the saved answers.')
    if sum(r['complete'] for r in terse) < sum(r['complete'] for r in base):
        raise SystemExit('Completion regression detected. Inspect the saved answers.')
    prose = [(b, t) for b, t in zip(base, terse) if b['correct'] is None]
    if sum(t['complete'] for _, t in prose) < sum(b['complete'] for b, _ in prose):
        raise SystemExit('Ordinary-reply completion regression detected. Inspect the saved answers.')
    if not paired or after * 100 >= before * 30:
        raise SystemExit('Thinking reduction must exceed 70%. This is not a terse release; inspect the traces.')
    ordinary = [(b, t) for b, t in prose if b['complete'] and t['complete']]
    before = sum(b['reasoning_tokens'] for b, _ in ordinary)
    after = sum(t['reasoning_tokens'] for _, t in ordinary)
    if ordinary:
        print(f'Ordinary completed replies: thinking {before} -> {after} tokens.', flush=True)
        if before and after * 100 >= before * 30:
            raise SystemExit('Thinking reduction on ordinary replies must also exceed 70%; inspect the traces.')
    if args.no_promote:
        print(f'Automatic checks passed for {args.output}; candidate remains flint-terse-eval for further checks.')
        return
    run('lms', 'unload', 'flint-terse-eval')
    run('node', 'load_lmstudio.mjs', '--model', key, '--adapter', args.output,
        '--identifier', 'flint-terse', '--context', args.context)
    print(f'Adapter saved in {args.output}; loaded in LM Studio as flint-terse. '
          'Review the prose captures too; these checks do not establish unchanged quality on every task.')


if __name__ == '__main__':
    main()
