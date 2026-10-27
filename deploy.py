"""Find the unchanged base GGUF and load a saved adapter in LM Studio."""

import argparse
import json
from pathlib import Path

from checkpoint import DEFAULT_MODEL, resolve_checkpoint
from finetune import lmstudio_key, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', nargs='?', default=DEFAULT_MODEL)
    parser.add_argument('--adapter', type=Path, default=Path('release/flint-terse'))
    parser.add_argument('--revision')
    parser.add_argument('--model-dir', action='append', default=[])
    parser.add_argument('--context', type=int, default=65536)
    args = parser.parse_args()
    config = json.loads((args.adapter / 'adapter_config.json').read_text())
    model, source = resolve_checkpoint(args.model, args.revision, args.model_dir)
    if source['sha256'] != config['base_model']['sha256']:
        parser.error('This adapter requires the exact GGUF it was trained against')
    run('lms', 'server', 'start')
    key = lmstudio_key(model)
    if key is None:
        run('lms', 'import', model, '--symbolic-link', '--user-repo', 'local/flint-terse-base', '--yes')
        key = lmstudio_key(model)
    if key is None:
        raise RuntimeError('Imported GGUF is not indexed yet; run deploy again')
    run('node', 'load_lmstudio.mjs', '--model', key, '--adapter', args.adapter,
        '--identifier', 'flint-terse', '--context', args.context)


if __name__ == '__main__':
    main()
