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
    parser.add_argument('--hf-user', default='gyohng', help='Hugging Face account the adapter GGUF is fetched from')
    args = parser.parse_args()
    if not (args.adapter / 'adapter.gguf').exists():
        # Adapter weights live on Hugging Face (<user>/<name>/<name>.gguf), not in git.
        import shutil
        from huggingface_hub import hf_hub_download
        name = args.adapter.name
        print(f'Downloading {args.hf_user}/{name}/{name}.gguf', flush=True)
        shutil.copy(hf_hub_download(f'{args.hf_user}/{name}', f'{name}.gguf'), args.adapter / 'adapter.gguf')
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
