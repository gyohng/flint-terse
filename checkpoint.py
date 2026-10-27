"""Resolve a pinned Hugging Face GGUF, preferring an existing local download."""

import hashlib
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

DEFAULT_MODEL = 'RecursiveSI/flint-int4-gguf:flint-int4-q4_1.gguf'
DEFAULT_REVISION = '9c47151036bc2c34331142dda04488bceab2bc9f'
DEFAULT_SHA256 = '43c07baccdba42eb7528a9630ee15699b1942c297291dc1b39a061f508ee1a4b'


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def resolve_checkpoint(model=DEFAULT_MODEL, revision=None, model_dirs=()):
    local = Path(model).expanduser()
    if local.is_file():
        return local.resolve(), {'filename': local.name, 'sha256': sha256(local)}
    if ':' not in model:
        raise ValueError('Use a local GGUF path or owner/repository:filename.gguf')
    repo, filename = model.split(':', 1)
    if model == DEFAULT_MODEL and revision in (None, DEFAULT_REVISION):
        revision, expected = DEFAULT_REVISION, DEFAULT_SHA256
    else:
        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        entry = next((f for f in info.siblings if f.rfilename == filename), None)
        if entry is None or not entry.lfs:
            raise ValueError(f'No GGUF file {filename} in {repo}')
        revision, expected = info.sha, entry.lfs.sha256
    metadata = {'repo': repo, 'revision': revision, 'filename': filename, 'sha256': expected}
    roots = [Path(p).expanduser() for p in model_dirs] + [Path('models'), Path.home() / '.lmstudio/models',
        Path.home() / '.lmstudio/apps/bionic/models']
    for root in roots:
        for candidate in root.rglob(Path(filename).name):
            if candidate.is_file() and sha256(candidate) == expected:
                print(f'Using existing GGUF: {candidate}', flush=True)
                return candidate.resolve(), metadata
    path = Path(hf_hub_download(repo, filename, revision=revision))
    if sha256(path) != expected:
        raise ValueError(f'GGUF checksum mismatch: {path}')
    return path.resolve(), metadata


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', nargs='?', default=DEFAULT_MODEL,
                        help='Local GGUF or owner/repository:filename.gguf')
    parser.add_argument('--revision', help='Hugging Face revision; the default checkpoint is pinned')
    parser.add_argument('--model-dir', action='append', default=[], help='Additional local model search directory')
    args = parser.parse_args()
    print(resolve_checkpoint(args.model, args.revision, args.model_dir)[0])
