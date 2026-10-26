"""Resolve a pinned Hugging Face GGUF, preferring an existing local download."""

import hashlib
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

DEFAULT_MODEL = 'lmstudio-community/flint-GGUF:flint-Q4_K_M.gguf'
DEFAULT_REVISION = 'a14b4396552988f5bf1427c2ab23f3ade4b683a8'
DEFAULT_SHA256 = '20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e'


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
