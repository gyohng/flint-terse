#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-help}" in
  setup)
    uv venv --python 3.13 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
    ;;
  setup-gguf)
    mkdir -p .tools/llama.cpp
    git -C .tools/llama.cpp init -q
    git -C .tools/llama.cpp fetch --depth 1 https://github.com/ggml-org/llama.cpp.git b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9
    git -C .tools/llama.cpp checkout --detach FETCH_HEAD
    uv venv --python 3.13 .tools/gguf-venv
    uv pip install --python .tools/gguf-venv/bin/python -r .tools/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
    ;;
  download)
    .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "RecursiveSI/flint-int4",
    revision="3b93f9053b530488405ae9fca51d4900d6ce21f0",
    local_dir="models/flint-int4",
    allow_patterns=["*.json", "*.jinja", "*.safetensors", "LICENSE"],
)
PY
    ;;
  prepare|capture|rewrite|compare|score)
    .venv/bin/python corpus.py "$@"
    ;;
  train|train-head|train-prompt|cache-features)
    command="$1"
    shift
    case "$command" in
      train) script=train.py ;;
      train-head) script=train_head.py ;;
      train-prompt) script=train_prompt.py ;;
      cache-features) script=cache_features.py ;;
    esac
    .venv/bin/python "$script" "$@"
    ;;
  build-corpus|select-corpus|batch-capture|distill|replay)
    command="$1"
    shift
    case "$command" in
      build-corpus) script=make_corpus.py ;;
      select-corpus) script=select_corpus.py ;;
      batch-capture) script=batch_capture.py ;;
      distill) script=distill.py ;;
      replay) script=replay.py ;;
    esac
    .venv/bin/python "$script" "$@"
    ;;
  openclaw)
    shift
    .venv/bin/python openclaw.py "$@"
    ;;
  export)
    shift
    .venv/bin/python export.py "$@"
    ;;
  test)
    .venv/bin/python -m unittest discover -s tests -v
    ;;
  *)
    cat <<'EOF'
./run.sh setup
./run.sh setup-gguf
./run.sh download
./run.sh prepare data/seed.jsonl
./run.sh capture data/eval.jsonl --output runs/base.jsonl
./run.sh train --iters 100
./run.sh capture data/eval.jsonl --adapter adapters/terse --output runs/terse.jsonl
./run.sh compare runs/base.jsonl runs/terse.jsonl
./run.sh export --adapter adapters/terse --output models/flint-terse-Q8_0.gguf --install

Corpus building:
./run.sh capture prompts.jsonl --output runs/raw.jsonl
./run.sh rewrite runs/raw.jsonl --output runs/edited.jsonl
# Review edited traces, set approved=true, then prepare them.
./run.sh prepare runs/edited.jsonl

OpenClaw conditioning and regression tasks:
./run.sh openclaw sync
./run.sh openclaw condition data/openclaw-eval.jsonl --output runs/openclaw-eval-prompts.jsonl
./run.sh capture runs/openclaw-eval-prompts.jsonl --api http://localhost:1234/v1 --api-chat --model flint-terse --output runs/openclaw-eval.jsonl

The Python commands accept --help.
EOF
    ;;
esac
