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
    cmake -S .tools/llama.cpp -B .tools/llama.cpp/build -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DGGML_METAL=OFF
    cmake --build .tools/llama.cpp/build --target llama-quantize -j 8
    ;;
  setup-lmstudio)
    npm install --prefix .tools/lms-sdk --no-save @lmstudio/sdk@2.1.0
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
  build-corpus|select-corpus|batch-capture|build-preferences|distill|prune|replay)
    command="$1"
    shift
    case "$command" in
      build-corpus) script=make_corpus.py ;;
      select-corpus) script=select_corpus.py ;;
      batch-capture) script=batch_capture.py ;;
      build-preferences) script=preferences.py ;;
      distill) script=distill.py ;;
      prune) script=prune.py ;;
      replay) script=replay.py ;;
    esac
    .venv/bin/python "$script" "$@"
    ;;
  openclaw)
    shift
    .venv/bin/python openclaw.py "$@"
    ;;
  serve)
    shift
    exec .venv/bin/mlx_lm.server \
      --model models/flint-int4 --adapter-path adapters/opening-pruned \
      --host 127.0.0.1 --port 8080 \
      --temp 1 --top-p .95 --top-k 20 --max-tokens 4096 \
      --decode-concurrency 1 --prompt-concurrency 1 \
      --prefill-step-size 512 --prompt-cache-size 1 "$@"
    ;;
  export)
    shift
    .venv/bin/python export.py "$@"
    ;;
  export-adapter)
    shift
    .tools/gguf-venv/bin/python export_adapter.py "$@"
    ;;
  load-lmstudio)
    shift
    node load_lmstudio.mjs "$@"
    ;;
  test)
    .venv/bin/python -m unittest discover -s tests -v
    ;;
  *)
    cat <<'EOF'
./run.sh setup
./run.sh setup-gguf
./run.sh download
./run.sh serve # Original MLX weights + separate adapter; API model: default_model
./run.sh export-adapter # FP32 GGUF adapter + Jinja override; no base conversion
./run.sh setup-lmstudio
./run.sh load-lmstudio # Original downloaded Q4 GGUF + separate FP32 adapter
# README.md contains the opening-only training recipe and attention alternatives.
./run.sh train-prompt --help
./run.sh capture data/direct-eval.jsonl --adapter adapters/opening-pruned --output runs/direct-check.jsonl
./run.sh batch-capture data/holdout.jsonl --independent-seeds --batch-size 8 --max-tokens 4096 --output runs/base.jsonl
./run.sh batch-capture data/holdout.jsonl --independent-seeds --batch-size 8 --max-tokens 4096 --adapter adapters/terse --output runs/terse.jsonl
./run.sh compare runs/base.jsonl runs/terse.jsonl
# Optional merged/requantized export; use export-adapter to keep the base unchanged.
./run.sh export --adapter adapters/terse --output models/flint-terse-Q8_0.gguf --install

Corpus building:
./run.sh batch-capture prompts.jsonl --samples 2 --output runs/raw.jsonl
./run.sh distill build prompts.jsonl --output runs/teacher-prompts.jsonl
./run.sh batch-capture runs/teacher-prompts.jsonl --samples 2 --output runs/teacher.jsonl
./run.sh distill select runs/teacher.jsonl --baseline runs/raw.jsonl --output runs/compact.jsonl
# Review both native and compact traces; mark accepted native rows approved=true.
./run.sh distill pair runs/compact.jsonl --baseline runs/raw.jsonl --output data/new-corpus.jsonl

OpenClaw conditioning and regression tasks:
./run.sh openclaw sync
./run.sh openclaw condition data/openclaw-eval.jsonl --output runs/openclaw-eval-prompts.jsonl
./run.sh capture runs/openclaw-eval-prompts.jsonl --api http://localhost:1234/v1 --api-chat --model flint-terse --output runs/openclaw-eval.jsonl

The Python commands accept --help.
EOF
    ;;
esac
