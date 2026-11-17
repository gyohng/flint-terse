#!/usr/bin/env bash
# CUDA llama-server on the unchanged int4 base, with optional adapter arguments.
#   scripts/serve.sh SLOTS [--lora adapters/X/adapter.gguf --lora-init-without-apply]
cd "$(dirname "$0")/.."
SLOTS=${1:-32}; shift || true
exec .tools/llama.cpp-cuda/build/bin/llama-server -m models/flint-int4-gguf/flint-int4-q4_1.gguf \
  -ngl 999 -fa on -c $((SLOTS * 10240)) -np "$SLOTS" --jinja --host 127.0.0.1 --port 8080 \
  --temp 1 --top-p 0.95 --top-k 20 --min-p 0 --no-webui "$@"
