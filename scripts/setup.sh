#!/usr/bin/env bash
# One-time setup of a CUDA host: Python packages, a CUDA llama.cpp server, the pinned int4 base.
#   PY=python3.15 scripts/setup.sh          (installs into that interpreter's environment)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
LLAMA_COMMIT=cb44fdb1892eebca072e0949875a60c11245a8bb
CUDA_ARCH=${CUDA_ARCH:-121}   # GB10; e.g. 90 for H100, 120 for RTX 50xx
BASE_URL=https://huggingface.co/RecursiveSI/flint-int4-gguf/resolve/9c47151036bc2c34331142dda04488bceab2bc9f/flint-int4-q4_1.gguf
BASE_SHA=43c07baccdba42eb7528a9630ee15699b1942c297291dc1b39a061f508ee1a4b

# Python: CUDA torch, transformers, the CUDA 13.0 toolchain from pip, gguf-py from the pinned llama.cpp.
PIP=(uv pip install --python "$PY" --system --break-system-packages)
command -v uv >/dev/null || PIP=("$PY" -m pip install)
"${PIP[@]}" torch --index-url https://download.pytorch.org/whl/cu130
"${PIP[@]}" -r requirements-cuda.txt

# llama.cpp with CUDA, built against the pip toolkit.
mkdir -p .tools
if [ ! -d .tools/llama.cpp-cuda ]; then
  git init -q .tools/llama.cpp-cuda
  git -C .tools/llama.cpp-cuda fetch -q --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_COMMIT"
  git -C .tools/llama.cpp-cuda checkout -q --detach FETCH_HEAD
fi
"${PIP[@]}" ./.tools/llama.cpp-cuda/gguf-py
CUDA=$("$PY" -c "import nvidia, pathlib; print(pathlib.Path(nvidia.__path__[0]) / 'cu13')")
ln -sfn lib "$CUDA/lib64"                       # nvcc.profile looks in lib64
mkdir -p "$CUDA/lib/stubs"
LIBCUDA=$(ldconfig -p | awk '/libcuda.so.1 /{print $NF; exit}')
ln -sf "$LIBCUDA" "$CUDA/lib/stubs/libcuda.so"
for f in "$CUDA"/lib/*.so.[0-9]*; do b=$(basename "$f"); [ -e "$CUDA/lib/${b%%.so.*}.so" ] || ln -s "$b" "$CUDA/lib/${b%%.so.*}.so"; done
if [ ! -x .tools/llama.cpp-cuda/build/bin/llama-server ]; then
  PATH="$CUDA/bin:$PATH" CUDA_HOME="$CUDA" cmake -S .tools/llama.cpp-cuda -B .tools/llama.cpp-cuda/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA" -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_WEBUI=OFF -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,$CUDA/lib -Wl,-rpath,$CUDA/lib" \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,$CUDA/lib"
  cmake --build .tools/llama.cpp-cuda/build --target llama-server -j "$(nproc)"
fi

# The pinned int4 base (26 GB) and the BF16 repository's tokenizer and config (no weights needed).
mkdir -p models/flint-int4-gguf models/flint
GGUF=models/flint-int4-gguf/flint-int4-q4_1.gguf
until echo "$BASE_SHA  $GGUF" | sha256sum -c --status 2>/dev/null; do
  curl -L --fail -C - --retry 5 -o "$GGUF" "$BASE_URL" || sleep 5
done
"$PY" -c "from huggingface_hub import snapshot_download as d; d('RecursiveSI/flint', local_dir='models/flint', allow_patterns=['config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'])"
echo "setup complete"
