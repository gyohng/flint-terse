#!/usr/bin/env bash
# Reproduce flint-terse and flint-terse-minimalist from the unchanged int4 base (run scripts/setup.sh first).
# About 20 hours on one NVIDIA GB10. RL sampling is stochastic: expect results within a few tasks of the README.
set -euo pipefail
cd "$(dirname "$0")/.."

# Baseline of the unchanged base on every eval set (runs/spark-base-*.jsonl).
[ -f runs/spark-base-release-eval.jsonl ] || scripts/eval.sh base base

# 1. Budget RL from the base: 300 code prompts x 8 answers per iteration, budgets 600 -> 400 -> 300 tokens.
EVAL=0 PROMPTS=300 LR=5e-5 ACCUM=4 SEED=20 scripts/rl.sh base budget "600 400 300"

# 2. Prose pass on iteration 3, replaying the answers it was trained on -> flint-terse.
scripts/prose_pass.sh adapters/budget-3 runs/budget-3-samples.jsonl flint-terse
scripts/eval.sh adapters/flint-terse flint-terse

# 3. Add per-expert LoRA (rank 8, last 16 MoE layers) and run two RL iterations at a fixed 300-token budget,
#    400 prompts each -> flint-terse-minimalist.
EVAL=0 PROMPTS=400 LR=2e-5 ACCUM=4 SEED=40 EXPERT_RANK=8 EXPERT_LAYERS=16 scripts/rl.sh adapters/flint-terse minimal "300 300"
rm -rf adapters/flint-terse-minimalist && cp -r adapters/minimal-2 adapters/flint-terse-minimalist
scripts/eval.sh adapters/flint-terse-minimalist flint-terse-minimalist

# Install the GGUF adapters next to their release metadata, ready for ./run.sh deploy.
for name in flint-terse flint-terse-minimalist; do
  cp "adapters/$name/adapter.gguf" "release/$name/adapter.gguf"
done
pkill -x llama-server || true
echo "done: release/flint-terse/adapter.gguf, release/flint-terse-minimalist/adapter.gguf"
