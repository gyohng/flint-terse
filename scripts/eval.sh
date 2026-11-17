#!/usr/bin/env bash
# Evaluate the base (ADAPTER=base) or an adapter on every eval set, compare with the saved base run.
#   scripts/eval.sh ADAPTER_DIR|base LABEL [SEED]
# Results: runs/spark-LABEL-SET.jsonl; summary table on stdout. Leaves the server running.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
ADAPTER=$1; LABEL=$2; SEED=${3:-42}
SETS="data/release-eval.jsonl data/confirmation-eval.jsonl data/direct-eval.jsonl data/code-prose-eval.jsonl
      data/retention-eval.jsonl data/release-chat.jsonl data/spark-code-chat-eval.jsonl data/spark-prose-eval.jsonl"
mkdir -p runs
pkill -x llama-server || true; sleep 4
if [ "$ADAPTER" = base ]; then
  nohup scripts/serve.sh 32 > runs/llama-server.log 2>&1 &
  LORA=()
else
  [ -f "$ADAPTER/adapter.gguf" ] || "$PY" export_adapter.py --adapter "$ADAPTER" --model models/flint-int4-gguf/flint-int4-q4_1.gguf
  nohup scripts/serve.sh 32 --lora "$ADAPTER/adapter.gguf" --lora-init-without-apply > runs/llama-server.log 2>&1 &
  LORA=(--lora-scale 1)
fi
until curl -sf localhost:8080/health >/dev/null; do sleep 3; done
"$PY" spark_eval.py --label "$LABEL" --seed "$SEED" --workers 32 "${LORA[@]}" --sets $SETS
[ "$LABEL" = base ] || "$PY" spark_eval.py --compare base "$LABEL" --sets $SETS
