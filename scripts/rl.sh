#!/usr/bin/env bash
# GRPO-style reinforcement learning under a thinking budget: one iteration per budget in the list.
# Each iteration serves the current adapter, samples GROUP answers for PROMPTS code prompts, scores
# them (reward 1 if the tests pass and thinking <= budget, else 0; group-centred, batch-normalised),
# trains one epoch of policy gradient plus prose replay, exports the GGUF and (EVAL=1) evaluates.
#   scripts/rl.sh START OUT "600 400 300"      START: adapter directory or "base"
#   scripts/rl.sh START OUT "rel0.3 rel0.3"     relative budget: 30% of the base's thinking per task
#   scripts/rl.sh START OUT "soft300"           soft budget: correct * (1 - ALPHA * min(1, L/300))
#   scripts/rl.sh START OUT "srel0.25"          per-task soft: correct * (1 - ALPHA * min(1, L/L_base - 0.25))
# Outputs adapters/OUT-1, OUT-2, ...; samples and logs in runs/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
START=$1; OUT=$2; BUDGETS=$3
PROMPTS=${PROMPTS:-300} GROUP=${GROUP:-8} LR=${LR:-5e-5} ACCUM=${ACCUM:-4} SEED=${SEED:-20} EVAL=${EVAL:-1}
EXPERT=(); [ -n "${EXPERT_RANK:-}" ] && EXPERT=(--expert-rank "$EXPERT_RANK" --expert-layers "${EXPERT_LAYERS:-16}")
GGUF=models/flint-int4-gguf/flint-int4-q4_1.gguf
mkdir -p runs
PREV=$START; it=0
for B in $BUDGETS; do
  it=$((it + 1)); NAME=$OUT-$it
  echo "=== $NAME: budget $B from $PREV ($(date +%T))"
  pkill -x llama-server || true; sleep 4
  if [ "$PREV" = base ]; then
    nohup scripts/serve.sh 32 > runs/llama-server.log 2>&1 &
    SCALE=(); RESUME=()
  else
    nohup scripts/serve.sh 32 --lora "$PREV/adapter.gguf" --lora-init-without-apply > runs/llama-server.log 2>&1 &
    SCALE=(--lora-scale 1); RESUME=(--resume "$PREV")
  fi
  until curl -sf localhost:8080/health >/dev/null; do sleep 3; done
  "$PY" rl_sample.py data/spark-pool.jsonl,data/spark-code-pool.jsonl --output runs/$NAME-samples.jsonl \
    --ref-log data/spark-r0.log.jsonl --families code --limit "$PROMPTS" --group "$GROUP" --seed $((SEED + it)) \
    "${SCALE[@]}" --max-tokens $([[ $B == [rs]* ]] && echo 3000 || echo $((B + 1500))) --workers 32 > runs/$NAME-samples.log 2>&1
  tail -1 runs/$NAME-samples.log
  if [[ $B == rel* ]]; then BUDGET=(--rel-budget "${B#rel}")
  elif [[ $B == soft* ]]; then BUDGET=(--soft-budget "${B#soft}" --alpha "${ALPHA:-0.5}")
  elif [[ $B == srel* ]]; then BUDGET=(--floor "${B#srel}" --alpha "${ALPHA:-0.5}")
  else BUDGET=(--budget "$B"); fi
  "$PY" rescore_rl.py runs/$NAME-samples.jsonl --ref-log data/spark-r0.log.jsonl data/spark-insight.log.jsonl \
    data/spark-ll-extra.log.jsonl data/spark-hard.log.jsonl "${BUDGET[@]}" --output runs/$NAME-scored.jsonl
  pkill -x llama-server || true; sleep 4
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" spark_train.py --data runs/$NAME-scored.jsonl \
    data/spark-replay-prose.jsonl "${RESUME[@]}" --output adapters/$NAME --epochs 1 --accumulate "$ACCUM" \
    --pack 8192 --rank 32 --alpha 32 --learning-rate "$LR" --kl 2 --valid 16 --max-tokens 8192 "${EXPERT[@]}" > runs/train-$NAME.log 2>&1
  "$PY" export_adapter.py --adapter adapters/$NAME --model "$GGUF" > /dev/null
  [ "$EVAL" = 1 ] && scripts/eval.sh adapters/$NAME $NAME | grep -E "^(set|math|code|code-prose|prose|direct|retention) "
  PREV=adapters/$NAME
done
echo "=== $OUT done ($(date +%T))"
