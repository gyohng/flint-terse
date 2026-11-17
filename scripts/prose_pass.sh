#!/usr/bin/env bash
# Prose pass: one supervised epoch on top of an RL adapter. Teaches terse prose/chat thinking (prose and
# tool rows of data/spark-train-r2.jsonl, replies held by answer KL) while replaying the adapter's own
# shortest correct code answers from an RL sample file so its code behaviour holds.
#   scripts/prose_pass.sh ADAPTER SAMPLES.jsonl OUT
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
ADAPTER=$1; SAMPLES=$2; OUT=$3
"$PY" - "$SAMPLES" "runs/$OUT-data.jsonl" <<'P'
import json, sys
best = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    notes = (r['assistant'].get('reasoning_content') or '').strip()
    if r['correct'] and notes and (r['id'] not in best or r['think_tokens'] < best[r['id']]['think_tokens']):
        best[r['id']] = r
code = [{k: v for k, v in r.items() if k not in ('advantage', 'reward', 'correct')} | {'approved': True}
        for r in best.values()]
prose = [r for r in map(json.loads, open('data/spark-train-r2.jsonl')) if r.get('family') in ('prose', 'tool')]
with open(sys.argv[2], 'w') as f:
    for r in code + prose:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'prose pass: {len(code)} own code answers + {len(prose)} prose/tool rows')
P
pkill -x llama-server || true; sleep 4
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" spark_train.py --data "runs/$OUT-data.jsonl" \
  --resume "$ADAPTER" --output "adapters/$OUT" --epochs 1 --accumulate 2 --pack 8192 --rank 32 --alpha 32 \
  --learning-rate 3e-5 --kl 2 --max-tokens 8192 > "runs/train-$OUT.log" 2>&1
"$PY" export_adapter.py --adapter "adapters/$OUT" --model models/flint-int4-gguf/flint-int4-q4_1.gguf > /dev/null
echo "saved adapters/$OUT"
