# flint-terse

Compress Flint's `<think>` into working notes and equations while keeping final replies fluent and complete. The earlier adapter `adapters/prompt-half` is installed in LM Studio as `flint-terse-prompt-half`. It still restates requests and can quote its learned instruction as user text. It is an experimental baseline, not a finished solution to that behavior.

```sh
lms load flint-terse-prompt-half --identifier flint-terse --context-length 16384 --parallel 1 --gpu max --no-speculative-draft-mtp -y
```

Use `flint-terse` as the API model name. That older GGUF includes a 17-token learned prefix in its chat template. Current prefix experiments place the instruction before the first user message; output-head experiments add no instruction.

## Current result

**Reliable recap removal is not solved.** The best recent attention adapter, `adapters/on-policy-qv`, scored 44/50 versus base Flint’s 45/50. It used 26% fewer thinking tokens on the 43 questions both answered correctly, but still recapped ordinary requests. It has not been exported or installed.

These native MLX comparisons use `data/fresh-eval.jsonl`, the same INT4 checkpoint, temperature 1, top-p .95, top-k 20, independent per-task seeds, batch size 8 and a 4,096-token budget:

| Model | Correct | Thinking tokens across all 50 tasks |
|---|---:|---:|
| Base Flint | 45/50 | 55,113 |
| `no-recaps-95` learned prefix | 42/50 | 5,169 |
| `on-policy-qv` attention adapter | 44/50 | 43,072 |

The stronger prefix compression loses accuracy. Resuming that prefix on checked self-generated responses scored only 38/50. Opening-preference experiments produced malformed wording or leaked thinking into final replies; their rejected-text penalty was removed. The current experiment supervises useful openings directly, adds more reviewed prose, and preserves answer probabilities from the original full-thinking teacher.

The older installed `prompt-half` remains experimental too. On an earlier 100-task comparison it scored 94/100 versus base 89/100, but a second code run scored 34/40 versus base 35/40. Its LM Studio OpenClaw check scored 15/24 versus 17/24, while six multi-turn tool episodes passed versus four for the original model. Those deployment comparisons also differ in quantization.

Captures are in `runs/fresh-{base,start,on-policy-qv}.jsonl`. Existing evaluation sets have been used for development. `data/release-eval.jsonl` reserves 100 unseen math/code tasks for a candidate that passes the ordinary-response checks.

## How it works

The attention adapter learns 557,056 values in rank-8 query/value updates across the last eight transformer layers. It adds no instruction to the prompt. The older deployed model instead learns 17 reserved-token embeddings, inserted before the assistant; this is where the sentence quoted in its thinking came from.

Training uses Flint’s probabilities. Compact thinking receives supervised loss, while KL divergence keeps final-answer probabilities near a frozen teacher conditioned on its original full thinking. Verified answer content can also receive supervised loss; its end-of-turn token receives no extra reward. Final replies have no shortening objective. This encourages a small behavioral change, but does not guarantee the globally shortest equivalent reasoning or unchanged accuracy.

The current training mix has **180 reviewed pairs: 135 train, 45 validation**—66 math, 56 code, 13 with the real OpenClaw context and 45 ordinary requests. The original 386-pair corpus remains available. New self-generated traces retain their matching answers; weak tests, contradictory reasoning and incorrect replies are rejected or replaced by a checked native solution. Some difficult tasks keep long reasoning. The 23 additional prose pairs preserve their native final replies and use reviewed working notes.

`--opening-only` supervises the first 16 reasoning tokens, plus the opening tag, with a fixed normalization so empty thoughts do not dominate. It leaves later reasoning available and keeps the original full-thinking answer teacher. `--resume` continues a compatible attention adapter with a fresh optimizer. This is still an experiment requiring generation checks.

## Train and evaluate

Requires Apple Silicon, `uv`, Git and LM Studio. This machine has 128 GB unified memory. The checkpoint uses about 22.6 GB, the GGUF 42.6 GB, and export needs roughly 170 GB temporary free space. Unload LM Studio models before training.

The following reconstructs the current experiment from the reviewed corpora. The learned prefix supplies a probability teacher; the final attention adapter does not use it at inference.

```sh
./run.sh setup
./run.sh setup-gguf
./run.sh download
./run.sh test
./run.sh cache-features data/corpus-paired.jsonl --original-reasoning --max-tokens 16384 --output runs/native-features.safetensors
./run.sh train-prompt data/corpus-paired.jsonl --teacher-features runs/native-features.safetensors --init-text 'No prompt recaps.' --prompt-role system --answer-kl 8 --start-weight 3 --strength .95 --output adapters/no-recaps-95
./run.sh cache-features data/on-policy-pilot.jsonl --adapter adapters/no-recaps-95 --original-reasoning --max-tokens 16384 --output runs/on-policy-teacher.safetensors
./run.sh train --paired data/on-policy-pilot.jsonl --teacher-features runs/on-policy-teacher.safetensors --answer-weight 1 --kl 1 --start-weight 3 --layers 8 --learning-rate .000005 --iters 400 --max-tokens 16384 --output adapters/on-policy-qv
./run.sh cache-features data/prose-extra-paired.jsonl --original-reasoning --max-tokens 16384 --output runs/prose-extra-teacher.safetensors
./run.sh train --paired data/on-policy-pilot.jsonl data/prose-extra-paired.jsonl --teacher-features runs/on-policy-teacher.safetensors runs/prose-extra-teacher.safetensors --opening-only --resume adapters/on-policy-qv --answer-weight 1 --kl 2 --layers 8 --learning-rate .000005 --iters 400 --max-tokens 16384 --output adapters/opening-sft-qv
./run.sh capture data/direct-eval.jsonl --adapter adapters/opening-sft-qv --temperature 1 --max-tokens 4096 --output runs/opening-sft-qv-direct.jsonl
# Export only after the generation and accuracy checks pass.
./run.sh export --adapter adapters/opening-sft-qv --output models/flint-terse-direct-Q8_0.gguf --install
```

Choose fresh output paths when rerunning. Dependencies and model revisions are pinned. `train-prompt` and `train-head` retain the earlier embedding and output-head experiments.

Export merges the adapter, restores HF expert tensors and uses the pinned llama.cpp converter. Adapted matrices stay FP16; other matrix weights use Q8. Temporary full-precision shards are removed afterward. Q8 limits additional quantization error; it cannot recover precision already lost in the INT4 source. Each new GGUF needs its own LM Studio check.

## Build more corpus

Most compact traces come from Flint after four short-thinking demonstrations. Selection keeps the shortest correct completion, falling back to a checked native response. The demonstrations are removed from training inputs. `pair` preserves the native final reply exactly and stores its full thinking for the probability teacher.

```sh
./run.sh batch-capture prompts.jsonl --samples 2 --batch-size 8 --output runs/native.jsonl
./run.sh distill build prompts.jsonl --output runs/teacher-prompts.jsonl
./run.sh batch-capture runs/teacher-prompts.jsonl --samples 2 --batch-size 8 --output runs/teacher.jsonl
./run.sh distill select runs/teacher.jsonl --baseline runs/native.jsonl --output runs/compact.jsonl
# Review native and compact traces; mark accepted native rows approved=true.
./run.sh distill pair runs/compact.jsonl --baseline runs/native.jsonl --output data/new-corpus.jsonl
```

Records need `id`, `messages`, `split`, and a numeric, exact-text, JSON, tool-call or Python-test check. Optional `tools` retain the real schema. Review answers and reasoning before training: passing a few code tests does not prove a solution correct. `score` recomputes likelihood under an edited prefix; likelihood alone is not a correctness test.

`build-corpus` imports [GSM8K](https://github.com/openai/grade-school-math) and [sanitized MBPP](https://github.com/google-research/google-research/blob/master/mbpp/sanitized-mbpp.json). `data/prose-prompts.jsonl` and `data/tool-prompts.jsonl` provide additional collection tasks. The original seed set alone was insufficient. More examples help only when the reasoning and replies survive review; the expanded corpus has not yet established reliable recap removal with preserved accuracy.

For OpenClaw tasks, run `./run.sh openclaw sync`, then `./run.sh openclaw condition data/tool-prompts.jsonl --output runs/tool-prompts.jsonl` before collecting responses.

## Evaluate

```sh
./run.sh batch-capture data/holdout.jsonl --batch-size 8 --max-tokens 4096 --independent-seeds --output runs/base.jsonl
./run.sh batch-capture data/holdout.jsonl --batch-size 8 --max-tokens 4096 --independent-seeds --adapter adapters/terse --output runs/candidate.jsonl
./run.sh compare runs/base.jsonl runs/candidate.jsonl
./run.sh openclaw condition data/openclaw-eval.jsonl --output runs/openclaw-eval-prompts.jsonl
./run.sh capture runs/openclaw-eval-prompts.jsonl --api http://localhost:1234/v1 --api-chat --model flint-terse --temperature 1 --max-tokens 4096 --output runs/openclaw-terse.jsonl
./run.sh replay data/openclaw-episodes.jsonl --model flint-terse --temperature 1 --output runs/episodes-terse.jsonl
```

The 159-task holdout, 100-task final set and 50-task fresh set are now development sets. Keep the new release set untouched until a candidate passes the direct-response checks. `--samples` greater than one is for corpus selection only. Use `compare --cross-runtime` for deployment comparisons with different model identifiers or quantization.

`openclaw sync` reads the authorized remote system instructions and exec/process schemas over SSH. Episode replay feeds actual model responses back into the conversation with fixed tool observations. Generated tool calls are checked without execution.

## Research

[ConPress](https://arxiv.org/abs/2602.01472), [ShorterBetter](https://arxiv.org/abs/2504.21370), [prompt tuning](https://arxiv.org/abs/2104.08691), and [Truncated-Reasoning Self-Distillation](https://arxiv.org/abs/2603.13274) informed the approach. The rejected opening-preference experiment was inspired by [DPO](https://arxiv.org/abs/2305.18290). This is a Flint-specific experiment, not a reproduction of their published results or a claim of a universally best method. The differentiable attention-sink implementation and shortest-correct sampling borrow useful ideas from `../flint-terse2`.

[Flint's model card](https://huggingface.co/RecursiveSI/flint) recommends temperature 1, top-p .95 and top-k 20. Our 4,096-token comparisons measure a practical local budget, below its recommendation for hard problems.
