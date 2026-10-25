# flint-terse

Compress Flint's `<think>` into working notes and equations while keeping final replies fluent and complete. The earlier adapter `adapters/prompt-half` is installed in LM Studio as `flint-terse-prompt-half`. It still restates requests and can quote its learned instruction as user text. It is an experimental baseline, not a finished solution to that behavior.

```sh
lms load flint-terse-prompt-half --identifier flint-terse --context-length 16384 --parallel 1 --gpu max --no-speculative-draft-mtp -y
```

Use `flint-terse` as the API model name. That older GGUF includes a 17-token learned prefix in its chat template. Current prefix experiments place the instruction before the first user message; output-head experiments add no instruction.

## Current result

**Reliable recap removal is not solved.** The five-token `adapters/no-recaps-95` candidate answered six ordinary requests without recapping them, using 92 thinking tokens versus 1,661 for base Flint. Its broader accuracy regressed, so it was not exported or installed.

These native MLX comparisons use the same 100 math/code tasks, INT4 checkpoint, temperature 1, top-p .95, top-k 20, independent per-task seeds, batch size 8 and 4,096-token budget:

| Model | Correct | Thinking tokens |
|---|---:|---:|
| Base Flint | 89/100 | 89,818 |
| Earlier `prompt-half` | 94/100 | 49,069 |
| `direct-system` | 80/100 | 16,348 |
| `no-recaps-95` | 76/100 | 13,961 |

The two newer candidates skipped necessary checks and made more math/code errors. The output-head candidates retained recaps; the attention QLoRA candidate also produced malformed thinking tags and incorrect code. Moving the older adapter into a system message corrected the input boundary but did not stop recaps or instruction echoing.

The earlier deployed model remains experimental: a second code run scored 34/40 versus base 35/40 and used more tokens; its LM Studio OpenClaw check scored 15/24 versus 17/24. It passed six multi-turn tool episodes versus four for the original model. Those deployment comparisons also differ in quantization. None of these checks establishes that accuracy is generally preserved.

Captures are in `runs/final-{base-paired,prompt-half,direct-system,no-recaps-95}.jsonl` and `runs/{direct-base,no-recaps-95}-native.jsonl`. The final set has now been used for development; future selection needs fresh tasks.

## How it works

The earlier adapter learns **34,816 values in 17 existing reserved-token embeddings**. Ordinary embeddings, the transformer and the output head stay frozen. The chat template inserts those 17 input tokens before the current assistant turn. Standard MLX LoRA loading and fusion handle the embedding adapter.

The corpus pairs compact, checked Flint-generated thinking with its original final replies. Training combines cross-entropy on compact thinking with KL divergence on final-answer probabilities: the student after short thinking should predict like frozen Flint after its original full thinking. Final replies receive no shortening objective. This uses Flint's own probabilities; it is not a guarantee of the globally shortest equivalent reasoning.

The current corpus has **386 pairs: 343 train, 43 validation**—190 math, 148 code, 26 with the real OpenClaw context, and 22 ordinary requests. Thinking targets start with useful deductions, not task recitals; trivial greetings and a joke use empty thinking. Original final replies are preserved. Ordinary requests receive eight times the loss weight. This is a small development corpus, not evidence of broad reliability.

`train-head` fits a low-rank output-head update without adding prompt text. It can match answer probabilities from the original full reasoning and replay original answers without reinforcing verbose thinking. `train-prompt` learns a short system prefix; `train` supports attention QLoRA. All require generation checks before deployment.

## Train and evaluate

Requires Apple Silicon, `uv`, Git and LM Studio. This machine has 128 GB unified memory. The training checkpoint uses about 22.6 GB, the GGUF 42.6 GB, and export needs roughly 170 GB temporary free space. Unload LM Studio models before training to avoid memory pressure.

```sh
./run.sh setup
./run.sh setup-gguf
./run.sh download
./run.sh test
./run.sh cache-features data/corpus-paired.jsonl --original-reasoning --max-tokens 16384 --output runs/native-features.safetensors
./run.sh train-prompt data/corpus-paired.jsonl --teacher-features runs/native-features.safetensors --init-text 'No prompt recaps.' --prompt-role system --answer-kl 8 --start-weight 3 --strength .95 --output adapters/terse
./run.sh capture data/direct-eval.jsonl --adapter adapters/terse --output runs/direct-check.jsonl
# Evaluate before exporting; the training command does not select a proven model.
./run.sh export --adapter adapters/terse --output models/flint-terse-Q8_0.gguf --install
```

This reproduces the recap-removal experiment above, including its accuracy limitation; it is not a recommended deployment recipe. Choose new output paths when rerunning. Dependencies and model revisions are pinned.

For attention QLoRA with the same full-reasoning answer teacher, use `./run.sh train --paired data/corpus-paired.jsonl --teacher-features runs/native-features.safetensors --layers 8 --iters 400 --kl 2 --start-weight 3 --max-tokens 16384 --output adapters/attention`. `train-head` provides a faster cached-feature experiment. All three trainers preserve final-answer probabilities without applying a shortening loss to final replies in these modes.

Export merges the adapter, restores HF expert tensors and uses the pinned llama.cpp converter. Adapted matrices stay FP16; other matrix weights use Q8. Temporary full-precision shards are removed afterward. Q8 limits additional quantization error; it cannot recover precision already lost in the INT4 source. The earlier `prompt-half` export has been loaded and exercised in normal LM Studio; each new export needs its own check.

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

`build-corpus` imports [GSM8K](https://github.com/openai/grade-school-math) and [sanitized MBPP](https://github.com/google-research/google-research/blob/master/mbpp/sanitized-mbpp.json). `data/prose-prompts.jsonl` and `data/tool-prompts.jsonl` provide additional collection tasks. The original seed set alone was insufficient. The expanded corpus also has not produced reliable recap removal with preserved accuracy.

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

The 159-task holdout is a development set. The 100-task final set has now been evaluated; reserve new tasks for future model selection. `--samples` greater than one is for corpus selection only. Use `compare --cross-runtime` for deployment comparisons with different model identifiers or quantization.

`openclaw sync` reads the authorized remote system instructions and exec/process schemas over SSH. Episode replay feeds actual model responses back into the conversation with fixed tool observations. Generated tool calls are checked without execution.

## Research

[ConPress](https://arxiv.org/abs/2602.01472), [ShorterBetter](https://arxiv.org/abs/2504.21370), [prompt tuning](https://arxiv.org/abs/2104.08691), and [Truncated-Reasoning Self-Distillation](https://arxiv.org/abs/2603.13274) informed the approach. This is a Flint-specific experiment, not a reproduction of their published results or a claim of a universally best method. The differentiable attention-sink implementation and shortest-correct sampling borrow useful ideas from `../flint-terse2`.

[Flint's model card](https://huggingface.co/RecursiveSI/flint) recommends temperature 1, top-p .95 and top-k 20. Our 4,096-token comparisons measure a practical local budget, below its recommendation for hard problems.
