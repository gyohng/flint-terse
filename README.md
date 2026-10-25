# flint-terse

Compress Flint's `<think>` into useful working notes and equations while keeping final replies fluent and complete. The current recipe tunes existing thinking-token embeddings and a small attention adapter. It adds no text to the prompt.

The older `flint-terse-prompt-half` is installed in LM Studio but unloaded. Its chat template appends a learned 17-token instruction to the user message; that caused the quoted “Think in brief working notes…” line. It is not a finished solution. No replacement has been installed yet.

## Current result

**Reliable recap removal with preserved accuracy is not solved.** The current run, `adapters/correctness-qv`, adds checked correctness preferences to the training objective. It has not been evaluated or exported.

These native MLX comparisons use `data/fresh-eval.jsonl`, the same INT4 checkpoint, temperature 1, top-p .95, top-k 20, independent per-task seeds, batch size 8 and a 4,096-token budget:

| Model | Correct | Thinking reduction on shared correct answers |
|---|---:|---:|
| Base Flint | 45/50 | — |
| `on-policy-qv` attention adapter | 44/50 | 26%; still recaps |
| `think-attention` native tokens + attention | 40/50 | 93%; loses accuracy |

Captures are in `runs/fresh-{base,on-policy-qv,think-attention}.jsonl`. These are development results, not independent proof of generalization. `data/release-eval.jsonl` reserves 100 unseen math/code tasks; `data/release-chat.jsonl` adds six ordinary and multistep requests. Neither has been used yet.

## How it works

The native-token adapter learns 4,096 values in the existing `<think>` and `</think>` input embeddings. These stay frozen during the next stage, which learns 557,056 values in rank-8 query/value updates across the last eight transformer layers. The output head and other model weights stay frozen.

Training uses Flint’s probabilities in three ways:

- Supervised loss teaches reviewed working notes. Some difficult examples retain longer reasoning.
- KL divergence anchors final-answer probabilities to a frozen teacher conditioned on the original thinking. Verified answers also receive supervised loss, with extra weight on their opening to preserve language and format.
- DPO prefers a correct complete response over Flint's own checked mistake, relative to the frozen starting model's likelihood ratio. It uses the entire response, including thinking.

Final replies have no shortening objective. Probability anchoring encourages a small behavioral change; it cannot guarantee unchanged accuracy or the shortest equivalent reasoning.

The current corpus has **410 reviewed pairs: 365 train, 45 validation**—189 math, 147 code, 26 OpenClaw and 48 prose. Of these, 137 training and 19 validation examples include a checked incorrect response. `data/adaptive-sampled.jsonl` contains 59 reliable self-generated traces with their matching answers; `data/adaptive-native.jsonl` contains 351 reviewed native pairs. Code tests and numeric checks guide selection, followed by review. More examples alone do not establish quality.

## Train and evaluate

Requires Apple Silicon, `uv`, Git and LM Studio. This machine has 128 GB unified memory. The checkpoint uses about 22.6 GB, the GGUF 42.6 GB, and export needs roughly 170 GB temporary free space. Unload LM Studio models before training.

Using the existing `adapters/think-tokens` checkpoint, rebuild the frozen answer features and train the current attention recipe:

```sh
./run.sh setup
./run.sh setup-gguf
./run.sh download
./run.sh test
./run.sh cache-features data/adaptive-sampled.jsonl --adapter adapters/think-tokens --original-reasoning --max-tokens 16384 --output runs/sampled-teacher-new.safetensors
./run.sh cache-features data/adaptive-native.jsonl --original-reasoning --max-tokens 16384 --output runs/native-teacher-new.safetensors
./run.sh train --paired data/adaptive-sampled.jsonl data/adaptive-native.jsonl \
  --teacher-features runs/sampled-teacher-new.safetensors runs/native-teacher-new.safetensors \
  --think-adapter adapters/think-tokens --preference-weight 1 --preference-beta .1 \
  --answer-weight 1 --kl 2 --start-weight 3 --layers 8 \
  --learning-rate .000005 --iters 800 --max-tokens 16384 --output adapters/correctness-new
./run.sh batch-capture data/direct-eval.jsonl --adapter adapters/correctness-new --independent-seeds --batch-size 8 --max-tokens 4096 --output runs/direct-new.jsonl
./run.sh batch-capture data/retention-eval.jsonl --adapter adapters/correctness-new --independent-seeds --batch-size 8 --max-tokens 4096 --output runs/retention-new.jsonl
```

Choose fresh output paths when rerunning. `--iters 800 --accumulate 4` means 200 optimizer updates. Validation selects the saved checkpoint. The native-token stage is also runnable with `train-prompt --think-tokens`, reviewed paired inputs and their original-reasoning teacher features. `--think-adapter` freezes that saved stage; it cannot be combined with `--resume`. Earlier prefix and output-head experiments remain available through `train-prompt` and `train-head`.

After reviewing ordinary replies, test accuracy with matched settings below. Only then export and load a candidate:

```sh
./run.sh export --adapter adapters/correctness-new --output models/flint-terse-direct-Q8_0.gguf --install
lms load flint-terse-direct --identifier flint-terse --context-length 16384 --parallel 1 --gpu max --no-speculative-draft-mtp -y
```

Export merges the adapter, restores HF expert tensors and uses the pinned llama.cpp converter. Adapted matrices stay FP16; other matrix weights use Q8. Temporary full-precision shards are removed afterward. Q8 limits additional quantization error; it cannot recover precision already lost in the INT4 source. Each new GGUF needs its own LM Studio check.

## Build more corpus

Collect several attempts per training prompt. Prefer the shortest correct trace only when repeated attempts show that short reasoning is reliable; otherwise keep useful longer work. Four short-thinking demonstrations can help collect candidates and are removed from the training inputs. `pair` preserves the native final reply exactly and stores its full thinking for the probability teacher.

```sh
./run.sh batch-capture prompts.jsonl --samples 2 --batch-size 8 --output runs/native.jsonl
./run.sh distill build prompts.jsonl --output runs/teacher-prompts.jsonl
./run.sh batch-capture runs/teacher-prompts.jsonl --samples 2 --batch-size 8 --output runs/teacher.jsonl
./run.sh distill select runs/teacher.jsonl --baseline runs/native.jsonl --output runs/compact.jsonl
# Review native and compact traces; mark accepted native rows approved=true.
./run.sh distill pair runs/compact.jsonl --baseline runs/native.jsonl --output data/new-corpus.jsonl
```

Records need `id`, `messages`, `split`, and a numeric, exact-text, JSON, tool-call or Python-test check. Optional `tools` retain the real schema. Review answers and reasoning before training: passing a few code tests does not prove a solution correct. `score` recomputes likelihood under an edited prefix; likelihood alone is not a correctness test.

To add correctness preferences, save every attempt and attach checked mistakes to reviewed positives:

```sh
./run.sh batch-capture prompts.jsonl --adapter adapters/think-tokens --samples 8 --independent-seeds --batch-size 8 --max-tokens 4096 --output runs/selected.jsonl --candidates-output runs/attempts.jsonl
./run.sh build-preferences data/new-corpus.jsonl --candidates runs/attempts.jsonl --output data/new-preferences.jsonl
```

The preference builder regrades complete responses, ignores numeric formatting-only differences and retains the shortest wrong answer as the negative. It leaves the reviewed positive and its probability teacher unchanged. Keep evaluation prompts out of collection and training.

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

[DPO](https://arxiv.org/abs/2305.18290) supplies the complete-response preference objective. [ConPress](https://arxiv.org/abs/2602.01472), [ShorterBetter](https://arxiv.org/abs/2504.21370), [prompt tuning](https://arxiv.org/abs/2104.08691), and [Truncated-Reasoning Self-Distillation](https://arxiv.org/abs/2603.13274) informed the other experiments. This combines supervised training, answer KL and correctness DPO; it does not implement GRPO or reproduce those papers' results. There is no claim of a universally best method. The differentiable attention-sink implementation and shortest-correct sampling borrow useful ideas from `../flint-terse2`.

[Flint's model card](https://huggingface.co/RecursiveSI/flint) recommends temperature 1, top-p .95 and top-k 20. Our 4,096-token comparisons measure a practical local budget, below its recommendation for hard problems.
