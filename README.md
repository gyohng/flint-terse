# flint-terse

Compress Flint's `<think>` into useful working notes and equations while keeping final replies fluent and complete. The current recipe trains a small attention adapter. It adds no text to the prompt.

The older `flint-terse-prompt-half` is installed in LM Studio but unloaded. Its chat template appends a learned 17-token instruction to the user message; that caused the quoted “Think in brief working notes…” line. It is not a finished solution. No replacement has been installed yet.

## Current result

**Reliable recap removal with preserved accuracy is not solved.** With the original embeddings, `adapters/correctness-base-qv` scored 43/50 but still recapped requests. Continuing it with compact-versus-verbose preferences did not fix recaps and produced irrelevant reasoning even for a greeting. `adapters/compact-preference-qv` is unsuitable for deployment. A forced bullet opening also failed. The next corpus will preserve more of Flint's native working and remove redundant passages using likelihood checks.

These native MLX comparisons use `data/fresh-eval.jsonl`, the same INT4 checkpoint, temperature 1, top-p .95, top-k 20, independent per-task seeds, batch size 8 and a 4,096-token budget:

| Model | Correct | Thinking reduction on shared correct answers |
|---|---:|---:|
| Base Flint | 45/50 | — |
| `on-policy-qv` attention adapter | 44/50 | 26%; still recaps |
| `think-attention` native tokens + attention | 40/50 | 93%; loses accuracy |
| `correctness-qv` native tokens + attention + DPO | 38/50 | 94%; loses accuracy |
| `correctness-base-qv` original embeddings + attention + DPO | 43/50 | 23%; still recaps |

Captures are in `runs/fresh-{base,on-policy-qv,think-attention,correctness-qv}.jsonl`. On ten ordinary requests, `correctness-qv` completed all ten with 61% fewer thinking tokens; longer constrained requests still produced drafting loops, and its binary-search explanation misstated which half gets discarded. It has not been exported. These are development results, not independent proof of generalization. `data/release-eval.jsonl` reserves 100 unseen math/code tasks; `data/release-chat.jsonl` adds six ordinary and multistep requests. Neither has been used yet.

## How it works

The attention adapter learns 557,056 values in rank-8 query/value updates across the last eight transformer layers. The input embeddings, output head and other model weights stay frozen. Training caches activations from the frozen first 41 layers and recomputes all eight trainable layers, retaining their complete context and gradients. The optional native-token experiment changes 4,096 values in the existing `<think>` and `</think>` embeddings; its aggressive compression has lost accuracy.

Training uses Flint’s probabilities in three ways:

- Supervised loss teaches reviewed working notes. Some difficult examples retain longer reasoning.
- KL divergence anchors final-answer probabilities to a frozen teacher conditioned on the original thinking. Verified answers also receive supervised loss, with extra weight on their opening to preserve language and format.
- DPO prefers a correct complete response over Flint's own checked mistake, relative to the frozen starting model's likelihood ratio. It uses the entire response, including thinking.
- `--compact-preference-weight` optionally adds a second DPO comparison: reviewed compact thinking versus the original verbose thinking, with identical final replies. This teaches brevity directly; it is separate from the correctness comparison.

Final replies have no shortening objective. Probability anchoring encourages a small behavioral change; it cannot guarantee unchanged accuracy or the shortest equivalent reasoning.

The current corpus has **410 reviewed pairs: 365 train, 45 validation**—189 math, 147 code, 26 OpenClaw and 48 prose. Of these, 137 training and 19 validation examples include a checked incorrect response. `data/adaptive-sampled.jsonl` contains 59 reliable self-generated traces with their matching answers; `data/adaptive-native.jsonl` contains 351 reviewed native pairs. Code tests and numeric checks guide selection, followed by review. More examples alone do not establish quality.

There are also 349 shorter-versus-original pairs (307 train, 42 validation). The compact preference stage uses weight `.1`, alongside correctness preference weight `1`, answer KL `2` and answer supervision `1`. Both comparisons use complete responses and a frozen reference likelihood ratio.

`data/expanded-prompts.jsonl` adds 880 new collection prompts: 300 math, 80 code and 500 prose. They are disjoint from the existing task IDs and exact prompts. The prose prompts come from the Apache-2.0 Smol-Magpie-Ultra, Smol-rewrite and Smol-summarize subsets of [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk); only prompts are imported, and Flint generates its own responses. These are collection inputs, not approved training examples.

## Train and evaluate

Requires Apple Silicon, `uv`, Git and LM Studio. This machine has 128 GB unified memory. The checkpoint uses about 22.6 GB, the GGUF 42.6 GB, and export needs roughly 170 GB temporary free space. Unload LM Studio models before training.

Using the existing `adapters/think-tokens` checkpoint as the probability teacher for its sampled responses, rebuild the frozen answer features and train the current attention recipe:

```sh
./run.sh setup
./run.sh setup-gguf
./run.sh download
./run.sh test
./run.sh cache-features data/adaptive-sampled.jsonl --adapter adapters/think-tokens --original-reasoning --max-tokens 16384 --output runs/sampled-teacher-new.safetensors
./run.sh cache-features data/adaptive-native.jsonl --original-reasoning --max-tokens 16384 --output runs/native-teacher-new.safetensors
./run.sh train --paired data/adaptive-sampled.jsonl data/adaptive-native.jsonl \
  --teacher-features runs/sampled-teacher-new.safetensors runs/native-teacher-new.safetensors \
  --preference-weight 1 --preference-beta .1 \
  --answer-weight 1 --kl 2 --start-weight 3 --layers 8 \
  --learning-rate .000005 --iters 800 --max-tokens 16384 --output adapters/correctness-new
./run.sh batch-capture data/direct-eval.jsonl --adapter adapters/correctness-new --independent-seeds --batch-size 8 --max-tokens 4096 --output runs/direct-new.jsonl
./run.sh batch-capture data/retention-eval.jsonl --adapter adapters/correctness-new --independent-seeds --batch-size 8 --max-tokens 4096 --output runs/retention-new.jsonl
```

Choose fresh output paths when rerunning. `--iters 800 --accumulate 4` means 200 optimizer updates. Validation selects the saved checkpoint. The native-token stage is also runnable with `train-prompt --think-tokens`, reviewed paired inputs and their original-reasoning teacher features. `--think-adapter` freezes that saved stage; it cannot be combined with `--resume`. Earlier prefix and output-head experiments remain available through `train-prompt` and `train-head`.

To run the compact preference stage, repeat the attention-training command with `--resume adapters/correctness-new --compact-preference-weight .1` and a new `--output`. Evaluate that new adapter before export.

After reviewing ordinary replies, test accuracy with matched settings below. Only then export and load a candidate:

```sh
./run.sh export --adapter adapters/correctness-new --output models/flint-terse-direct-Q8_0.gguf --install
lms load flint-terse-direct --identifier flint-terse --context-length 16384 --parallel 1 --gpu max --no-speculative-draft-mtp -y
```

Export merges the adapter, restores HF expert tensors and uses the pinned llama.cpp converter. Adapted matrices stay FP16; other matrix weights use Q8. Temporary full-precision shards are removed afterward. Q8 limits additional quantization error; it cannot recover precision already lost in the INT4 source. Each new GGUF needs its own LM Studio check.

## Build more corpus

To collect a broader native corpus:

```sh
./run.sh build-corpus --expand data/expanded-prompts-new.jsonl
./run.sh batch-capture data/expanded-prompts-new.jsonl --batch-size 16 --max-tokens 4096 --independent-seeds --output runs/expanded-native-new.jsonl
```

The expansion command skips task IDs and exact prompts already present in `data/`. Review the captured responses before using them as training targets.

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
