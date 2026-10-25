# flint-terse

Compress Flint's `<think>` into working notes and equations while keeping final replies fluent and complete. The selected adapter is `adapters/prompt-half`; its merged GGUF is `models/flint-terse-prompt-half-Q8_0.gguf`, already imported into LM Studio.

```sh
lms load flint-terse-prompt-half --identifier flint-terse --context-length 16384 --parallel 1 --gpu max --no-speculative-draft-mtp -y
```

Use `flint-terse` as the API model name. The learned prefix is included in the GGUF's chat template; no extra user prompt is needed.

## Results

On 100 math/code tasks kept out of adapter selection:

| Measure | Base Flint | Selected adapter |
|---|---:|---:|
| Correct answers | 89/100 | 94/100 |
| Completed responses | 90/100 | 96/100 |
| Thinking tokens | 89,818 | 49,069 (−45.4%) |
| Total output tokens | 91,805 | 51,175 (−44.3%) |
| Math correct | 51/60 | 58/60 |
| Code correct | 38/40 | 36/40 |

For the 86 tasks both models answered correctly, thinking fell 41.8%. The benefit was concentrated in math: −69% thinking on shared successes; code thinking was essentially unchanged. These native MLX runs used the same INT4 base, temperature 1, top-p .95, top-k 20, independent per-task random streams, batch size 8 and a 4,096-token budget.

In normal LM Studio, the GGUF passed all six multi-turn OpenClaw episodes versus four for the original model. The original calculator request used 168 thinking tokens versus 476 and made the correct tool call. All four ordinary-response checks completed, with full explanations, an email and a plan.

The separate 24-task OpenClaw challenge scored **15/24 versus 17/24**: exact JSON formatting still regressed, alongside a list-order error and an unwanted calculator tool call. Deployment comparisons use the user's original Q4 GGUF versus an export from the INT4 MLX checkpoint, so quantization also differs. This candidate improves overall efficiency, with uneven results across tasks; it does not establish preserved coding accuracy or consistently minimal agent reasoning.

Detailed captures are in `runs/final-{base-paired,prompt-half}.jsonl`, `runs/openclaw-{base-v1,prompt-half-gguf}.jsonl` and `runs/episodes-{base,prompt-half-gguf}.jsonl`.

## How it works

The adapter learns **34,816 values in 17 existing reserved-token embeddings**. Ordinary embeddings, the transformer and the output head stay frozen. The chat template inserts those 17 input tokens before the current assistant turn. Standard MLX LoRA loading and fusion handle the embedding adapter.

The corpus pairs compact, checked Flint-generated thinking with its original final replies. Training combines cross-entropy on compact thinking with KL divergence on final-answer probabilities: the student after short thinking should predict like frozen Flint after its original full thinking. Final replies receive no shortening objective. This uses Flint's own probabilities; it is not a guarantee of the globally shortest equivalent reasoning.

The selected corpus has **378 pairs: 335 train, 43 validation**—190 math, 148 code, 26 with the real OpenClaw context, and 14 reviewed ordinary explanations, emails, plans and creative responses. Ordinary prose receives eight times the sampling weight. Attention QLoRA and output-head experiments remain available through `train` and `train-head`; the learned prefix performed best in the combined development checks.

## Reproduce

Requires Apple Silicon, `uv`, Git and LM Studio. This machine has 128 GB unified memory. The training checkpoint uses about 22.6 GB, the GGUF 42.6 GB, and export needs roughly 170 GB temporary free space. Unload LM Studio models before training to avoid memory pressure.

```sh
./run.sh setup
./run.sh setup-gguf
./run.sh download
./run.sh test
./run.sh cache-features data/corpus-paired.jsonl --original-reasoning --max-tokens 16384 --output runs/native-features.safetensors
./run.sh train-prompt data/corpus-paired.jsonl --teacher-features runs/native-features.safetensors --strength .5 --output adapters/terse
./run.sh export --adapter adapters/terse --output models/flint-terse-Q8_0.gguf --install
```

Training defaults reproduce the selected recipe: 100 updates, accumulation 4, learning rate .001, answer KL weight 1 and seed 42. `--strength .5` blends the learned prefix halfway toward its initialization. Choose new output paths when rerunning. Dependencies and model revisions are pinned.

Export merges the adapter, restores HF expert tensors and uses the pinned llama.cpp converter. The changed embedding matrix stays FP16; other matrix weights use Q8. Temporary full-precision shards are removed afterward. Q8 limits additional quantization error; it cannot recover precision already lost in the INT4 source. The exported file has been loaded and exercised in normal LM Studio.

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

`build-corpus` imports [GSM8K](https://github.com/openai/grade-school-math) and [sanitized MBPP](https://github.com/google-research/google-research/blob/master/mbpp/sanitized-mbpp.json). `data/prose-prompts.jsonl` and `data/tool-prompts.jsonl` provide additional collection tasks. The original seed set alone was insufficient; the measured result above comes from the reviewed paired corpus.

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
