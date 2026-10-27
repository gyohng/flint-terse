# flint-terse

Teach Flint to use short working notes and equations inside `<think>`, while keeping final replies fluent and complete. Deployment uses the **unchanged original GGUF plus a separate FP32 adapter**. No instruction is appended to the user's message.

**Experimental:** no candidate yet combines the desired brevity with sufficiently preserved code accuracy. The files in `release/flint-terse` are an older experiment, not a validated release.

## Run

Requires Apple Silicon, `uv`, Node.js/npm and regular LM Studio. The launcher installs missing Python and SDK environments.

```sh
./run.sh finetune
# Other Flint checkpoints:
./run.sh finetune /path/to/flint.gguf --output adapters/my-flint-terse
./run.sh finetune owner/repository:filename.gguf --revision COMMIT
```

The script finds the checkpoint in local `models/` or LM Studio folders, downloads it if missing, trains, exports, evaluates and loads passing candidates as **flint-terse**. `--model-dir PATH` adds a search directory. The current default training recipe is the experimental opening-token adapter; it has not passed the release checks.

The default base is [`RecursiveSI/flint-int4-gguf`](https://huggingface.co/RecursiveSI/flint-int4-gguf), **flint-int4-q4_1.gguf**, revision `9c47151036bc2c34331142dda04488bceab2bc9f`. SHA-256: `43c07baccdba42eb7528a9630ee15699b1942c297291dc1b39a061f508ee1a4b`.

This is the publisher's GGUF counterpart of `RecursiveSI/flint-int4`, distinct from the Q4_K_M conversion. Its integer codes match the MLX export; some BF16 scales/offsets were rounded to FP16 by the publisher. Training uses the actual GGUF coefficients. MLX supplies differentiation; LM Studio's llama.cpp runtime handles deployment. The base is never merged or requantized. Supported inputs use Flint's architecture and Q4_1, Q4_K, Q6_K, Q8_0, F16 or F32 tensors.

Training has been run on a 128 GB Mac. Deployment needs much less memory: LM Studio estimates 28.27 GiB at 64K context, 31.59 GiB at 128K and 38.23 GiB at 256K, with full GPU offload, one prediction slot and no MTP. These are load estimates, not measured peak memory or long-context quality results. Separate adapters range from 609 KB for the opening token to about 18.5 MB for attention plus opening.

## Training and corpus

Supervised loss teaches reviewed short notes. Probability matching preserves final replies; optional same-prefix retention also protects existing math/code behavior. Final replies have no shortening objective. Correctness checks and probability agreement are complementary: a likely short trace can still be wrong.

- `--method opening` learns 2,048 values in the existing `<think>` embedding. The pinned base defaults to the packaged initializer in `data/opening-init`; `--from-scratch` skips it.
- `--method attention --layers 49` trains rank-8 query/value LoRA. `--resume-adapter PATH` continues a compatible adapter at its saved scale; `--train-layers 8` can freeze its earlier layers.
- `--same-prefix-teacher`, with attention resumption, preserves answer probabilities after the compact notes. Rows marked `preserve_only` match the starting policy throughout the response.
- `--teacher-adapter PATH` keeps an earlier probability reference fixed during further training. `--narration-penalty` discourages stock first words (`We`, `The`, `Okay`, `Hmm`, `I`) only in supervised thinking; it adds no inference-time filter or final-reply restriction.
- `--entry-weight` emphasizes a direct first thinking token alongside full-note training. `--entry-only` instead supervises just that token; answer and code/math probability retention still apply.
- `--retention-kl` controls probability retention separately from the prose answer objective. `build-preferences --policy-retention` builds these rows from checked correct captures, keeping the model's actual notes and replies together.
- `--attention-adapter PATH` instead freezes attention and learns only the opening-token adjustment alongside it.
- `--preference-weight` adds DPO against checked wrong responses. `--compact-preference-weight` prefers reviewed short notes over native notes with identical final replies.
- `--teacher-features PATH` reuses matching cached probability features. `--adapter-scale` scales the entire exported update, retaining FP32 matrices.

`--steps` counts optimizer updates, each accumulating four examples. Validation selects the saved checkpoint. `--no-promote` keeps evaluation under `flint-terse-eval`.

The initial attention corpus has 410 reviewed pairs: 189 math, 147 code, 26 OpenClaw and 48 prose. `data/style-retention.jsonl` has 190 prose/tool examples and 401 math/code retention examples. Its 212 code responses pass their tests; 61 also have checked incorrect alternatives for preference training. The 87 prose wording variants keep final replies unchanged. The original few examples were insufficient.

To extend the corpus:

1. Collect several native responses per **training** prompt using `capture` or `batch-capture`. Keep evaluation prompts separate.
2. Use `distill build/select/pair` to collect shorter candidates and pair them with unchanged verified final replies. Prefer short traces only when their answers pass checks.
3. Alternatively, `prune --build` asks for redundant passages to remove. `prune --selections FILE --model GGUF` compares retained-token and answer likelihoods before accepting edits. Regenerate answers after the fixed notes to check them again.
4. Review the notes and replies. Keep useful calculations and edge-case checks; remove narration and prompt repetition. Mark accepted rows `approved=true`.

Rows contain `id`, `messages`, `split`, `assistant`, and a numeric, text, JSON, tool-call or Python-test check. Paired rows also retain `teacher_reasoning`. `build-preferences` attaches checked mistakes; its `--answer-repair` mode pairs sampled student notes with verified final answers for `--answer-only` training.

`build-corpus` imports [GSM8K](https://github.com/openai/grade-school-math) and [sanitized MBPP](https://github.com/google-research/google-research/blob/master/mbpp/sanitized-mbpp.json). Expanded prose prompts come from [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk). Raw captures require review before training. Every command accepts `--help`.

## Evaluation

The two saved 100-task sets contain 120 math and 80 code tasks. Both have been used to compare candidates, so they are development evaluations, not untouched confirmation sets.

| Candidate | Math | Code | Total | Less thinking on shared correct answers |
| --- | ---: | ---: | ---: | ---: |
| Unchanged official GGUF | 99/120 | 73/80 | 172/200 | — |
| Repaired opening token | 103/120 | 63/80 | 166/200 | 79.1% |
| All-layer attention at 55% strength | 105/120 | 72/80 | 177/200 | 69.9% |

The attention candidate restores code accuracy but still narrates ordinary requests. The current work targets that remaining verbosity. These finite comparisons do not establish unchanged quality on other tasks.

Evaluations use temperature 1, top-p .95, top-k 20, min-p 0, repetition penalty 1, seed 42 per request, a 4,096-token output budget and 64K context. Reuse the saved baseline rather than generating it again:

```sh
./run.sh finetune --evaluate-only --output adapters/CANDIDATE \
  --baseline runs/flint-terse-base \
  --eval data/release-eval.jsonl data/confirmation-eval.jsonl \
         data/direct-eval.jsonl data/retention-eval.jsonl data/release-chat.jsonl \
  --no-promote
```

`--resume-eval` continues saved candidate captures. A matching `runs/flint-terse-base` baseline is reused automatically; the pipeline checks tasks, checkpoint and sampling settings. Automatic release checks allow at most two percentage points of accuracy loss overall **and within each task family**, require more than 70% less thinking on shared correct answers **and** completed ordinary replies, and reject completion regressions or leaked thinking tags. Inspect final replies too; the checks do not grade prose quality.

OpenClaw replay checks tool calls without executing them:

```sh
./run.sh replay data/openclaw-episodes.jsonl --model flint-terse-eval \
  --tokenizer models/flint-int4-gguf/flint-int4-q4_1.gguf \
  --temperature 1 --seed 42 --max-tokens 4096 --output runs/candidate-episodes.jsonl
```

`openclaw sync` reads the authorized remote instructions and tool schemas; `openclaw condition` adds them to collection prompts. Reuse the same context for matched comparisons. Run `./run.sh test` for the Python checks.

## Deploy a validated adapter

The output folder contains `adapter.gguf`, `adapters.safetensors`, its configuration, a sampling preset and a Jinja override. Export verifies every adapter coefficient byte-for-byte. The override removes only the incompatible `| replace("\x00", "")` filter.

Copy the folder to another machine and import the matching original GGUF into regular LM Studio:

```sh
./run.sh setup-lmstudio
./run.sh load-lmstudio --adapter /path/to/adapter
# Optional: --model MODEL_KEY --context 65536
```

The loaded identifier is **flint-terse**. Run the loader again after unloading or restarting LM Studio to reattach the adapter. Select its installed sampling preset for GUI chats; API requests should send the same sampling values explicitly. The preset adds no system prompt. `export-adapter` creates portable files without rewriting the base. The older `export` command merges/requantizes and is not this deployment path.

## Research

This is supervised compression with probability retention and optional [DPO](https://arxiv.org/abs/2305.18290), not a claim of a universally best method. [TokenSkip](https://arxiv.org/abs/2502.12067) informed trace pruning; [prompt tuning](https://arxiv.org/abs/2104.08691) informed the small embedding experiments. [On-policy distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) motivates matching a teacher at the student's prefixes; the current recipe uses fixed training prefixes, not full online distillation or GRPO. The attention-sink implementation and shortest-correct sampling borrow ideas from `../flint-terse2`.
