# flint-terse

Shorter working notes inside `<think>`, with normal, complete final replies. Deployment uses the **unchanged official Flint GGUF plus a separate FP32 LoRA adapter**. No instruction is appended to the user's message.

The packaged adapter is the best balanced result tested here. **It does not achieve the requested >70% thinking reduction consistently**, especially for conversational coding. The last expanded-corpus run produced thinking-delimiter leakage and duplicated text in final replies, so it was rejected and its evaluation stopped.

## Use in LM Studio

Requires Apple Silicon, `uv`, Node.js/npm and regular LM Studio with its `lms` command available.

```sh
./run.sh deploy
# Optional longer context:
./run.sh deploy --context 131072
```

This finds the exact base in local `models/` or LM Studio folders, downloads it if missing, and loads the packaged adapter as **flint-terse**. `--model-dir PATH` adds a search directory. Deployment installs only the lightweight download and LM Studio dependencies; it does not require the MLX training environment.

The portable bundle is in `release/flint-terse/`: the GGUF adapter, identical FP32 factors in Safetensors, configuration and Jinja override. The GGUF adapter is 13,661,152 bytes. Copy the bundle with these scripts or clone the repo on another Mac, then run `deploy`. The base must match exactly; a different quantization is a different checkpoint.

LM Studio's API identifier is `flint-terse`; its GUI may still display the base model's name. The launcher installs the `flint-terse` sampling preset. Select it in the GUI for the tested sampling settings. Rerun `deploy` after unloading or restarting: the adapter attachment is not persistent.

## Base and precision

Default: [`RecursiveSI/flint-int4-gguf`](https://huggingface.co/RecursiveSI/flint-int4-gguf), **flint-int4-q4_1.gguf**, revision `9c47151036bc2c34331142dda04488bceab2bc9f`.

SHA-256: `43c07baccdba42eb7528a9630ee15699b1942c297291dc1b39a061f508ee1a4b`.

This is the publisher's GGUF counterpart of `RecursiveSI/flint-int4`, distinct from the Q4_K_M conversion. Its integer codes match the MLX export; some BF16 scales and offsets were rounded to FP16 by the publisher. Training uses the actual GGUF coefficients. The base is never merged, rewritten or requantized, and the adapter remains FP32. Export verifies every adapter coefficient after the layout conversion.

The external Jinja override only removes the incompatible `replace("\x00", "")` operation. The downloaded GGUF stays untouched.

Training has run on a 128 GB Mac. LM Studio estimates 28.27 GiB at 64K context, 31.59 GiB at 128K and 38.23 GiB at 256K, with full GPU offload, one prediction slot and no MTP. These are load estimates, not measured peak memory or long-context quality results.

## Results

| Test | Unchanged base | Packaged adapter |
| --- | ---: | ---: |
| Math | 99/120 | 105/120 |
| Code | 73/80 | 72/80 |
| Combined math/code | 172/200 | 177/200 |
| Conversational code | 6/6 | 6/6 |
| OpenClaw scenarios | 6/6 | 6/6 |

Thinking falls by **69.9% on shared correct math/code answers**, **40.7% on 14 ordinary requests completed by both models**, and **6.0% on the six conversational code tasks**. Ordinary prose and coding requests can still produce long internal narration. The packaged adapter is the unchanged `gguf-attention-055` candidate.

Evaluations use temperature 1, top-p .95, top-k 20, min-p 0, repetition penalty 1, seed 42 per request, a 4,096-token output budget and 64K context. The two 100-task sets have been used for candidate selection: these are development results, not untouched confirmation tests. The saved baseline is reused when checkpoint, tasks and settings match.

## Retrain

```sh
./run.sh finetune
# Other Flint checkpoints, trained against their own unchanged coefficients:
./run.sh finetune /path/to/flint.gguf --output adapters/my-flint-terse
./run.sh finetune owner/repository:filename.gguf --revision COMMIT
# Require the original compression target before automatic deployment:
./run.sh finetune --output adapters/strict --min-thinking-reduction 70
```

The default reproduces the balanced recipe: rank-8 query/value LoRA across 49 layers, 200 optimizer updates with four-example accumulation, learning rate 0.00002, answer KL weight 8 and exported strength 0.55. It uses the 410 reviewed pairs in `data/adaptive-sampled.jsonl` and `data/adaptive-native.jsonl`. MLX supplies differentiation; LM Studio's llama.cpp runtime runs the exported adapter.

The pipeline finds or downloads the base, trains, verifies and exports the separate adapter, evaluates it and loads passing results as `flint-terse`. Existing output folders are not overwritten. Accuracy may drop by at most two percentage points overall and per task family; reply completion and delimiter checks must also pass. **The default compression floor is 0% to permit the selected balanced result.** Set `--min-thinking-reduction 70` to enforce >70% reduction on both shared correct answers and ordinary completed replies. New training runs are not guaranteed to reproduce the saved scores.

Explicit `--data`, `--steps`, `--layers`, `--rank` and `--adapter-scale` override the recipe. Other experimental methods remain available through `./run.sh finetune --help`. Supported inputs use Flint's architecture and Q4_1, Q4_K, Q6_K, Q8_0, F16 or F32 tensors.

## Corpus

Supervised loss teaches short calculations, algorithms and necessary checks. Probability matching protects final replies, which have no shortening objective. The method is LoRA on a frozen quantized base, not a claim of a universally best fine-tuning method.

The final experimental corpus, `data/final-paired.jsonl`, contains 699 examples, including 320 checked code answers across 212 distinct tasks. It restores useful constraint notes for writing instead of teaching empty thoughts. Its 76 new conversational pairs passed native code checks, a likelihood screen against the exact GGUF, and code checks after regenerating answers from the short notes. This corpus did **not** produce the selected release.

To extend the corpus:

1. Collect native responses with `capture` or `batch-capture`; keep evaluation tasks separate.
2. Use `rewrite`, or `prune --build` followed by `prune --selections`, to propose shorter notes and check answer likelihood. `distill build/select/pair` offers a demonstration-based alternative.
3. Regenerate answers with `batch-capture --continue-thinking` and run their checks. This verifies supplied notes; it is not a free-generation benchmark.
4. Review the notes and final replies, then mark accepted rows `approved=true`. Test success and probability agreement alone do not establish correctness.

`data/code-prose-extra-prompts.jsonl` contains the 180 additional conversational prompts; recreate it at a new path with `./run.sh build-corpus --code-prose OUTPUT`. Corpus collection, editing and scoring accept `--model` and default to the same pinned GGUF.

Sources: [GSM8K](https://github.com/openai/grade-school-math), Google's [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) (CC BY 4.0; prompts adapted), and [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk). [TokenSkip](https://arxiv.org/abs/2502.12067) informed trace pruning; optional preference training follows [DPO](https://arxiv.org/abs/2305.18290). The current recipe uses fixed training prefixes, not full online distillation or GRPO. Attention-sink handling and shortest-correct sampling borrow ideas from `../flint-terse2`.

## Check the scripts

```sh
./run.sh test
```

Dependencies are pinned in `requirements.txt`. The tests cover exact GGUF coefficients and gradients, adapter export, probability retention, corpus handling and evaluation checks. Every command accepts `--help`.
