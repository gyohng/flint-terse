# flint-terse

LoRA adapters that make [Recursive Flint](https://huggingface.co/RecursiveSI/flint) think briefly, in shorthand, inside `<think>` while keeping its final answers. Each adapter is a small FP32 GGUF LoRA for the unchanged official int4 GGUF and runs in LM Studio and llama.cpp. The base is never merged, rewritten or requantized, and nothing is added to the user's prompt.

```
"Hi!"     base:  123 thinking tokens ("The user has just said "Hi!" — a simple greeting. ...")
          terse:  15 tokens   User greets; respond friendly; ask how can help; brief; casual.
```

## Models

| Adapter | Use it for | Code (110 tasks) | Math (120) | Thinking cut: code / math / prose |
| --- | --- | ---: | ---: | --- |
| unchanged base | | 102.3 | 103.3 | — |
| **flint-terse** | coding, with the base's accuracy | **102.7** | **109.7** | 62% / 79% / 85% |
| **flint-terse-minimalist** | the least thinking | 98.0 | 108.0 | 81% / 86% / 85% |

Scores are means over three seeds (42, 7, 11), one sample per task. Thinking cuts are measured on tasks both models solve (graded sets) or both complete (ungraded sets). Neither adapter leaked a `<think>` delimiter into a final reply. Adapter weights will be published on Hugging Face; `release/<name>/` holds each adapter's configuration, the exact base it requires and the chat template.

Both adapters need the exact base: [`RecursiveSI/flint-int4-gguf`](https://huggingface.co/RecursiveSI/flint-int4-gguf), file `flint-int4-q4_1.gguf`, revision `9c47151036bc2c34331142dda04488bceab2bc9f`, SHA-256 `43c07baccdba42eb7528a9630ee15699b1942c297291dc1b39a061f508ee1a4b`. A different quantization is a different checkpoint.

## Use in LM Studio (Mac)

Requires Apple Silicon, `uv`, Node.js/npm and LM Studio with its `lms` command. Put the adapter's `adapter.gguf` into its `release/<name>/` directory, then:

```sh
./run.sh deploy                                         # flint-terse
./run.sh deploy --adapter release/flint-terse-minimalist
./run.sh deploy --context 131072                        # longer context
```

`deploy` finds or downloads the exact base, imports it, loads it with the adapter as `flint-terse` and installs the tested sampling preset (temperature 1, top-p .95, top-k 20). The Jinja override only removes the base template's `replace("\x00", "")`, which LM Studio's engine does not support. Rerun `deploy` after unloading or restarting.

The adapters were verified in CUDA llama.cpp, the engine LM Studio uses, on the int4 base. flint-terse-minimalist also uses llama.cpp's per-expert LoRA path. It is verified there but untested in LM Studio's bundled llama.cpp. Their log-probabilities match the training model: mean |Δlogp| 0.01–0.04 and 96–100% top-token agreement (`check_port.py`).

## Reproduce the training (NVIDIA GPU)

Tested on an NVIDIA GB10 (119 GB unified memory, aarch64, Ubuntu 24.04, driver 580). Training peaks at about 40 GB, and the llama.cpp sampling server uses about 45 GB with 32 slots. They never run at the same time.

```sh
PY=python3.15 scripts/setup.sh        # CUDA torch + toolkit, pinned llama.cpp, int4 base (26 GB)
PY=python3.15 scripts/reproduce.sh    # baseline, both adapters, evaluation (~20 h)
```

`setup.sh` installs into the given interpreter's environment; it uses `uv` when available. Set `CUDA_ARCH` for GPUs other than GB10 (for example `90` for H100). `reproduce.sh` writes `release/flint-terse/adapter.gguf` and `release/flint-terse-minimalist/adapter.gguf`. RL sampling is stochastic, so expect scores within a few tasks of the tables.

The individual steps:

```sh
scripts/eval.sh base base                                   # baseline -> runs/spark-base-*.jsonl
PROMPTS=300 LR=5e-5 SEED=20 scripts/rl.sh base budget "600 400 300"
scripts/prose_pass.sh adapters/budget-3 runs/budget-3-samples.jsonl flint-terse
PROMPTS=400 LR=2e-5 SEED=40 EXPERT_RANK=8 scripts/rl.sh adapters/flint-terse minimal "300 300"
scripts/eval.sh adapters/flint-terse flint-terse [SEED]     # compare with the base run
```

## How it works

Everything trains against the int4 GGUF coefficients that LM Studio runs. Transformers has no Flint support, so `flint_torch.py` implements it:
- Routed experts stay packed as Q4_1, about 28 GB on the GPU.
- A small CUDA kernel dequantizes them bit-exactly (`q4_kernels.py`), and they run as grouped GEMMs.
- Attention handles learned sinks, the 513-token sliding window and packed sequences.

The LoRA is rank 32, FP32, on attention q/k/v/o, the shared expert and the two dense MLPs. These are all tensors llama.cpp applies adapters to. Optionally, `--expert-rank 8 --expert-layers 16` adds per-expert LoRA on the routed experts' down-projections in the last 16 MoE layers. It is stored as FP16 three-dimensional GGUF tensors, about 170 MB, and llama.cpp applies it through its expert LoRA path. Rows 30–31 show it moves code accuracy off the earlier trade-off curve. Sampling and evaluation use a CUDA llama.cpp server with the GGUF adapter applied (`scripts/serve.sh`, `spark_eval.py`).

1. **Budget RL from the base** (`rl_sample.py`, `rescore_rl.py`, `scripts/rl.sh`):
   - Each iteration samples 8 answers for each of 300 code prompts (MBPP-derived, evaluation tasks excluded).
   - An answer gets reward 1 if its tests pass *and* its thinking fits the budget, otherwise 0.
   - Rewards are centred per group and scaled by the batch standard deviation, then trained with policy gradient over the whole completion. Terse prose rows are replayed with KL holding replies to the base.
   - Budgets go 600 → 400 → 300 tokens.
2. **Prose pass** (`scripts/prose_pass.sh`): one supervised epoch adds terse thinking for prose and chat, from the base's own compressed notes (`gen_corpus.py`). It also replays the adapter's own shortest correct code answers so code holds. This gives **flint-terse**.
3. **Per-expert LoRA and fixed-budget RL**: this step adds rank-8 LoRA on the routed experts' down-projections in the last 16 MoE layers. Two more 300-token iterations with 400 prompts each then give **flint-terse-minimalist**. The GGUF is 339 MB (FP32 base LoRA plus FP16 expert factors). Without the expert LoRA, the same recipe reached only 93.3 on code (rows 21–22).

## What limits code compression

The base itself needs its reasoning on code. With thinking cut off at a fixed length (`budget_probe.py`) it scores:

| Base thinking budget | 0 | 128 | 256 | 512 | unlimited |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code (110 tasks) | 81 | 90 | 92 | 102 | 102 |

Without expert adaptation, every adapter lay on the same accuracy/terseness curve for code (rows 1–29). Math, prose and chat compress 80–90% with no loss. Code kept base accuracy down to about 60% less thinking and lost tasks steadily beyond that. Given the first 25–50% of the base's own reasoning, terse adapters solve 92–94 of 95 base-solved tasks, but they then continue verbosely.

Adding LoRA on the routed experts (rows 30–31) moved off that curve: about 98/110 at 81% less code thinking, against about 93 for the same recipe without it. The gap to the base at that compression is about 4 tasks.

## What was tried

All runs use the int4 base, llama.cpp, temperature 1, top-p .95, top-k 20, seed 42 and 4,096 output tokens unless noted. Scores are out of 110 code tasks and 120 math tasks; the base gets 102 and 104 with this seed. Cuts are thinking tokens saved against the base. "Prose" covers the 36 held-out prose and chat prompts.

| # | Approach | Code | Math | Code cut | Math cut | Prose cut | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | Previous MLX adapter (rank 8, q/v; Mac evals) | 72/80 | 105/120 | ~70% (math+code) | | 41% | conversational code only 6% cut |
| 1 | SFT round 0: base compresses its own thinking, answers checked (1,118 rows) | 84 | 86 | 92% | 94% | 88% | thinking ~20 tokens even on hard tasks |
| 2 | SFT round 1: + verified samples of round 0 (expert iteration) | 87 | 92 | 92% | 96% | 88% | |
| 3 | Round 1 at adapter strength 0.5 | 91 | 105 | 48% | 80% | — | |
| 4 | SFT round 2: budgeted notes keeping checks and traps | 81 | 91 | 87% | 95% | 87% | |
| 5 | RL, length reward normalised per group | 82 | 85 | 98% | 97% | 91% | length noise dominated the signal |
| 6 | RL with a length floor and batch normalisation (RL 1b) | 88 | 96 | 86% | 95% | 89% | best terse SFT/RL start |
| 7 | RL 1b, second iteration | 85 | 96 | 94% | 96% | 89% | plateau |
| 8 | SFT round 3: insight-only notes, empty when not needed | 85 | 90 | 100% | 95% | 86% | learned to never think on code |
| 9 | SFT round 3b: one-line scan notes instead of empty | 85 | 89 | 74% | 95% | 89% | |
| 10 | SFT round 4: adaptive easy/medium/hard notes | 84 | 89 | 89% | 95% | 89% | |
| 11 | SFT round 5: full-shorthand rewrite of every step | 87 | 91 | 45% | 95% | 85% | more thinking did not help |
| 12 | RL 1b at strength 0.35 / 0.5 / 0.6 / 0.65 / 0.7 | 103 / 103 / 100 / 95 / 96 | 107 / 106 / 105 / 103 / 104 | 32 / 30 / 58 / 71 / 74% | 65 / 81 / 89 / 92 / 94% | 8 / 39 / 79 / 76 / 82% | smooth trade-off |
| 13 | **Budget RL from the base, B=600** | 100 | 110 | 40% | 76% | 53% | starting from the accurate base |
| 14 | Budget RL, B=400 | 100 | 111 | 54% | 80% | 55% | |
| 15 | Budget RL, B=300 | 98 | 109 | 70% | 86% | 79% | three seeds: 98 / 94 / 89 |
| 16 | Budget RL, B=200 | — | — | — | — | — | collapsed into repetition |
| 17 | **#15 + prose pass = flint-terse** | **106** | **111** | 60% | 83% | 85% | three seeds: code 106 / 102 / 100, math 111 / 111 / 107 |
| 18 | flint-terse + budget RL at B=250 | 83 | 109 | 85% | 88% | 89% | overshoots |
| 19 | Exact mix 0.7·flint-terse + 0.3·#18 | 95 | 108 | 68% | 85% | 84% | |
| 20 | Exact mix 0.5·flint-terse + 0.5·#18 | 91 | 105 | 77% | 86% | 85% | |
| 21 | flint-terse + fixed-budget RL B=300, iteration 1 | 93 | 109 | 78% | 86% | 87% | |
| 22 | iteration 2 (the first minimalist) | 97 | 107 | 84% | 86% | 89% | three seeds: code 97 / 94 / 89 |
| 23 | iteration 3 | 89 | 104 | 81% | 86% | 88% | oscillates |
| 24 | #22 + prose pass on its shortest answers | 90 | 106 | 92% | 88% | 86% | overshoots |
| 25 | On-policy mini-batch GRPO from flint-terse, B=250, step 4 (100 prompts × 8 per step) | 91 | 108 | 87% | 90% | — | |
| 26 | same, step 8 | 88 | 111 | 85% | 88% | — | no recovery at this terseness |
| 27 | flint-terse + RL with a per-task relative budget (thinking ≤ 30% of the base's own length on that task, floor 64) | 80 | 103 | 94% | 93% | 91% | overshoots hardest; stopped after one iteration |
| 28 | Exact mix 0.6·flint-terse + 0.4·#22 (three seeds) | 97 / 96 / 98 | 109 / 109 / 105 | 70% | 84% | — | code mean 97.0 |
| 29 | Exact mix 0.4·flint-terse + 0.6·#22, rank 64 and its rank-32 SVD refit (six runs) | 94 / 97 / 101 / 96 / 96 / 91 | 105 / 109 / 105 / 108 / 108 / 105 | 75% | 84% | 85% | code mean 95.8; direct 82%, explanations 91% cut |
| 30 | flint-terse + **per-expert LoRA** (rank 8, down_proj, last 16 MoE layers) + fixed-budget RL B=300, iteration 1 | 89 | 110 | 83% | 87% | 86% | same recipe as #21, plus expert capacity |
| 31 | **same, iteration 2 = flint-terse-minimalist** (three seeds) | **100 / 97 / 97** | 110 / 110 / 104 | 81% | 86% | 85% | **code mean 98.0 at 81% less code thinking**: off the earlier curve; direct 77%, explanations 91% cut |
| 32 | flint-terse + per-expert LoRA + fixed-budget RL at B=400, iteration 1 (three seeds) | 97 / 98 / 100 | 107 / 111 / 107 | 66% | 85% | 87% | code mean 98.3; gentler budget does not keep base-level code |
| 33 | same, iteration 2 (three seeds) | 97 / 94 / 96 | 109 / 107 / 106 | 67% | 84% | 88% | code mean 95.7 |

Rows 1–11 come from `gen_corpus.py`, `merge_corpus.py`, `gen_insight.py`, `clean_insight.py`, `build_r4.py` and `spark_train.py`; their corpora are in `data/spark-*`. Rows 5–7 use `rescore_rl.py` without `--budget`. Rows 13–33 use `scripts/rl.sh`, `scripts/prose_pass.sh` and `interp_adapters.py`.

## Repository

| Path | Contents |
| --- | --- |
| `flint_torch.py`, `q4_kernels.py` | PyTorch Flint on the int4 GGUF, CUDA Q4_1 dequant |
| `spark_train.py` | LoRA training: thinking SFT with answer KL, policy gradient |
| `rl_sample.py`, `rescore_rl.py` | grouped sampling with verified rewards; truncation or length-floor advantages |
| `gen_corpus.py`, `merge_corpus.py`, `gen_insight.py`, `clean_insight.py`, `build_r4.py` | on-policy compressed-thinking corpora |
| `build_pool.py`, `build_code_pool.py` | prompt pools (evaluation prompts excluded) |
| `spark_eval.py`, `budget_probe.py`, `check_port.py` | evaluation, base thinking-budget probe, llama.cpp parity check |
| `interp_adapters.py` | exact LoRA interpolation and SVD refit |
| `scripts/` | `setup.sh`, `serve.sh`, `eval.sh`, `rl.sh`, `prose_pass.sh`, `reproduce.sh` |
| `data/` | prompts, evaluation sets, all generated corpora and base-sample logs |
| `release/` | adapter metadata for deployment |
| `export_adapter.py`, `deploy.py`, `load_lmstudio.mjs`, `run.sh` | GGUF export and LM Studio deployment |

The earlier Apple Silicon pipeline (`./run.sh finetune`, `train.py`, `gguf_model.py`, `finetune.py`) still runs. It trained rank-8 q/v LoRA with MLX on the same frozen GGUF coefficients (row 0). `./run.sh test` runs its tests.

## Sources

Prompts and tasks: [GSM8K](https://github.com/openai/grade-school-math), Google's [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) (CC BY 4.0; prompts adapted), [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk).

Methods:
- truncation-reward and budget-schedule RL: DLER ([arXiv 2510.15110](https://arxiv.org/abs/2510.15110)) and ThinkPrune ([arXiv 2504.01296](https://arxiv.org/abs/2504.01296));
- compressed-trace SFT: TokenSkip ([arXiv 2502.12067](https://arxiv.org/abs/2502.12067));
- LoRA placement and RL capacity: [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (Thinking Machines, 2025).
