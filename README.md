# flint-terse

Compress Flint's `<think>` into useful working notes and equations while keeping final replies fluent and complete. Train small adapters without adding text to the prompt.

## Train and deploy from the original GGUF

```sh
./run.sh finetune
# Or choose another Flint checkpoint:
./run.sh finetune /path/to/flint.gguf --output release/my-flint-terse
./run.sh finetune owner/repository:filename.gguf --revision COMMIT
```

Requires Apple Silicon, `uv`, Node.js/npm, and regular LM Studio. Missing Python and SDK environments are installed automatically. Training reads the GGUF coefficients into MLX for differentiation; deployment uses the original GGUF in LM Studio's llama.cpp runtime. It does not use the old MLX checkpoint or requantize the base. Training needs more memory than deployment because the lossless working representation and gradients occupy extra RAM.

The default is [`lmstudio-community/flint-GGUF`](https://huggingface.co/lmstudio-community/flint-GGUF), `flint-Q4_K_M.gguf`, revision `a14b4396552988f5bf1427c2ab23f3ade4b683a8`. Its SHA-256 is `20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e`. The script searches local `models/` and LM Studio folders, verifies a matching file, and downloads only if necessary. `--model-dir` adds another search directory. Other checkpoints must currently use Flint's architecture and supported GGUF quantizations (Q4_K, Q6_K, Q8_0, F16 or F32).

The pipeline caches the frozen teacher, trains the opening-token adapter, exports it separately, checks the base checksum, and compares base versus adapter in LM Studio on `data/release-eval.jsonl`. It fails if the candidate loses aggregate checked accuracy or completions, or fails to shorten thinking on shared correct answers. Passing this finite test does not prove unchanged quality on every task; inspect final replies and task-level regressions too. The GGUF-trained release is still being validated.

Output goes to `release/flint-terse`: the FP32 adapter in GGUF and Safetensors, its configuration identifying the exact base, and a Jinja override removing only `| replace("\x00", "")`. No instruction is appended to the prompt. The base GGUF stays untouched. The loaded model identifier is **`flint-terse`**.

For redeployment, copy that small directory, download/import the matching original GGUF in LM Studio, and run:

```sh
./run.sh setup-lmstudio
./run.sh load-lmstudio
# If the model has a different library key:
./run.sh load-lmstudio --model MODEL_KEY --adapter release/flint-terse
```

## Earlier MLX experiment

The results below belong to `adapters/opening-pruned`, trained on the separately downloaded `RecursiveSI/flint-int4` MLX checkpoint. They do not establish the quality of the new GGUF-trained adapter. Its old Q4_K_M and Q8_0 exports are merged, requantized derivatives. The old exact MLX checkpoint plus adapter remains runnable through `./run.sh serve` with API model `default_model`.

**Useful compression, with remaining quality problems.** On 100 previously unseen math/code tasks, the opening-only adapter scored 82/100 versus native Flint's 84/100. Thinking fell 70.2% on the 75 tasks both answered correctly. This small comparison does not establish unchanged accuracy, and the model still sometimes recaps or drafts its answer twice.

These matched MLX captures use the same INT4 checkpoint, temperature 1, top-p .95, top-k 20, independent per-task seeds, batch size 8 and a 4,096-token budget:

| Measure | Base Flint | Opening-only adapter |
|---|---:|---:|
| Correct math/code answers | 84/100 | 82/100 |
| Complete responses | 89/100 | 95/100 |
| Median thinking tokens | 362 | 54.5 |
| Total output tokens | 97,159 | 35,322 |
| OpenClaw checks | 16/24 | 16/24 |

The shared-correct reduction was 91% for math but only 19% for code. Nine base-correct answers became wrong and seven base failures became correct. One counted regression was numeric formatting (`$800` instead of `800`). Captures are `runs/release-{base,opening-pruned}.jsonl` and `runs/openclaw-{native-base,opening-pruned}.jsonl`.

On ten ordinary development prompts, thinking fell from 10,150 to 785 tokens (92%), with all replies complete and in English. Results on six unseen conversational tasks were mixed: all completed versus five for base, but thinking increased from 6,141 to 7,069 tokens. The bakery response added an unwanted keyword list, and the capacity explanation used the wrong percentage formula despite reaching the right worker count. A constrained email also missed its word count. Final-answer preservation is therefore still a training objective, not a guarantee. Halving the adapter caused a loop on “Hi!” and was rejected.

The Q8 GGUF completed all 16 ordinary/conversational smoke prompts in LM Studio. The exact prompt “write some python code” used three thinking tokens and returned a Python function with a normal explanation. Interval-merging passed its four checks; OpenClaw replay passed 5/6 episodes, failing when it reran completed tests instead of returning the requested JSON. Manual review also found incorrect printed outputs in the nested-list explanation, a missed email word count, and an inconsistent migration/rollback sequence. Captures are `runs/direct-experimental-{ordinary,chat,episodes}.jsonl`. These deployment checks are separate from the 100-task MLX accuracy result; runtime, quantization and seed scheduling differ.

To load the earlier merged export:

```sh
lms load 'flint-terse-direct-experimental@q4_k_m' --identifier flint-terse --context-length 65536 --parallel 1 --gpu max --no-speculative-draft-mtp -y
```

The Q4_K_M file is **24.61 GB**, versus 42.64 GB for Q8_0. Its adapted embedding stays FP16 and is byte-identical to the Q8 version's; other matrices use Q4_K/Q6_K. Copy `models/flint-terse-direct-experimental-Q4_K_M.gguf` to the other machine and import it into LM Studio. No separate adapter is needed.

LM Studio's memory estimates with full GPU offload, one prediction slot and no speculative draft:

| Context | Q4_K_M | Q8_0 |
|---|---:|---:|
| 65,536 tokens | 26.94 GiB | 44.23 GiB |
| 131,072 tokens | 30.26 GiB | 47.56 GiB |

These are model/runtime estimates, not measured peak usage or guarantees of long-context answer quality. Leave room for macOS and other apps. Q4 testing at 64K context completed all 16 short conversational prompts and passed the four interval-merging assertions. Ordinary-prompt thinking totaled 2,585 tokens versus Q8's 859, mostly due to one long lists explanation. Percentage explanations and reply-length constraints still failed review. Captures are `runs/direct-q4-{ordinary,chat,episodes}.jsonl`.

A matched LM Studio comparison of the exported GGUFs on the same 100 tasks scored **Q8 84/100, Q4 82/100**: math 50→47/60, code 34→35/40. Both used 64K context, one prediction slot, no speculative draft, temperature 1, top-p .95, top-k 20, seed 42 per request and a 4,096-token output budget. Q4 used 39,711 thinking tokens versus Q8's 30,096 (+31.9%); on the 78 tasks both answered correctly, it used 9.5% more. Six Q8 successes became failures and four failures became successes. Scores include strict formatting checks. This single run shows a close score, not equivalent behavior.

Matched 64K OpenClaw replay remained **Q8 5/6, Q4 4/6**. Q8 unnecessarily reran finished tests; Q4 added an extra file check and wrapped required JSON in Markdown. Captures are `runs/release-gguf-{q8,q4}.jsonl` and `runs/release-gguf-{q8,q4}-episodes.jsonl`. These GGUF results are separate from the MLX comparison above, despite the coincidentally identical aggregate scores.

## How it works

The deployed experiment learns just 2,048 values in the existing `<think>` input embedding: 8 KiB in float32. The saved adapter is about 609 KB because it also contains a fixed token selector. The `</think>` embedding and all other weights remain frozen. It trains on 220 reviewed pairs (169 train, 51 validation), selecting update 50 of 100 by validation loss. The learned update's L2 norm equals the original token vector's norm: a small parameter count does not mean a small change at that control point.

The training tools use Flint's probabilities in these ways:

- Supervised loss teaches reviewed working notes. Some difficult examples retain longer reasoning.
- KL divergence anchors final-answer probabilities to a frozen teacher conditioned on the original thinking. Verified answers also receive supervised loss, with extra weight on their opening to preserve language and format.
- The optional attention trainer uses DPO to prefer a correct complete response over Flint's own checked mistake, relative to the frozen starting model's likelihood ratio. It uses the entire response, including thinking. The deployed opening-only adapter uses supervised loss and KL, not DPO.
- `--compact-preference-weight` optionally adds a second DPO comparison: reviewed compact thinking versus the original verbose thinking, with identical final replies. This teaches brevity directly; it is separate from the correctness comparison.

Final replies have no shortening objective. Probability anchoring encourages a small behavioral change; it cannot guarantee unchanged accuracy or the shortest equivalent reasoning.

The separate attention-training corpus has **410 reviewed pairs: 365 train, 45 validation**—189 math, 147 code, 26 OpenClaw and 48 prose. Of these, 137 training and 19 validation examples include a checked incorrect response. `data/adaptive-sampled.jsonl` contains 59 reliable self-generated traces with their matching answers; `data/adaptive-native.jsonl` contains 351 reviewed native pairs. Code tests and numeric checks guide selection, followed by review. More examples alone do not establish quality.

There are also 349 shorter-versus-original pairs (307 train, 42 validation). The compact preference stage uses weight `.1`, alongside correctness preference weight `1`, answer KL `2` and answer supervision `1`. Both comparisons use complete responses and a frozen reference likelihood ratio.

`data/expanded-prompts.jsonl` adds 880 new collection prompts: 300 math, 80 code and 500 prose. They are disjoint from the existing task IDs and exact prompts. The prose prompts come from the Apache-2.0 Smol-Magpie-Ultra, Smol-rewrite and Smol-summarize subsets of [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk); only prompts are imported, and Flint generates its own responses. All 880 captures are saved in `runs/expanded-native*.jsonl`: 830 completed and 333/380 math/code checks passed. Raw captures need review before training.

`data/pruned-pilot.jsonl` adds 40 reviewed pairs from a 120-example pilot: 23 math, 13 code and four prose (34 train, six validation). Final replies are unchanged. The 36 math/code examples also passed answer regeneration after the shortened thinking. The prose examples passed review and likelihood checks. All remain separate from the release sets.

## Train and evaluate

Training requires Apple Silicon, `uv` and Git; GGUF export also needs CMake and a C++ compiler. This machine has 128 GB unified memory. The source checkpoint uses about 22.6 GB. Export needs roughly 170 GB temporary free space for Q8 or 250 GB for Q4's FP16 intermediate; running the finished GGUF does not need that scratch space. Unload LM Studio models before training.

To reproduce the opening-only experiment with fresh output paths:

```sh
./run.sh setup
./run.sh setup-gguf
./run.sh download
./run.sh test
./run.sh cache-features data/on-policy-pilot.jsonl data/prose-extra-paired.jsonl data/pruned-pilot.jsonl \
  --original-reasoning --max-tokens 16384 --output runs/opening-teacher-new.safetensors
./run.sh train-prompt data/on-policy-pilot.jsonl data/prose-extra-paired.jsonl data/pruned-pilot.jsonl \
  --teacher-features runs/opening-teacher-new.safetensors --think-opening \
  --answer-kl 8 --answer-weight 1 --start-weight 3 --steps 100 \
  --max-tokens 16384 --output adapters/opening-new
./run.sh batch-capture data/direct-eval.jsonl --adapter adapters/opening-new --independent-seeds --batch-size 8 --max-tokens 4096 --output runs/direct-new.jsonl
./run.sh batch-capture data/retention-eval.jsonl --adapter adapters/opening-new --independent-seeds --batch-size 8 --max-tokens 4096 --output runs/retention-new.jsonl
```

`--steps 100` means 100 optimizer updates, with four accumulated examples each. Validation selects the saved checkpoint. `--think-tokens` instead trains both native thinking boundaries. Earlier prefix and output-head experiments remain available through `train-prompt` and `train-head`.

The optional attention/DPO recipe trains 557,056 values in rank-8 query/value updates across the last eight layers. It caches activations from the frozen first 41 layers and recomputes the trainable layers with their full context. Using the saved `adapters/think-tokens` teacher for its sampled responses:

```sh
./run.sh cache-features data/adaptive-sampled.jsonl --adapter adapters/think-tokens --original-reasoning --max-tokens 16384 --output runs/sampled-teacher-new.safetensors
./run.sh cache-features data/adaptive-native.jsonl --original-reasoning --max-tokens 16384 --output runs/native-teacher-new.safetensors
./run.sh train --paired data/adaptive-sampled.jsonl data/adaptive-native.jsonl \
  --teacher-features runs/sampled-teacher-new.safetensors runs/native-teacher-new.safetensors \
  --preference-weight 1 --preference-beta .1 --answer-weight 1 --kl 2 --start-weight 3 \
  --layers 8 --learning-rate .000005 --iters 800 --max-tokens 16384 --output adapters/correctness-new
```

Here `--iters 800 --accumulate 4` means 200 optimizer updates. To add compact preferences, repeat this command with `--resume adapters/correctness-new --compact-preference-weight .1` and a new `--output`. These attention variants have not reliably removed recaps while preserving accuracy.

After reviewing ordinary replies, test accuracy with matched settings below. Only then export and load a candidate:

```sh
./run.sh export --adapter adapters/opening-new --outtype q4_k_m --output models/flint-terse-new-Q4_K_M.gguf --install
lms load flint-terse-new --identifier flint-terse --context-length 16384 --parallel 1 --gpu max --no-speculative-draft-mtp -y
```

Export merges the adapter, restores HF expert tensors and uses the pinned llama.cpp converter. Adapted matrices stay FP16. `--outtype q4_k_m` builds an FP16 intermediate from the INT4 source, then quantizes other matrices with llama-quantize. `--outtype q8_0` remains available. Temporary shards and intermediate files are removed afterward. Neither format recovers precision already lost in the source; GGUF's different quantization can change responses. Each new GGUF needs its own LM Studio check.

## Build more corpus

To collect a broader native corpus:

```sh
./run.sh build-corpus --expand data/expanded-prompts-new.jsonl
for offset in 0 200 400 600 800; do
  ./run.sh batch-capture data/expanded-prompts-new.jsonl --offset "$offset" --limit 200 --batch-size 16 --max-tokens 4096 --independent-seeds --output "runs/expanded-native-new-$offset.jsonl"
done
```

The expansion command skips task IDs and exact prompts already present in `data/`. Separate 200-prompt runs avoid the Metal resource limit encountered during a single 880-prompt collection; `--offset` preserves independent per-task seeds. Review the captured responses before using them as training targets. Commands such as `prune` and `cache-features` accept multiple input files.

To remove redundant passages while keeping Flint's wording and final replies:

```sh
./run.sh prune runs/native.jsonl --build --output runs/edit-prompts.jsonl
./run.sh batch-capture runs/edit-prompts.jsonl --temperature 0 --batch-size 16 --max-tokens 4096 --output runs/edits.jsonl
./run.sh prune runs/native.jsonl --selections runs/edits.jsonl --reasoning-slack 1.5 --answer-start-slack .3 --output runs/pruned.jsonl
./run.sh batch-capture runs/pruned.jsonl --continue-thinking --independent-seeds --batch-size 8 --max-tokens 4096 --output runs/pruned-answers.jsonl
```

`prune` compares the likelihood of retained tokens under the original and shortened contexts, including the closing thinking tag. It also limits the increase in final-answer NLL to `.1` nats/token and checks the first eight answer tokens separately. Failed candidates retain the native trace; edited rows need review. These thresholds are experimental, not a guarantee of equivalent reasoning. The continuation check regenerates answers after fixed thinking, so it checks the corpus rather than a trained model's ability to think briefly.

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

To compare the installed GGUF variants under the same settings:

```sh
for quant in q8_0 q4_k_m; do
  lms load "flint-terse-direct-experimental@$quant" --identifier "flint-$quant-eval" --context-length 65536 --parallel 1 --gpu max --no-speculative-draft-mtp -y
  ./run.sh capture data/release-eval.jsonl --api http://localhost:1234/v1 --api-chat --model "flint-$quant-eval" --temperature 1 --seed 42 --max-tokens 4096 --output "runs/gguf-$quant-new.jsonl"
  lms unload "flint-$quant-eval"
done
./run.sh compare runs/gguf-q8_0-new.jsonl runs/gguf-q4_k_m-new.jsonl --cross-runtime
```

For MLX adapters and OpenClaw checks:

```sh
./run.sh batch-capture data/holdout.jsonl --batch-size 8 --max-tokens 4096 --independent-seeds --output runs/base.jsonl
./run.sh batch-capture data/holdout.jsonl --batch-size 8 --max-tokens 4096 --independent-seeds --adapter adapters/terse --output runs/candidate.jsonl
./run.sh compare runs/base.jsonl runs/candidate.jsonl
./run.sh openclaw condition data/openclaw-eval.jsonl --output runs/openclaw-eval-prompts.jsonl
./run.sh capture runs/openclaw-eval-prompts.jsonl --api http://localhost:1234/v1 --api-chat --model flint-terse --temperature 1 --max-tokens 4096 --output runs/openclaw-terse.jsonl
./run.sh replay data/openclaw-episodes.jsonl --model flint-terse --temperature 1 --output runs/episodes-terse.jsonl
```

The 159-task holdout, 100-task final set, 50-task fresh set and now the release sets have all been inspected. Future tuning needs a fresh evaluation set. `--samples` greater than one is for corpus selection only. Use `compare --cross-runtime` for deployment comparisons with different model identifiers or quantization.

`openclaw sync` reads the authorized remote system instructions and exec/process schemas over SSH. Episode replay feeds actual model responses back into the conversation with fixed tool observations. Generated tool calls are checked without execution.

## Research

[DPO](https://arxiv.org/abs/2305.18290) supplies the complete-response preference objective. [ConPress](https://arxiv.org/abs/2602.01472), [ShorterBetter](https://arxiv.org/abs/2504.21370), [prompt tuning](https://arxiv.org/abs/2104.08691), and [Truncated-Reasoning Self-Distillation](https://arxiv.org/abs/2603.13274) informed the other experiments. This combines supervised training, answer KL and correctness DPO; it does not implement GRPO or reproduce those papers' results. There is no claim of a universally best method. The differentiable attention-sink implementation and shortest-correct sampling borrow useful ideas from `../flint-terse2`.

[STOP](https://arxiv.org/html/2605.13165v1) and [TokenSkip](https://arxiv.org/abs/2502.12067) motivated pruning native traces. Our passage selector additionally compares retained-token likelihoods and checks answer continuations; it is not a reproduction of either paper.

[Flint's model card](https://huggingface.co/RecursiveSI/flint) recommends temperature 1, top-p .95 and top-k 20. Our 4,096-token comparisons measure a practical local budget, below its recommendation for hard problems.
