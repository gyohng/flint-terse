# flint-terse

Goal: teach Flint, via a LoRA adapter, to write terse shorthand (math notation, caveman speak) inside `<think>`, with final replies unchanged. Prompting can't reach the thinking section. `example-vomit.txt` shows a before/after thinking trace. README.md covers the results so far and the MLX pipeline.

## Where it stands
Oct 25–27, on the Mac with MLX: rank-8 q/v LoRA on the frozen `flint-int4` GGUF coefficients, answer KL protecting final replies, 410 reviewed pairs. Released as `release/flint-terse/`. It cuts thinking about 70% on math/code, about 40% on ordinary requests and only about 6% on conversational code. The target is over 70% everywhere. A larger corpus (`data/final-paired.jsonl`) leaked think delimiters and was rejected. Per-experiment logs are in `runs/`, which is not in git.

## Moving to spark-vimu
The trainers (`train.py`, `gguf_model.py`, `finetune.py`) are MLX and run only on Apple Silicon. Corpus, eval and export logic carries over. On Spark, train against the bf16 original instead of quantized GGUF coefficients.

- Box: NVIDIA GB10, 119 GB unified memory, aarch64, Ubuntu 24.04, driver 580. nvidia-smi is at `/run/nvidia/driver/usr/bin/nvidia-smi`. No host CUDA toolkit. Use uv with CUDA torch wheels, or an NGC container.
- Base: `RecursiveSI/flint`. `FlintForCausalLM`, MoE, 40B total / ~2.5B active, bf16, 80 GB, downloading to `models/flint` (log in `models/download.log`). The repo has an `sglang/` folder, so check whether transformers supports the arch before writing a trainer.
- Arch: hybrid full/sliding attention (window 513), attention sinks, 256 experts plus a shared expert, layers 0–1 dense, vocab 150144.
- Deploy target is still LM Studio on the Mac: a GGUF LoRA adapter on the unchanged base (`./run.sh deploy`).

## Next
1. Get a CUDA trainer running on the bf16 base (LoRA, or QLoRA if memory needs it). Reproduce the release recipe as a sanity check.
2. Push compression further: on-policy distillation or RL (GRPO-style) with a length penalty plus KL to the base, focused on ordinary and conversational prompts.
3. Use Spark for corpus generation and eval. On-policy sampling from the base was the slow part on the Mac.
4. Export the adapter to GGUF so it applies to the int4 base used in LM Studio. Verify on the Mac.
