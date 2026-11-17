# flint-terse

Goal: LoRA adapters that make Flint think briefly, in shorthand, inside `<think>` while final replies stay as good as the base's; prompting can't reach the thinking section. `example-vomit.txt` shows a before/after trace. README.md has the models, results, the full experiment table and how to reproduce.

## Constraints
- Train and evaluate against the unchanged int4 GGUF (`RecursiveSI/flint-int4-gguf`, `flint-int4-q4_1.gguf`, pinned revision and SHA-256 in `checkpoint.py`), never bf16. Deliverable: a small FP32 GGUF LoRA adapter that LM Studio/llama.cpp apply over that exact file (`./run.sh deploy`). Adapter weights are not committed; they go to Hugging Face.
- Coding quality comes first; math is a secondary check.
- LoRA targets must be tensors llama.cpp adapts: attention q/k/v/o, shared expert, dense MLPs (no per-expert LoRA).

## Training host (NVIDIA GB10, 119 GB unified memory)
- `scripts/setup.sh` (CUDA torch + pip CUDA 13.0 toolchain, pinned llama.cpp, base download), then `scripts/reproduce.sh`.
- GPU and system share memory: run one memory-heavy job at a time (llama-server with 32 slots ~45 GB, training ~40 GB).
- Logs and samples go to `runs/` (not in git); adapters to `adapters/` (not in git).

## Where it stands
flint-terse: code 102.7/110 (base 102.3) at 62% less code thinking, 79-85% less on math/prose. flint-terse-minimalist: 82-89% less everywhere, code 93.3. Larger code cuts cost accuracy on this model (base: 81/110 with no thinking, 102 only at ~512 tokens). Not yet verified in LM Studio on a Mac.
