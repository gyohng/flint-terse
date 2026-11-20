---
license: apache-2.0
base_model: RecursiveSI/flint-int4
base_model_relation: adapter
library_name: mlx
pipeline_tag: text-generation
tags:
- lora
- mlx
- flint
- reasoning
- code
---

# {TITLE} (MLX)

A LoRA adapter for [Recursive Flint](https://huggingface.co/RecursiveSI/flint) that makes it think briefly and in shorthand inside `<think>` while keeping its final answers. This repository holds the MLX LoRA for mlx-lm on Apple Silicon, applied on top of the unchanged MLX int4 base. The merged GGUF model for LM Studio and llama.cpp is [{USER}/{NAME}](https://huggingface.co/{USER}/{NAME}); the GGUF LoRA is [{USER}/{NAME}-lora](https://huggingface.co/{USER}/{NAME}-lora).

```
"Hi!"   base:   123 thinking tokens  ("The user has just said "Hi!" — a simple greeting. ...")
        terse:   15 thinking tokens  User greets; respond friendly; ask how can help; brief; casual.
```

{SUMMARY}

## Results

Measured on the unchanged int4 base in llama.cpp. Scores are the mean of three seeds with one sample per task (temperature 1, top-p 0.95, top-k 20). The thinking reduction is relative to the base on the same tasks.

| | Code (110 tasks) | Math (120 tasks) | Less thinking: code | math | prose & chat |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flint int4 (base) | 102.3 | 103.3 | — | — | — |
| [flint-terse](https://huggingface.co/{USER}/flint-terse-mlx) | **102.7** | **109.7** | 62% | 79% | 85% |
| [flint-terse-minimalist](https://huggingface.co/{USER}/flint-terse-minimalist-mlx) | 98.0 | 108.0 | **81%** | **86%** | 85% |

**flint-terse** keeps the base's coding accuracy. **flint-terse-minimalist** thinks the least, at about 4 code tasks lower. Final replies stay complete, and no `<think>` tags leak into them.

## Files

| File | Use |
| --- | --- |
| `adapters.safetensors`, `adapter_config.json` | MLX LoRA for mlx-lm ({MLX_SIZE}) |

## Base model

[`RecursiveSI/flint-int4`](https://huggingface.co/RecursiveSI/flint-int4), the MLX export of Flint int4. The adapter was trained on the same int4 integer codes, in their GGUF form. Other quantizations are different checkpoints and are not supported.

## Usage

These adapters are for mlx-lm. LM Studio's MLX engine does not support LoRA adapters; for LM Studio, download the merged GGUF model [{USER}/{NAME}](https://huggingface.co/{USER}/{NAME}).


```sh
hf download {USER}/{NAME}-mlx --local-dir {NAME}-mlx
mlx_lm.generate --model RecursiveSI/flint-int4 --adapter-path {NAME}-mlx \
  --temp 1 --top-p 0.95 --top-k 20 --max-tokens 4096 --prompt "Write a function that merges overlapping intervals."
# or an OpenAI-compatible server:
mlx_lm.server --model RecursiveSI/flint-int4 --adapter-path {NAME}-mlx
```

Recommended sampling is temperature 1, top-p 0.95 and top-k 20, the base model's defaults.
{EXTRA}
## How it was made

Training ran directly on the int4 coefficients with rank-32 FP32 LoRA on attention, the shared expert and the dense MLPs. The model learned from its own answers by reinforcement learning: an answer earned reward only if it passed its tests and fitted a shrinking thinking budget. A short supervised pass then taught terse thinking for prose and chat. Code, data and a one-script reproduction are at **https://github.com/gyohng/flint-terse**.

## License

Apache-2.0, the same licence as Flint. This fine-tune adds no further conditions. Flint is by [Recursive](https://huggingface.co/RecursiveSI).
