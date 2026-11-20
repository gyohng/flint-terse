---
license: apache-2.0
base_model: RecursiveSI/flint-int4
base_model_relation: finetune
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- lm-studio
- flint
- reasoning
- code
---

# {TITLE} (MLX)

[Recursive Flint](https://huggingface.co/RecursiveSI/flint), fine-tuned to think briefly and in shorthand inside `<think>` while keeping its final answers. This is a complete MLX model for Apple Silicon: download it in LM Studio or load it with mlx-lm, with nothing else to install. The GGUF version is [{USER}/{NAME}](https://huggingface.co/{USER}/{NAME}).

```
"Hi!"   base:   123 thinking tokens  ("The user has just said "Hi!" — a simple greeting. ...")
        terse:   15 thinking tokens  User greets; respond friendly; ask how can help; brief; casual.
```

{SUMMARY}

## Results

Measured on the int4 base with the adapter in llama.cpp. Scores are the mean of three seeds with one sample per task (temperature 1, top-p 0.95, top-k 20). The thinking reduction is relative to the base on the same tasks.

| | Code (110 tasks) | Math (120 tasks) | Less thinking: code | math | prose & chat |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flint int4 (base) | 102.3 | 103.3 | — | — | — |
| [flint-terse](https://huggingface.co/{USER}/flint-terse-mlx) | **102.7** | **109.7** | 62% | 79% | 85% |
| [flint-terse-minimalist](https://huggingface.co/{USER}/flint-terse-minimalist-mlx) | 98.0 | 108.0 | **81%** | **86%** | 85% |

**flint-terse** keeps the base's coding accuracy. **flint-terse-minimalist** thinks the least, at about 4 code tasks lower. Final replies stay complete, and no `<think>` tags leak into them.

## What this model is

The MLX int4 base [`RecursiveSI/flint-int4`](https://huggingface.co/RecursiveSI/flint-int4) with the LoRA merged in ({MLX_SIZE}). The merge rewrote {MERGED_TENSORS}. They are stored unquantized as float16: re-quantizing them to 4 bits, as `mlx_lm.fuse` does, would round away most of the fine-tune, and bfloat16 would round away 5–18% of it. Every other tensor is the base's original int4. As with any mlx-lm LoRA, activations then run in float32. {MLX_CHECK} The chat template is Flint's, minus a `replace("\x00", "")` that LM Studio's template engine does not support.

The unmerged adapters (GGUF for llama.cpp, MLX for mlx-lm `--adapter-path`) are at [{USER}/{NAME}-lora](https://huggingface.co/{USER}/{NAME}-lora).

## Usage

**LM Studio:** search for `{USER}/{NAME}-mlx`, download, load. Set temperature 1, top-p 0.95 and top-k 20 (the base model's defaults).

**mlx-lm:**

```sh
mlx_lm.generate --model {USER}/{NAME}-mlx --temp 1 --top-p 0.95 --top-k 20 --max-tokens 4096 \
  --prompt "Write a function that merges overlapping intervals."
mlx_lm.server --model {USER}/{NAME}-mlx   # OpenAI-compatible server
```

## How it was made

Training ran directly on the int4 coefficients with rank-32 FP32 LoRA on attention, the shared expert and the dense MLPs{EXPERT_CLAUSE}. The model learned from its own answers by reinforcement learning: an answer earned reward only if it passed its tests and fitted a shrinking thinking budget. A short supervised pass then taught terse thinking for prose and chat. Code, data and a one-script reproduction are at **https://github.com/gyohng/flint-terse**.

## License

Apache-2.0, the same licence as Flint. This fine-tune adds no further conditions. Flint is by [Recursive](https://huggingface.co/RecursiveSI).
