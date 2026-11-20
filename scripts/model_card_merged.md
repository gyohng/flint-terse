---
license: apache-2.0
base_model: RecursiveSI/flint-int4-gguf
base_model_relation: finetune
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- lm-studio
- flint
- reasoning
- code
---

# {TITLE}

[Recursive Flint](https://huggingface.co/RecursiveSI/flint), fine-tuned to think briefly and in shorthand inside `<think>` while keeping its final answers. This is a complete GGUF model: download it in LM Studio or pass it to llama.cpp, with nothing else to install.

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
| [flint-terse](https://huggingface.co/{USER}/flint-terse) | **102.7** | **109.7** | 62% | 79% | 85% |
| [flint-terse-minimalist](https://huggingface.co/{USER}/flint-terse-minimalist) | 98.0 | 108.0 | **81%** | **86%** | 85% |

The merged GGUF is the same network as base plus adapter: over 16K tokens of chat transcripts its mean KL divergence from base plus adapter is 0.003 with 98% the same top token, no more than base plus adapter differs from itself run with another batch size (llama.cpp `llama-perplexity --kl-divergence`).

**flint-terse** keeps the base's coding accuracy. **flint-terse-minimalist** thinks the least, at about 4 code tasks lower. Final replies stay complete, and no `<think>` tags leak into them.

## What this file is

`{NAME}.gguf` ({MERGED_SIZE}) is the original [`RecursiveSI/flint-int4-gguf`](https://huggingface.co/RecursiveSI/flint-int4-gguf) `flint-int4-q4_1.gguf` with the LoRA merged in by llama.cpp's `llama-export-lora`, which rewrote {MERGED_TENSORS} as F16. Every other tensor is copied unchanged as Q4_1. The embedded chat template is Flint's, minus a `replace("\x00", "")` that LM Studio's template engine does not support. Context length is the base's 262,144 tokens.

The unmerged adapter is also available: GGUF LoRA for llama.cpp at [{USER}/{NAME}-lora](https://huggingface.co/{USER}/{NAME}-lora), and MLX LoRA for mlx-lm at [{USER}/{NAME}-mlx](https://huggingface.co/{USER}/{NAME}-mlx).

## Usage

**LM Studio:** search for `{USER}/{NAME}`, download, load. Set temperature 1, top-p 0.95 and top-k 20 (the base model's defaults).

**llama.cpp:**

```sh
hf download {USER}/{NAME} {NAME}.gguf --local-dir .
llama-server -m {NAME}.gguf --jinja --temp 1 --top-p 0.95 --top-k 20
```

## How it was made

Training ran directly on the int4 coefficients with rank-32 FP32 LoRA on attention, the shared expert and the dense MLPs{EXPERT_CLAUSE}. The model learned from its own answers by reinforcement learning: an answer earned reward only if it passed its tests and fitted a shrinking thinking budget. A short supervised pass then taught terse thinking for prose and chat. Code, data and a one-script reproduction are at **https://github.com/gyohng/flint-terse**.

## License

Apache-2.0, the same licence as Flint. This fine-tune adds no further conditions. Flint is by [Recursive](https://huggingface.co/RecursiveSI).
