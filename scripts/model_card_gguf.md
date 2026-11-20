---
license: apache-2.0
base_model: RecursiveSI/flint-int4-gguf
base_model_relation: adapter
pipeline_tag: text-generation
tags:
- lora
- gguf
- llama.cpp
- lm-studio
- flint
- reasoning
- code
---

# {TITLE}

A LoRA adapter for [Recursive Flint](https://huggingface.co/RecursiveSI/flint) that makes it think briefly and in shorthand inside `<think>` while keeping its final answers. This repository holds the GGUF LoRA for llama.cpp, applied on top of the unchanged int4 GGUF base. **For LM Studio, download the merged model [{USER}/{NAME}](https://huggingface.co/{USER}/{NAME})**: LM Studio cannot apply LoRA adapters, and this file alone will not load there. The MLX version is [{USER}/{NAME}-mlx](https://huggingface.co/{USER}/{NAME}-mlx).

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
| [flint-terse](https://huggingface.co/{USER}/flint-terse) | **102.7** | **109.7** | 62% | 79% | 85% |
| [flint-terse-minimalist](https://huggingface.co/{USER}/flint-terse-minimalist) | 98.0 | 108.0 | **81%** | **86%** | 85% |

**flint-terse** keeps the base's coding accuracy. **flint-terse-minimalist** thinks the least, at about 4 code tasks lower. Final replies stay complete, and no `<think>` tags leak into them.

## Files

| File | Use |
| --- | --- |
| `{NAME}.gguf` | GGUF LoRA for llama.cpp ({GGUF_SIZE}) |
| `chat_template.jinja` | Flint's chat template without its `replace("\x00", "")`, which LM Studio's template engine does not support |

## Base model

Use the exact base the adapter was trained on.

[`RecursiveSI/flint-int4-gguf`](https://huggingface.co/RecursiveSI/flint-int4-gguf), file `flint-int4-q4_1.gguf`, revision `9c47151036bc2c34331142dda04488bceab2bc9f`, SHA-256 `43c07baccdba42eb7528a9630ee15699b1942c297291dc1b39a061f508ee1a4b`.
Other quantizations are different checkpoints and are not supported.

## Usage

**llama.cpp:**

```sh
hf download RecursiveSI/flint-int4-gguf flint-int4-q4_1.gguf --revision 9c47151036bc2c34331142dda04488bceab2bc9f --local-dir .
hf download {USER}/{NAME}-lora --local-dir {NAME}
llama-server -m flint-int4-q4_1.gguf --lora {NAME}/{NAME}.gguf \
  --jinja --chat-template-file {NAME}/chat_template.jinja --temp 1 --top-p 0.95 --top-k 20
```

**MLX:** see [{USER}/{NAME}-mlx](https://huggingface.co/{USER}/{NAME}-mlx).

LM Studio users can also attach this adapter to the base with the [GitHub repository](https://github.com/gyohng/flint-terse)'s `./run.sh deploy --adapter release/{NAME}`, but the merged model is simpler.

Recommended sampling is temperature 1, top-p 0.95 and top-k 20, the base model's defaults.
{EXTRA}
## How it was made

Training ran directly on the int4 coefficients with rank-32 FP32 LoRA on attention, the shared expert and the dense MLPs. The model learned from its own answers by reinforcement learning: an answer earned reward only if it passed its tests and fitted a shrinking thinking budget. A short supervised pass then taught terse thinking for prose and chat. Code, data and a one-script reproduction are at **https://github.com/gyohng/flint-terse**.

## License

Apache-2.0, the same licence as Flint. This fine-tune adds no further conditions. Flint is by [Recursive](https://huggingface.co/RecursiveSI).
