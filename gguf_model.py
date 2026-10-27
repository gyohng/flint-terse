"""Train against GGUF coefficients. Repack integers losslessly; never quantize weights."""

from pathlib import Path

import gguf
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.flint import Model, ModelArgs
from mlx_lm.tokenizer_utils import TokenizerWrapper
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def unpack(blocks, kind):
    """Return unsigned codes and exact FP32 affine scales/biases per group."""
    n = len(blocks)
    if kind == gguf.GGMLQuantizationType.Q4_1:
        s = blocks[:, :2].copy().view(np.float16).astype(np.float32)
        b = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
        q = blocks[:, 4:, None] >> np.array([0, 4], np.uint8)
        return (q & 15).transpose(0, 2, 1).reshape(n, 32), s, b, 4, 32
    if kind == gguf.GGMLQuantizationType.Q4_K:
        d = blocks[:, :2].copy().view(np.float16).astype(np.float32)
        dm = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
        s, m = gguf.quants.Q4_K.get_scale_min(blocks[:, 4:16])
        q = blocks[:, 16:].reshape(n, 4, 1, 32) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)
        return (q & 15).reshape(n, 256), d * s, -dm * m, 4, 32
    if kind == gguf.GGMLQuantizationType.Q6_K:
        ql = blocks[:, :128].reshape(n, 2, 1, 64) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)
        qh = blocks[:, 128:192].reshape(n, 2, 1, 32) >> np.array([0, 2, 4, 6], np.uint8).reshape(1, 1, 4, 1)
        q = (ql & 15).reshape(n, 8, 32) | ((qh & 3).reshape(n, 8, 32) << 4)
        s = blocks[:, 208:].copy().view(np.float16).astype(np.float32) * blocks[:, 192:208].view(np.int8)
        return q.reshape(n, 256), s, -32 * s, 8, 16
    if kind == gguf.GGMLQuantizationType.Q8_0:
        s = blocks[:, :2].copy().view(np.float16).astype(np.float32)
        q = (blocks[:, 2:].view(np.int8).astype(np.int16) + 128).astype(np.uint8)
        return q, s, -128 * s, 8, 32
    raise ValueError(f"Unsupported GGUF quantization: {kind.name}")


def repack(tensor):
    shape = tuple(int(x) for x in tensor.shape[::-1])
    block_size, block_bytes = gguf.GGML_QUANT_SIZES[tensor.tensor_type]
    blocks = tensor.data.reshape(-1, block_bytes)
    padded = tensor.tensor_type == gguf.GGMLQuantizationType.Q6_K
    bits = 4 if tensor.tensor_type in (gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q4_1) else 8
    width = shape[-1] * (2 if padded else 1)
    weights = np.empty((int(np.prod(shape[:-1])), width * bits // 32), np.uint32)
    scales = np.empty((weights.shape[0], width // 32), np.float32)
    biases = np.empty_like(scales)
    wflat, sflat, bflat = weights.reshape(-1), scales.reshape(-1), biases.reshape(-1)
    chunk_size = 32768 if tensor.tensor_type == gguf.GGMLQuantizationType.Q4_1 else 4096
    for start in range(0, len(blocks), chunk_size):
        chunk = blocks[start:start + chunk_size]
        per_word = 32 // bits
        if tensor.tensor_type == gguf.GGMLQuantizationType.Q4_1:
            # Rearrange nibbles directly, then invert the permutation to verify every code.
            q = chunk[:, 4:]
            lo = (q[:, ::2] & 15) | ((q[:, 1::2] & 15) << 4)
            hi = (q[:, ::2] >> 4) | (q[:, 1::2] & 240)
            restored = np.empty_like(q)
            restored[:, ::2] = (lo & 15) | ((hi & 15) << 4)
            restored[:, 1::2] = (lo >> 4) | (hi & 240)
            if not np.array_equal(q, restored):
                raise ValueError(f"Integer code mismatch in {tensor.name}, block {start}")
            packed = np.concatenate((lo, hi), axis=1).view(np.uint32).reshape(-1)
            # Every FP16 scale/offset is exactly representable in FP32.
            s = chunk[:, :2].copy().view(np.float16).astype(np.float32)
            b = chunk[:, 2:4].copy().view(np.float16).astype(np.float32)
            group = 32
        else:
            q, s, b, _, group = unpack(chunk, tensor.tensor_type)
            # Check every coefficient against llama.cpp's GGUF reference decoder.
            decoded = (q.reshape(len(chunk), -1, group).astype(np.float32) * s[..., None] + b[..., None]).reshape(-1)
            expected = gguf.dequantize(chunk, tensor.tensor_type).reshape(-1)
            if not np.array_equal(decoded, expected):
                raise ValueError(f"Coefficient mismatch in {tensor.name}, block {start}")
            if padded:
                q = np.pad(q.reshape(-1, 16), ((0, 0), (0, 16)), constant_values=32).reshape(len(chunk), -1)
            packed = np.bitwise_or.reduce(q.reshape(-1, per_word).astype(np.uint32) << (np.arange(per_word, dtype=np.uint32) * bits), axis=-1)
        wstart = start * block_size * (2 if padded else 1) // per_word
        sstart = start * block_size // group
        wflat[wstart:wstart + packed.size] = packed
        sflat[sstart:sstart + s.size] = s.reshape(-1)
        bflat[sstart:sstart + b.size] = b.reshape(-1)
    return tuple(mx.array(a.reshape(shape[:-1] + (a.shape[-1],))) for a in (weights, scales, biases)), bits, padded


class GGUFLinear(nn.Module):
    def __init__(self, tensor):
        super().__init__()
        (self.weight, self.scales, self.biases), self.bits, self.padded = repack(tensor)
        self.freeze()

    def to_lora(self, r, scale, dropout):
        from mlx_lm.tuner.lora import LoRALinear
        if self.weight.ndim != 2:
            raise ValueError('GGUF LoRA currently supports dense projections, not routed experts')
        width = self.weight.shape[-1] * 32 // self.bits // (2 if self.padded else 1)
        layer = LoRALinear(width, self.weight.shape[0], r=r, scale=scale, dropout=dropout)
        layer.linear = self
        return layer

    def __call__(self, x, indices=None, sorted_indices=False):
        x = x.astype(mx.float32)
        if self.padded:
            shape = x.shape
            x = mx.concatenate((x.reshape(*shape[:-1], -1, 16), mx.zeros((*shape[:-1], shape[-1] // 16, 16))), axis=-1).reshape(*shape[:-1], -1)
        args = dict(transpose=True, group_size=32, bits=self.bits)
        if indices is not None:
            return mx.gather_qmm(x, self.weight, self.scales, self.biases, rhs_indices=indices,
                                 sorted_indices=sorted_indices, **args)
        return mx.quantized_matmul(x, self.weight, self.scales, self.biases, **args)


def tokenizer_from_gguf(reader):
    fields = reader.fields
    if fields['tokenizer.ggml.pre'].contents() != 'flint':
        raise ValueError('This trainer currently supports Flint GGUF checkpoints')
    vocab = fields['tokenizer.ggml.tokens'].contents()
    merges = [tuple(pair.split(' ')) for pair in fields['tokenizer.ggml.merges'].contents()]
    tokenizer = Tokenizer(models.BPE(vocab={s: i for i, s in enumerate(vocab)}, merges=merges))
    patterns = [r'\p{N}{1,3}', '[一-龥぀-ゟ゠-ヿ]+',
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"]
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        *(pre_tokenizers.Split(Regex(p), behavior='isolated') for p in patterns),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    for token, kind in zip(vocab, fields['tokenizer.ggml.token_type'].contents()):
        if kind in (3, 4):
            tokenizer.add_tokens([AddedToken(token, normalized=False, special=kind == 3)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
        chat_template=fields['tokenizer.chat_template'].contents(),
        bos_token=vocab[fields['tokenizer.ggml.bos_token_id'].contents()],
        eos_token=vocab[fields['tokenizer.ggml.eos_token_id'].contents()])
    return TokenizerWrapper(fast, eos_token_ids={fast.eos_token_id, fast.convert_tokens_to_ids('<|user|>'),
                                               fast.convert_tokens_to_ids('<|observation|>')})


def load_tokenizer(path):
    if Path(path).suffix.lower() == '.gguf':
        if not Path(path).is_file():
            from checkpoint import resolve_checkpoint
            path, _ = resolve_checkpoint(path)
        return tokenizer_from_gguf(gguf.GGUFReader(path))
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)


def load_model(path, adapter_path=None, **kwargs):
    if Path(path).suffix.lower() != '.gguf':
        from mlx_lm import load
        return load(path, adapter_path=adapter_path, **kwargs)
    if not Path(path).is_file():
        from checkpoint import resolve_checkpoint
        path, _ = resolve_checkpoint(path)
    reader = gguf.GGUFReader(path)
    def field(name):
        return reader.fields['flint.' + name].contents()
    if reader.fields['general.architecture'].contents() != 'flint':
        raise ValueError('This trainer currently supports Flint GGUF checkpoints')
    sliding = field('attention.sliding_window_pattern')
    if sliding != field('attention.rope_pattern'):
        raise ValueError('Unsupported Flint attention/rope pattern')
    args = ModelArgs(model_type='flint', hidden_size=field('embedding_length'),
        num_hidden_layers=field('block_count'), intermediate_size=field('feed_forward_length'),
        num_attention_heads=field('attention.head_count'), num_key_value_heads=field('attention.head_count_kv'),
        head_dim=field('attention.key_length'), num_experts=field('expert_count'),
        num_experts_per_tok=field('expert_used_count'), moe_intermediate_size=field('expert_feed_forward_length'),
        shared_expert_intermediate_size=field('expert_shared_feed_forward_length') if 'flint.expert_shared_feed_forward_length' in reader.fields else field('expert_shared_count') * field('expert_feed_forward_length'),
        mlp_only_layers=list(range(field('leading_dense_block_count'))),
        layer_types=['sliding_attention' if s else 'full_attention' for s in sliding],
        sliding_window=field('attention.sliding_window'), rms_norm_eps=field('attention.layer_norm_rms_epsilon'),
        vocab_size=len(reader.fields['tokenizer.ggml.tokens'].contents()), rope_theta=field('rope.freq_base'))
    model = Model(args)
    tensors = {t.name: t for t in reader.tensors}
    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.FLINT, args.num_hidden_layers)
    params = list(tree_flatten(model.parameters()))
    seen = set()
    for i, (name, _) in enumerate(params):
        mapped = name.replace('.switch_mlp.', '.experts.')
        if mapped.endswith('.softmax_offset'):
            mapped += '.weight'
        mapped = names.get_name(mapped, try_suffixes=('.weight', '.bias'))
        tensor = tensors[mapped]
        seen.add(mapped)
        if name.endswith('.weight') and len(tensor.shape) >= 2 and name != 'model.embed_tokens.weight' and tensor.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            module = GGUFLinear(tensor)
            model.update_modules(tree_unflatten([(name.removesuffix('.weight'), module)]))
            mx.eval(module.parameters())
        else:
            value = mx.array(gguf.dequantize(tensor.data, tensor.tensor_type).reshape(tuple(tensor.shape[::-1])))
            model.update(tree_unflatten([(name, value)]))
            mx.eval(value)
        if i % 50 == 0:
            print(f'GGUF coefficients verified: {i + 1}/{len(params)}', flush=True)
        mx.clear_cache()
    if seen != set(tensors):
        raise ValueError(f'Unused GGUF tensors: {set(tensors) - seen}')
    tokenizer = tokenizer_from_gguf(reader)
    model.eval()
    if adapter_path:
        from mlx_lm.tuner.utils import load_adapters
        model = load_adapters(model, adapter_path)
    return model, tokenizer
