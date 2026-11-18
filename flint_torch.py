"""PyTorch Flint on the int4 GGUF coefficients, for CUDA LoRA training.

The base is the unchanged `flint-int4-q4_1.gguf`. Routed experts stay packed as Q4_1 blocks on the
GPU (~24 GB) and are dequantized one layer at a time inside the forward pass; the small dense
tensors (attention, shared expert, dense MLPs, router, head) are dequantized once to BF16. Values
are exactly d*q+m from the file, rounded to BF16 for the matmuls.

Mirrors llama.cpp's src/models/flint.cpp: pre-norm decoder, grouped-query attention with per-head
QK RMSNorm and learned sinks, sliding-window layers with RoPE (window 513 including the current
token), full-attention layers without positions, dense SwiGLU on the leading layers, sigmoid top-k
routed experts with renormalized weights plus an ungated shared expert.

LoRA (FP32 factors) can be attached to attention q/k/v/o and to the shared-expert and dense MLP
projections; these are tensors llama.cpp applies GGUF LoRA adapters to.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def rms_norm(x, weight, eps):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype)


def dequant(blocks):
    try:
        import q4_kernels
        return q4_kernels.dequant_q4_1(blocks)
    except Exception:  # no compiler: the reference path below is bit-identical
        return dequant_q4_1(blocks)


def dequant_q4_1(blocks):
    """uint8 [..., n*20] Q4_1 rows -> bf16 [..., n*32]. Block: fp16 d, fp16 m, 16 bytes of nibbles
    (low nibbles are values 0-15, high nibbles 16-31)."""
    shape = blocks.shape
    b = blocks.reshape(-1, 20)
    d = b[:, 0:2].contiguous().view(torch.float16).float()
    m = b[:, 2:4].contiguous().view(torch.float16).float()
    qs = b[:, 4:]
    q = torch.cat((qs & 15, qs >> 4), dim=1).float()
    return (q * d + m).to(torch.bfloat16).reshape(*shape[:-1], shape[-1] // 20 * 32)


class Proj(nn.Module):
    """Frozen BF16 projection with an optional FP32 low-rank update: W x + scale * B (A x)."""

    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.lora_a = self.lora_b = None
        self.scale = 0.0
        self.enabled = True

    def add_lora(self, rank, scale):
        out, inp = self.weight.shape
        dev = self.weight.device
        self.lora_a = nn.Parameter(torch.empty(rank, inp, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.lora_b = nn.Parameter(torch.zeros(out, rank, device=dev, dtype=torch.float32))
        self.scale = scale

    def forward(self, x):
        y = F.linear(x, self.weight)
        if self.lora_a is not None and self.enabled:
            y = y + (F.linear(F.linear(x.float(), self.lora_a), self.lora_b) * self.scale).to(y.dtype)
        return y


class Attention(nn.Module):
    def __init__(self, cfg, layer, w):
        super().__init__()
        p = f"model.layers.{layer}.self_attn."
        self.heads, self.kv_heads, self.dim = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        for n in ATTN:
            setattr(self, n, Proj(w.pop(p + n + ".weight")))
        self.q_norm = nn.Parameter(w.pop(p + "q_norm.weight"), requires_grad=False)
        self.k_norm = nn.Parameter(w.pop(p + "k_norm.weight"), requires_grad=False)
        self.sinks = nn.Parameter(w.pop(p + "sinks"), requires_grad=False)
        self.window = cfg["sliding_window"] if cfg["layer_types"][layer] == "sliding_attention" else None
        self.rope = layer not in set(cfg["no_rope_layer_indices"])
        self.theta = cfg["rope_theta"]

    def forward(self, x, cache=None, seg=None):
        """seg: optional (positions [t], segment ids [t]) for packed sequences (no cache)."""
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.kv_heads, self.dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_heads, self.dim).transpose(1, 2)
        q, k = rms_norm(q, self.q_norm, self.eps), rms_norm(k, self.k_norm, self.eps)
        start = cache["offset"] if cache is not None else 0
        if seg is not None:
            pos, ids = seg
        else:
            pos, ids = torch.arange(start, start + t, device=x.device), None
        if self.rope:  # NeoX-style rotation (first half / second half)
            inv = self.theta ** (-torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32) / self.dim)
            ang = pos.float()[:, None] * inv[None]
            cos, sin = ang.cos(), ang.sin()

            def rot(u):
                u1, u2 = u.float().chunk(2, dim=-1)
                return torch.cat((u1 * cos - u2 * sin, u2 * cos + u1 * sin), dim=-1).to(u.dtype)

            q, k = rot(q), rot(k)
        if cache is not None:
            if "k" in cache:
                k, v = torch.cat((cache["k"], k), dim=2), torch.cat((cache["v"], v), dim=2)
            keep = self.window - 1 if self.window else k.shape[2]
            cache.update(k=k[:, :, -keep:], v=v[:, :, -keep:], offset=start + t)
        s = k.shape[2]
        rep = self.heads // self.kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        # Learned sink: an extra logit per head in the softmax denominator with no value, as a zero
        # key/value column whose score comes from an additive bias.
        zeros = torch.zeros(b, self.heads, 1, self.dim, device=x.device, dtype=k.dtype)
        k = torch.cat((zeros, k), dim=2)
        v = torch.cat((zeros, v), dim=2)
        kpos = torch.arange(s, device=x.device)
        outs = []
        for c0 in range(0, t, 1024):  # chunk queries to bound the mask/score memory
            c1 = min(t, c0 + 1024)
            qi = torch.arange(s - t + c0, s - t + c1, device=x.device)[:, None]
            allowed = kpos[None] <= qi
            if self.window:
                allowed = allowed & (qi - kpos[None] < self.window)
            if ids is not None:
                allowed = allowed & (ids[c0:c1, None] == ids[None])
            bias = torch.empty(1, self.heads, c1 - c0, s + 1, device=x.device, dtype=q.dtype)
            bias[..., 1:] = torch.zeros((), dtype=q.dtype, device=x.device).masked_fill(~allowed, float("-inf"))
            bias[..., 0] = self.sinks.to(q.dtype).view(1, -1, 1)
            outs.append(F.scaled_dot_product_attention(q[:, :, c0:c1], k, v, attn_mask=bias, scale=self.dim ** -0.5))
        out = torch.cat(outs, dim=2).transpose(1, 2).reshape(b, t, self.heads * self.dim)
        return self.o_proj(out)


class DenseMLP(nn.Module):
    def __init__(self, w, p):
        super().__init__()
        for n in MLP:
            setattr(self, n, Proj(w.pop(f"{p}{n}.weight")))

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoE(nn.Module):
    def __init__(self, cfg, layer, w):
        super().__init__()
        p = f"model.layers.{layer}.mlp."
        self.topk = cfg["num_experts_per_tok"]
        self.router = nn.Parameter(w.pop(p + "gate.weight"), requires_grad=False)
        # Packed Q4_1 rows: gate/up [E, 512, 64*20], down [E, 2048, 16*20].
        self.gate_q = nn.Parameter(w.pop(p + "experts.gate_q"), requires_grad=False)
        self.up_q = nn.Parameter(w.pop(p + "experts.up_q"), requires_grad=False)
        self.down_q = nn.Parameter(w.pop(p + "experts.down_q"), requires_grad=False)
        self.shared_expert = DenseMLP(w, p + "shared_expert.")
        self.down_a = self.down_b = None  # optional per-expert LoRA on down_proj: [E, r, I], [E, H, r]
        self.expert_scale = 0.0
        self.expert_enabled = True

    def add_expert_lora(self, rank, scale):
        n, hidden, inter = self.down_q.shape[0], self.down_q.shape[1], self.gate_q.shape[1]
        dev = self.router.device
        self.down_a = nn.Parameter(torch.randn(n, rank, inter, device=dev) / math.sqrt(inter))
        self.down_b = nn.Parameter(torch.zeros(n, hidden, rank, device=dev))
        self.expert_scale = scale

    def _expert_lora(self):
        return self.down_a is not None and self.expert_enabled

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        scores = torch.sigmoid(F.linear(x.float(), self.router.float()))
        weights, ids = torch.topk(scores, self.topk, dim=-1)
        weights = (weights / (weights.sum(-1, keepdim=True) + 1e-20)).to(x.dtype)
        out = self.shared_expert(x)
        flat = ids.reshape(-1)
        order = flat.argsort()
        token = torch.arange(x.shape[0], device=x.device).repeat_interleave(self.topk)[order]
        wflat = weights.reshape(-1)[order]
        xs = x[token]
        n_exp = self.gate_q.shape[0]
        if flat.numel() >= n_exp:
            # Training/prefill: dequantize every expert once, then grouped GEMMs.
            counts = torch.bincount(flat, minlength=n_exp)
            offs = counts.cumsum(0).to(torch.int32)
            gate, up, down = (dequant(q) for q in (self.gate_q, self.up_q, self.down_q))
            h = F.silu(torch._grouped_mm(xs, gate.transpose(1, 2), offs=offs)) * \
                torch._grouped_mm(xs, up.transpose(1, 2), offs=offs)
            y = torch._grouped_mm(h, down.transpose(1, 2), offs=offs)
            if self._expert_lora():
                t = torch._grouped_mm(h, self.down_a.to(h.dtype).transpose(1, 2), offs=offs)
                y = y + self.expert_scale * torch._grouped_mm(t, self.down_b.to(h.dtype).transpose(1, 2), offs=offs)
        else:
            # Decode: only the few routed experts.
            ys = []
            for j, e in enumerate(flat[order].tolist()):
                g, u, d = (dequant(q[e]) for q in (self.gate_q, self.up_q, self.down_q))
                hj = F.silu(F.linear(xs[j:j + 1], g)) * F.linear(xs[j:j + 1], u)
                yj = F.linear(hj, d)
                if self._expert_lora():
                    yj = yj + self.expert_scale * F.linear(F.linear(hj.float(), self.down_a[e]), self.down_b[e]).to(yj.dtype)
                ys.append(yj)
            y = torch.cat(ys)
        out = out.index_add(0, token, y * wflat[:, None])
        return out.view(shape)


class Layer(nn.Module):
    def __init__(self, cfg, layer, w):
        super().__init__()
        p = f"model.layers.{layer}."
        self.eps = cfg["rms_norm_eps"]
        self.norm1 = nn.Parameter(w.pop(p + "input_layernorm.weight"), requires_grad=False)
        self.norm2 = nn.Parameter(w.pop(p + "post_attention_layernorm.weight"), requires_grad=False)
        self.self_attn = Attention(cfg, layer, w)
        self.mlp = DenseMLP(w, p + "mlp.") if layer in set(cfg["mlp_only_layers"]) else SparseMoE(cfg, layer, w)

    def forward(self, x, cache=None, seg=None):
        x = x + self.self_attn(rms_norm(x, self.norm1, self.eps), cache, seg)
        return x + self.mlp(rms_norm(x, self.norm2, self.eps))


class Flint(nn.Module):
    def __init__(self, cfg, w):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Parameter(w.pop("model.embed_tokens.weight"), requires_grad=False)
        self.layers = nn.ModuleList(Layer(cfg, i, w) for i in range(cfg["num_hidden_layers"]))
        self.norm = nn.Parameter(w.pop("model.norm.weight"), requires_grad=False)
        self.lm_head = nn.Parameter(w.pop("lm_head.weight"), requires_grad=False)

    def hidden(self, tokens, cache=None, checkpoint=False, seg=None):
        x = F.embedding(tokens, self.embed).to(torch.bfloat16)
        for i, layer in enumerate(self.layers):
            c = None if cache is None else cache.setdefault(i, {"offset": 0})
            if checkpoint and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(layer, x, c, seg, use_reentrant=False)
            else:
                x = layer(x, c, seg)
        return rms_norm(x, self.norm, self.cfg["rms_norm_eps"])

    def logits(self, hidden):
        return F.linear(hidden, self.lm_head).float()

    def forward(self, tokens, cache=None, checkpoint=False, seg=None):
        return self.logits(self.hidden(tokens, cache, checkpoint, seg))

    # ---- LoRA ----
    def projections(self):
        """(checkpoint-style name, Proj) for every adaptable projection."""
        for i, layer in enumerate(self.layers):
            for n in ATTN:
                yield f"model.layers.{i}.self_attn.{n}", getattr(layer.self_attn, n)
            mlp = layer.mlp if isinstance(layer.mlp, DenseMLP) else layer.mlp.shared_expert
            prefix = f"model.layers.{i}.mlp" + ("" if isinstance(layer.mlp, DenseMLP) else ".shared_expert")
            for n in MLP:
                yield f"{prefix}.{n}", getattr(mlp, n)

    def add_lora(self, rank, scale, targets=ATTN + MLP):
        for name, proj in self.projections():
            if name.rsplit(".", 1)[1] in targets:
                proj.add_lora(rank, scale)
        return [p for p in self.parameters() if p.requires_grad]

    def set_lora(self, enabled):
        for _, proj in self.projections():
            proj.enabled = enabled
        for _, moe in self.moe_layers():
            moe.expert_enabled = enabled

    def moe_layers(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer.mlp, SparseMoE):
                yield i, layer.mlp

    def add_expert_lora(self, rank, scale, last):
        """Per-expert LoRA on down_proj in the last `last` MoE layers."""
        layers = list(self.moe_layers())[-last:] if last else []
        for _, moe in layers:
            moe.add_expert_lora(rank, scale)
        return [p for p in self.parameters() if p.requires_grad]

    def lora_state(self):
        """FP32 factors in MLX layout (lora_a [in, r], lora_b [r, out]; experts [E, in, r], [E, r, out]) as
        export_adapter.py expects. Expert factors carry their own scale folded into lora_b."""
        state = {k: v for name, proj in self.projections() if proj.lora_a is not None
                 for k, v in ((name + ".lora_a", proj.lora_a.detach().T.contiguous().cpu()),
                              (name + ".lora_b", proj.lora_b.detach().T.contiguous().cpu()))}
        for i, moe in self.moe_layers():
            if moe.down_a is not None:
                name = f"model.layers.{i}.mlp.experts.down_proj"
                state[name + ".lora_a"] = moe.down_a.detach().transpose(1, 2).contiguous().cpu()
                state[name + ".lora_b"] = (moe.down_b.detach() * moe.expert_scale).transpose(1, 2).contiguous().cpu()
        return state

    def load_lora(self, state, scale):
        projs = dict(self.projections())
        moes = dict(self.moe_layers())
        for key in {k.rsplit(".", 1)[0] for k in state}:
            if ".mlp.experts." in key:  # expert scale is folded into lora_b
                moe = moes[int(key.split(".")[2])]
                dev = moe.router.device
                moe.down_a = nn.Parameter(state[key + ".lora_a"].transpose(1, 2).contiguous().to(dev, torch.float32))
                moe.down_b = nn.Parameter(state[key + ".lora_b"].transpose(1, 2).contiguous().to(dev, torch.float32))
                moe.expert_scale = 1.0
                continue
            proj = projs[key]
            dev = proj.weight.device
            proj.lora_a = nn.Parameter(state[key + ".lora_a"].T.contiguous().to(dev, torch.float32))
            proj.lora_b = nn.Parameter(state[key + ".lora_b"].T.contiguous().to(dev, torch.float32))
            proj.scale = scale


GGUF_DEFAULT = "models/flint-int4-gguf/flint-int4-q4_1.gguf"


def load(path=GGUF_DEFAULT, device="cuda", config="models/flint/config.json"):
    """Flint on the int4 GGUF's own coefficients."""
    import gguf

    cfg = json.loads(Path(config).read_text())
    reader = gguf.GGUFReader(path)
    tensors = {t.name: t for t in reader.tensors}
    n_layers = cfg["num_hidden_layers"]

    def raw(name):
        t = tensors.pop(name)
        return t, np.asarray(t.data)

    def dense(name):
        t, data = raw(name)
        shape = tuple(int(x) for x in t.shape[::-1])
        if t.tensor_type == gguf.GGMLQuantizationType.Q4_1:
            return dequant_q4_1(torch.from_numpy(data.reshape(shape[0], -1)).to(device))
        if t.tensor_type != gguf.GGMLQuantizationType.F32:
            raise ValueError(f"{name}: unexpected type {t.tensor_type.name}")
        return torch.from_numpy(data.reshape(shape)).to(device)

    def packed(name):
        t, data = raw(name)
        if t.tensor_type != gguf.GGMLQuantizationType.Q4_1:
            raise ValueError(f"{name}: expected Q4_1 experts")
        experts, rows = int(t.shape[2]), int(t.shape[1])
        return torch.from_numpy(data.reshape(experts, rows, -1)).to(device)

    bf16 = lambda x: x.to(torch.bfloat16)
    w = {"model.embed_tokens.weight": bf16(dense("token_embd.weight")),
         "model.norm.weight": dense("output_norm.weight"),
         "lm_head.weight": bf16(dense("output.weight"))}
    for i in range(n_layers):
        p, g = f"model.layers.{i}.", f"blk.{i}."
        w[p + "input_layernorm.weight"] = dense(g + "attn_norm.weight")
        w[p + "post_attention_layernorm.weight"] = dense(g + "ffn_norm.weight")
        for hf, gg in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
            w[p + f"self_attn.{hf}.weight"] = bf16(dense(g + gg + ".weight"))
        w[p + "self_attn.q_norm.weight"] = dense(g + "attn_q_norm.weight")
        w[p + "self_attn.k_norm.weight"] = dense(g + "attn_k_norm.weight")
        w[p + "self_attn.sinks"] = dense(g + "attn_sinks.weight")
        if i in set(cfg["mlp_only_layers"]):
            for hf, gg in (("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"), ("down_proj", "ffn_down")):
                w[p + f"mlp.{hf}.weight"] = bf16(dense(g + gg + ".weight"))
        else:
            w[p + "mlp.gate.weight"] = dense(g + "ffn_gate_inp.weight")
            for hf, gg in (("gate_proj", "ffn_gate_shexp"), ("up_proj", "ffn_up_shexp"), ("down_proj", "ffn_down_shexp")):
                w[p + f"mlp.shared_expert.{hf}.weight"] = bf16(dense(g + gg + ".weight"))
            w[p + "mlp.experts.gate_q"] = packed(g + "ffn_gate_exps.weight")
            w[p + "mlp.experts.up_q"] = packed(g + "ffn_up_exps.weight")
            w[p + "mlp.experts.down_q"] = packed(g + "ffn_down_exps.weight")
    if tensors:
        raise ValueError(f"Unused GGUF tensors: {sorted(tensors)[:5]}")
    model = Flint(cfg, w)
    if w:
        raise ValueError(f"Unused tensors: {sorted(w)[:5]}")
    return model.eval()
