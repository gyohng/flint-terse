"""CUDA Q4_1 dequantization (GGUF block layout), compiled on first use with the pip CUDA toolkit."""

import os
from pathlib import Path

import torch

_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// One thread per 32-value block: 2-byte fp16 d, 2-byte fp16 m, 16 bytes of nibbles.
__global__ void dq_kernel(const uint8_t* __restrict__ in, __nv_bfloat16* __restrict__ out, long nblocks) {
    long b = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const uint8_t* p = in + b * 20;
    float d = __half2float(*reinterpret_cast<const __half*>(p));
    float m = __half2float(*reinterpret_cast<const __half*>(p + 2));
    __nv_bfloat16* o = out + b * 32;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        uint8_t q = p[4 + j];
        o[j] = __float2bfloat16(d * (float)(q & 15) + m);
        o[j + 16] = __float2bfloat16(d * (float)(q >> 4) + m);
    }
}

torch::Tensor dequant_q4_1(torch::Tensor in) {
    TORCH_CHECK(in.is_cuda() && in.dtype() == torch::kUInt8 && in.is_contiguous());
    TORCH_CHECK(in.size(-1) % 20 == 0);
    auto sizes = in.sizes().vec();
    sizes.back() = sizes.back() / 20 * 32;
    auto out = torch::empty(sizes, in.options().dtype(torch::kBFloat16));
    long nblocks = in.numel() / 20;
    int threads = 256;
    dq_kernel<<<(nblocks + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        in.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), nblocks);
    return out;
}
"""

_CPP = "torch::Tensor dequant_q4_1(torch::Tensor in);"
_mod = None


def _load():
    global _mod
    if _mod is None:
        import nvidia
        cuda = Path(nvidia.__path__[0]) / "cu13"
        os.environ.setdefault("CUDA_HOME", str(cuda))
        os.environ["PATH"] = f"{cuda / 'bin'}:{os.environ['PATH']}"
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.1")
        from torch.utils.cpp_extension import load_inline
        _mod = load_inline("flint_q4", cpp_sources=_CPP,
                           cuda_sources=_SRC.replace("#include <torch/extension.h>",
                                                     "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>"),
                           functions=["dequant_q4_1"], extra_cuda_cflags=["-O3"],
                           build_directory=str(_build_dir()), verbose=False)
    return _mod


def _build_dir():
    d = Path(__file__).resolve().parent / ".tools" / "torch-ext"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dequant_q4_1(blocks):
    return _load().dequant_q4_1(blocks.contiguous())
