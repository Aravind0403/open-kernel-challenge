# okc: problem=muse_attention_decode
"""Attention decode for Muse Spark Open on H200 with fused cast & store."""

from typing import TypedDict
import torch
from torch.utils.cpp_extension import load_inline
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate


class Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float


class Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: Config


RMS_EPS = 1e-5
FP8 = torch.float8_e4m3fn
_NORMS = {}
_ROPE = {}
ROPE_POSITIONS = 1 << 21
_STATE = None

CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

__global__ void fast_cast_store_kernel(
    const __nv_bfloat16* __restrict__ q,       // [T, 1024]
    const __nv_bfloat16* __restrict__ k,       // [T, 128]
    const __nv_bfloat16* __restrict__ qkv,     // [T, qkv_stride]
    const int64_t* __restrict__ cache_loc,     // [T]
    __nv_fp8_e4m3* __restrict__ q_fp8,         // [T, 1024]
    __nv_fp8_e4m3* __restrict__ k_cache,       // [N, 128]
    __nv_fp8_e4m3* __restrict__ v_cache,       // [N, 128]
    int qkv_stride,
    int v_offset,
    float inv_k_scale,
    float inv_v_scale,
    int T
) {
    int t = blockIdx.x;
    if (t >= T) return;

    int tid = threadIdx.x; // 256 threads
    int64_t loc = cache_loc[t];

    // 1. Cast Q (1024 elements per token): 256 threads * 4 elements = 1024
    const __nv_bfloat16* q_row = q + t * 1024;
    __nv_fp8_e4m3* q_dst = q_fp8 + t * 1024;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = tid + i * 256;
        float val = __bfloat162float(q_row[idx]);
        q_dst[idx] = __nv_fp8_e4m3(val);
    }

    // 2. Cast and store K and V (128 elements each per token)
    if (loc >= 0) {
        if (tid < 128) {
            float val = __bfloat162float(k[t * 128 + tid]) * inv_k_scale;
            k_cache[loc * 128 + tid] = __nv_fp8_e4m3(val);
        } else {
            int v_idx = tid - 128;
            const __nv_bfloat16* v_row = qkv + t * qkv_stride + v_offset;
            float val = __bfloat162float(v_row[v_idx]) * inv_v_scale;
            v_cache[loc * 128 + v_idx] = __nv_fp8_e4m3(val);
        }
    }
}

void run_fast_cast_store(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor qkv,
    torch::Tensor cache_loc,
    torch::Tensor q_fp8,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    int64_t qkv_stride,
    int64_t v_offset,
    double inv_k_scale,
    double inv_v_scale,
    int64_t T
) {
    auto stream = c10::cuda::getCurrentCUDAStream().stream();
    fast_cast_store_kernel<<<T, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr()),
        reinterpret_cast<const int64_t*>(cache_loc.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(q_fp8.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(k_cache.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3*>(v_cache.data_ptr()),
        static_cast<int>(qkv_stride),
        static_cast<int>(v_offset),
        static_cast<float>(inv_k_scale),
        static_cast<float>(inv_v_scale),
        static_cast<int>(T)
    );
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
void run_fast_cast_store(
    torch::Tensor q, torch::Tensor k, torch::Tensor qkv, torch::Tensor cache_loc,
    torch::Tensor q_fp8, torch::Tensor k_cache, torch::Tensor v_cache,
    int64_t qkv_stride, int64_t v_offset, double inv_k_scale, double inv_v_scale, int64_t T
);
"""


def _init_state(device):
    global _STATE
    cuda_mod = load_inline(
        name="fast_attn_cast_store_mod",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        functions=["run_fast_cast_store"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_90a"],
        with_cuda=True,
    )
    max_tokens = 512
    q_fp8_buf = torch.empty((max_tokens, 16, 64), dtype=torch.float8_e4m3fn, device=device)
    out_buf = torch.empty((max_tokens, 16 * 64), dtype=torch.bfloat16, device=device)
    _STATE = {
        "cuda_mod": cuda_mod,
        "q_fp8_buf": q_fp8_buf,
        "out_buf": out_buf,
        "device": device,
    }


def _qk_norm(hq, hkv, d):
    key = (hq, hkv, d)
    if key not in _NORMS:
        _NORMS[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=RMS_EPS)
    return _NORMS[key]


def _cos_sin(theta, d, device):
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache


def kernel_fn(data: Inputs) -> torch.Tensor:
    global _STATE
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device

    if _STATE is None or _STATE["device"] != device:
        _init_state(device)

    if cfg["rope_theta"]:
        q, k = _qk_norm(hq, hkv, d)(qkv, data["positions"], _cos_sin(cfg["rope_theta"], d, device))
    else:
        q, k = _qk_norm(hq, hkv, d)(qkv)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]

    k_cache, v_cache = data["k_cache"], data["v_cache"]

    if hq == 16 and hkv == 2:
        k_cache_flat = k_cache.view(-1, hkv * d)
        v_cache_flat = v_cache.view(-1, hkv * d)

        if tokens > _STATE["q_fp8_buf"].size(0):
            q_fp8 = torch.empty((tokens, hq, d), dtype=torch.float8_e4m3fn, device=device)
            out_res = torch.empty((tokens, hq * d), dtype=torch.bfloat16, device=device)
        else:
            q_fp8 = _STATE["q_fp8_buf"][:tokens]
            out_res = _STATE["out_buf"][:tokens]

        inv_scale = 1.0 / cfg["kv_scale"]
        v_offset = (hq + hkv) * d
        _STATE["cuda_mod"].run_fast_cast_store(
            q, k, qkv, data["cache_loc"], q_fp8.view(tokens, -1), k_cache_flat, v_cache_flat,
            qkv.stride(0), v_offset, inv_scale, inv_scale, tokens
        )
    else:
        scaled = cfg["kv_scale"] != 1.0
        k = k.reshape(tokens, hkv * d)
        if scaled:
            k, v = k / data["k_scale"], v / data["v_scale"]
        store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                    k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                    data["cache_loc"], reserved_skip_index=-1)
        q_fp8 = q.view(tokens, hq, d).to(FP8)
        out_res = None

    window = cfg["window"]
    descale = {}
    if cfg["kv_scale"] != 1.0:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q_fp8,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    out_bf16 = out.reshape(tokens, hq * d)
    if out_bf16.dtype != torch.bfloat16:
        out_bf16 = out_bf16.to(torch.bfloat16)
    o = rms_norm_gate(out_bf16, gate, None, head_dim=d, eps=RMS_EPS, has_norm=True, out=out_res)
    return o.view(batch, length, hq * d)
