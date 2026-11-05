# okc: problem=muse_decode_layer
"""High-performance fused decoder layer for Muse Spark Open on H200."""

from typing import TypedDict
import types
import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn
from torch.utils.cpp_extension import load_inline

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin

CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

__global__ void gemv_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ y,
    int M, int N, int K
) {
    int n = blockIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x + m * K);
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(W + n * K);
    int num_pairs = K / 2;

    float acc = 0.0f;
    int tid = threadIdx.x;
    int stride = blockDim.x * 4;

    for (int i = tid * 4; i < num_pairs; i += stride) {
        if (i + 3 < num_pairs) {
            float4 x_raw = *reinterpret_cast<const float4*>(&x2[i]);
            float4 w_raw = *reinterpret_cast<const float4*>(&w2[i]);
            const __nv_bfloat162* x_vec = reinterpret_cast<const __nv_bfloat162*>(&x_raw);
            const __nv_bfloat162* w_vec = reinterpret_cast<const __nv_bfloat162*>(&w_raw);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc += __bfloat162float(x_vec[j].x) * __bfloat162float(w_vec[j].x)
                     + __bfloat162float(x_vec[j].y) * __bfloat162float(w_vec[j].y);
            }
        } else {
            for (int j = i; j < num_pairs; ++j) {
                __nv_bfloat162 xv = x2[j];
                __nv_bfloat162 wv = w2[j];
                acc += __bfloat162float(xv.x) * __bfloat162float(wv.x)
                     + __bfloat162float(xv.y) * __bfloat162float(wv.y);
            }
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, mask);
    }

    __shared__ float warp_sums[8];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < 8) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int mask = 4; mask > 0; mask /= 2) {
            val += __shfl_down_sync(0xffffffff, val, mask);
        }
        if (lane_id == 0) {
            y[m * N + n] = __float2bfloat16(val);
        }
    }
}

__global__ void gemv_router_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ hi,
    const __nv_bfloat16* __restrict__ lo,
    float* __restrict__ y,
    int M, int N, int K
) {
    int n = blockIdx.x;
    int m = blockIdx.y;
    if (n >= N || m >= M) return;

    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x + m * K);
    const __nv_bfloat162* hi2 = reinterpret_cast<const __nv_bfloat162*>(hi + n * K);
    const __nv_bfloat162* lo2 = reinterpret_cast<const __nv_bfloat162*>(lo + n * K);
    int num_pairs = K / 2;

    float acc = 0.0f;
    int tid = threadIdx.x;
    int stride = blockDim.x * 4;

    for (int i = tid * 4; i < num_pairs; i += stride) {
        if (i + 3 < num_pairs) {
            float4 x_raw = *reinterpret_cast<const float4*>(&x2[i]);
            float4 hi_raw = *reinterpret_cast<const float4*>(&hi2[i]);
            float4 lo_raw = *reinterpret_cast<const float4*>(&lo2[i]);
            const __nv_bfloat162* x_vec = reinterpret_cast<const __nv_bfloat162*>(&x_raw);
            const __nv_bfloat162* hi_vec = reinterpret_cast<const __nv_bfloat162*>(&hi_raw);
            const __nv_bfloat162* lo_vec = reinterpret_cast<const __nv_bfloat162*>(&lo_raw);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float h0 = __bfloat162float(hi_vec[j].x);
                float l0 = __bfloat162float(lo_vec[j].x);
                float h1 = __bfloat162float(hi_vec[j].y);
                float l1 = __bfloat162float(lo_vec[j].y);
                acc += __bfloat162float(x_vec[j].x) * h0 + __bfloat162float(x_vec[j].x) * l0
                     + __bfloat162float(x_vec[j].y) * h1 + __bfloat162float(x_vec[j].y) * l1;
            }
        } else {
            for (int j = i; j < num_pairs; ++j) {
                __nv_bfloat162 xv = x2[j];
                __nv_bfloat162 hv = hi2[j];
                __nv_bfloat162 lv = lo2[j];
                acc += __bfloat162float(xv.x) * __bfloat162float(hv.x) + __bfloat162float(xv.x) * __bfloat162float(lv.x)
                     + __bfloat162float(xv.y) * __bfloat162float(hv.y) + __bfloat162float(xv.y) * __bfloat162float(lv.y);
            }
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        acc += __shfl_down_sync(0xffffffff, acc, mask);
    }

    __shared__ float warp_sums[8];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < 8) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int mask = 4; mask > 0; mask /= 2) {
            val += __shfl_down_sync(0xffffffff, val, mask);
        }
        if (lane_id == 0) {
            y[m * N + n] = val;
        }
    }
}

__global__ void fast_pre_expert_norm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ norm_weight,  // FP32: 1.0 + w
    __nv_bfloat16* __restrict__ out,
    int T
) {
    int t = blockIdx.x;
    if (t >= T) return;

    int tid = threadIdx.x;
    const int D = 4096;
    const int STRIDE = 256;
    const int PAIRS_PER_THREAD = 8; // 256 * 8 = 2048 pairs = 4096 elements

    const __nv_bfloat162* x_row = reinterpret_cast<const __nv_bfloat162*>(x + t * D);
    __nv_bfloat162* out_row = reinterpret_cast<__nv_bfloat162*>(out + t * D);

    __nv_bfloat162 x_local[PAIRS_PER_THREAD];
    float sum_sq = 0.0f;

    #pragma unroll
    for (int k = 0; k < PAIRS_PER_THREAD; ++k) {
        int pair_idx = tid + k * STRIDE;
        __nv_bfloat162 val = x_row[pair_idx];
        x_local[k] = val;
        float fx = __bfloat162float(val.x);
        float fy = __bfloat162float(val.y);
        sum_sq += fx * fx + fy * fy;
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, mask);
    }

    __shared__ float s_red[8];
    int wid = tid / 32;
    int lane = tid % 32;
    if (lane == 0) s_red[wid] = sum_sq;
    __syncthreads();

    if (wid == 0) {
        float val = (lane < 8) ? s_red[lane] : 0.0f;
        #pragma unroll
        for (int mask = 4; mask > 0; mask /= 2) {
            val += __shfl_down_sync(0xffffffff, val, mask);
        }
        if (lane == 0) s_red[0] = val;
    }
    __syncthreads();

    float rsqrt_val = rsqrtf(s_red[0] * (1.0f / 4096.0f) + 1e-5f);

    #pragma unroll
    for (int k = 0; k < PAIRS_PER_THREAD; ++k) {
        int pair_idx = tid + k * STRIDE;
        int elem_idx = pair_idx * 2;
        float w0 = norm_weight[elem_idx];
        float w1 = norm_weight[elem_idx + 1];
        float o0 = (__bfloat162float(x_local[k].x) * rsqrt_val) * w0;
        float o1 = (__bfloat162float(x_local[k].y) * rsqrt_val) * w1;
        out_row[pair_idx] = __halves2bfloat162(__float2bfloat16(o0), __float2bfloat16(o1));
    }
}

__global__ void fuse_expert_mapping_kernel(
    const int* __restrict__ topk_ids,
    const int* __restrict__ local_start_ptr,
    int* __restrict__ local_ids,
    int count,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int gid = topk_ids[idx];
    int start = *local_start_ptr;
    int diff = (gid - start) & 255;
    local_ids[idx] = (diff < count) ? diff : -1;
}

// 256 threads, 128-byte coalesced vector loads via __nv_bfloat162
__global__ void fast_post_expert_norm_weighted_sum_kernel(
    const __nv_bfloat16* __restrict__ y,          // [T, 8, 4096]
    const float* __restrict__ topk_weights,       // [T, 8]
    const int* __restrict__ slot_expert_ids,      // [T, 8]
    const float* __restrict__ norm_weight,        // [4096] (FP32: 1.0 + w)
    __nv_bfloat16* __restrict__ out,              // [T, 4096]
    int T
) {
    int t = blockIdx.x;
    if (t >= T) return;

    int tid = threadIdx.x;
    const int D = 4096;
    const int STRIDE = 256;
    const int PAIRS_PER_THREAD = 8; // 256 * 8 = 2048 pairs = 4096 elements

    float w0_local[PAIRS_PER_THREAD];
    float w1_local[PAIRS_PER_THREAD];
    #pragma unroll
    for (int k = 0; k < PAIRS_PER_THREAD; ++k) {
        int pair_idx = tid + k * STRIDE;
        w0_local[k] = norm_weight[pair_idx * 2];
        w1_local[k] = norm_weight[pair_idx * 2 + 1];
    }

    float out_acc0[PAIRS_PER_THREAD] = {0.0f};
    float out_acc1[PAIRS_PER_THREAD] = {0.0f};

    const int* ids_row = slot_expert_ids + t * 8;
    const float* weights_row = topk_weights + t * 8;
    const __nv_bfloat16* y_token = y + t * 8 * D;

    __shared__ float s_red[8];
    int wid = tid / 32;
    int lane = tid % 32;

    for (int s = 0; s < 8; ++s) {
        if (ids_row[s] < 0) continue;

        float gw = weights_row[s];
        const __nv_bfloat162* y_pairs = reinterpret_cast<const __nv_bfloat162*>(y_token + s * D);

        float scaled0[PAIRS_PER_THREAD];
        float scaled1[PAIRS_PER_THREAD];
        float sum_sq = 0.0f;

        #pragma unroll
        for (int k = 0; k < PAIRS_PER_THREAD; ++k) {
            int pair_idx = tid + k * STRIDE;
            __nv_bfloat162 y_val = y_pairs[pair_idx];
            float v0 = __bfloat162float(y_val.x) * w0_local[k];
            float v1 = __bfloat162float(y_val.y) * w1_local[k];
            scaled0[k] = v0;
            scaled1[k] = v1;
            sum_sq += v0 * v0 + v1 * v1;
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, mask);
        }
        if (lane == 0) s_red[wid] = sum_sq;
        __syncthreads();

        if (wid == 0) {
            float val = (lane < 8) ? s_red[lane] : 0.0f;
            #pragma unroll
            for (int mask = 4; mask > 0; mask /= 2) {
                val += __shfl_down_sync(0xffffffff, val, mask);
            }
            if (lane == 0) s_red[0] = val;
        }
        __syncthreads();

        float rsqrt_val = rsqrtf(s_red[0] * (1.0f / 4096.0f) + 1e-8f);

        #pragma unroll
        for (int k = 0; k < PAIRS_PER_THREAD; ++k) {
            out_acc0[k] += gw * (scaled0[k] * rsqrt_val);
            out_acc1[k] += gw * (scaled1[k] * rsqrt_val);
        }
    }

    __nv_bfloat162* out_pairs = reinterpret_cast<__nv_bfloat162*>(out + t * D);
    #pragma unroll
    for (int k = 0; k < PAIRS_PER_THREAD; ++k) {
        int pair_idx = tid + k * STRIDE;
        out_pairs[pair_idx] = __halves2bfloat162(
            __float2bfloat16(out_acc0[k]),
            __float2bfloat16(out_acc1[k])
        );
    }
}

void run_gemv_bf16(torch::Tensor x, torch::Tensor W, torch::Tensor y) {
    int M = x.size(0);
    int K = x.size(1);
    int N = W.size(0);
    dim3 block(256);
    dim3 grid(N, M);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gemv_bf16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        M, N, K
    );
}

void run_gemv_router(torch::Tensor x, torch::Tensor hi, torch::Tensor lo, torch::Tensor y) {
    int M = x.size(0);
    int K = x.size(1);
    int N = hi.size(0);
    dim3 block(256);
    dim3 grid(N, M);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gemv_router_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(hi.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(lo.data_ptr()),
        y.data_ptr<float>(),
        M, N, K
    );
}

void run_fast_pre_expert_norm(
    torch::Tensor x,
    torch::Tensor norm_weight,
    torch::Tensor out,
    int T
) {
    dim3 grid(T);
    dim3 block(256);
    cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();
    fast_pre_expert_norm_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        norm_weight.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        T
    );
}

void run_fuse_expert_mapping(
    torch::Tensor topk_ids,
    torch::Tensor local_start,
    torch::Tensor local_ids,
    int count,
    int total_elements
) {
    dim3 block(64);
    dim3 grid((total_elements + 63) / 64);
    cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();
    fuse_expert_mapping_kernel<<<grid, block, 0, s>>>(
        topk_ids.data_ptr<int>(),
        local_start.data_ptr<int>(),
        local_ids.data_ptr<int>(),
        count,
        total_elements
    );
}

void run_fast_post_expert_norm_weighted_sum(
    torch::Tensor y,
    torch::Tensor topk_weights,
    torch::Tensor slot_expert_ids,
    torch::Tensor norm_weight,
    torch::Tensor out,
    int T
) {
    dim3 grid(T);
    dim3 block(256);
    cudaStream_t s = c10::cuda::getCurrentCUDAStream().stream();
    fast_post_expert_norm_weighted_sum_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()),
        topk_weights.data_ptr<float>(),
        slot_expert_ids.data_ptr<int>(),
        norm_weight.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        T
    );
}

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

    const __nv_bfloat16* q_row = q + t * 1024;
    __nv_fp8_e4m3* q_dst = q_fp8 + t * 1024;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int idx = tid + i * 256;
        float val = __bfloat162float(q_row[idx]);
        q_dst[idx] = __nv_fp8_e4m3(val);
    }

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

CPP_SRC = """
#include <torch/extension.h>
void run_gemv_bf16(torch::Tensor x, torch::Tensor W, torch::Tensor y);
void run_gemv_router(torch::Tensor x, torch::Tensor hi, torch::Tensor lo, torch::Tensor y);
void run_fast_pre_expert_norm(
    torch::Tensor x,
    torch::Tensor norm_weight,
    torch::Tensor out,
    int T
);
void run_fuse_expert_mapping(
    torch::Tensor topk_ids,
    torch::Tensor local_start,
    torch::Tensor local_ids,
    int count,
    int total_elements
);
void run_fast_post_expert_norm_weighted_sum(
    torch::Tensor y,
    torch::Tensor topk_weights,
    torch::Tensor slot_expert_ids,
    torch::Tensor norm_weight,
    torch::Tensor out,
    int T
);
void run_fast_cast_store(
    torch::Tensor q, torch::Tensor k, torch::Tensor qkv, torch::Tensor cache_loc,
    torch::Tensor q_fp8, torch::Tensor k_cache, torch::Tensor v_cache,
    int64_t qkv_stride, int64_t v_offset, double inv_k_scale, double inv_v_scale, int64_t T
);
"""

_CUDA_MODULE = None


def get_cuda_module():
    global _CUDA_MODULE
    if _CUDA_MODULE is None:
        _CUDA_MODULE = load_inline(
            name="okc_decode_fused_v6",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=[
                "run_gemv_bf16",
                "run_gemv_router",
                "run_fast_pre_expert_norm",
                "run_fuse_expert_mapping",
                "run_fast_post_expert_norm_weighted_sum",
                "run_fast_cast_store",
            ],
            extra_cuda_cflags=["-O3", "-use_fast_math", "--threads=4"],
            with_cuda=True,
        )
    return _CUDA_MODULE


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    attn_residual_gate: torch.Tensor
    pre_ffn_norm: torch.Tensor
    router: torch.Tensor
    router_bias: torch.Tensor
    pooler: torch.Tensor
    pre_expert_norm: torch.Tensor
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_expert_norm: torch.Tensor
    unpooler: torch.Tensor
    post_ffn_norm: torch.Tensor
    ffn_residual_gate: torch.Tensor
    next_pre_attn_norm: torch.Tensor


class Config(TypedDict):
    batch: int
    q_len: int
    tokens: int
    hidden: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float
    n_experts: int
    top_k: int
    local_experts: int


class Inputs(TypedDict):
    stream: torch.Tensor
    attn_input: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    local_start: torch.Tensor
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    gate.to(w.device)
    return gate


def _norm(cls, w, size, eps, device="cuda", **kw):
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
        norm.to(w.device)
    else:
        norm.to(device)
    return norm


def _marlin_bank(weights):
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def prepare(weights: Weights):
    cuda_mod = get_cuda_module()
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16).contiguous()
    lo = (router - hi.float()).to(torch.bfloat16).contiguous()
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS, device=device)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS, device=device)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, device=device, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS, device=device,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)

    dense = {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")}
    dense_t = {name: weights[name].t().contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")}
    dense["unpooler_slice"] = weights["unpooler"][:UNPOOL_SLICE].contiguous()
    dense_t["unpooler_slice"] = weights["unpooler"][:UNPOOL_SLICE].t().contiguous()

    bank = _marlin_bank(weights)
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    max_tokens = 512
    max_tok_k = max_tokens * TOP_K

    cache2_buf = torch.empty((max_tok_k, size_n), device=device, dtype=torch.bfloat16)
    cache13_buf = torch.empty((max_tok_k * max(gemm1_n, HIDDEN),), device=device, dtype=torch.bfloat16)
    pre_expert_effective_w = (weights["pre_expert_norm"].float() + 1.0).contiguous()
    post_expert_effective_w = (weights["post_expert_norm"].float() + 1.0).contiguous()

    # Pre-allocated workspace buffers for zero allocation during decode steps
    qkv_1_buf = torch.empty((1, 2304), dtype=torch.bfloat16, device=device)
    expert_in_1_buf = torch.empty((1, 4096), dtype=torch.bfloat16, device=device)
    ffn_output_1_buf = torch.zeros((1, HIDDEN), dtype=torch.bfloat16, device=device)
    logits_buf = torch.empty((512, 256), dtype=torch.float32, device=device)
    local_ids_buf = torch.empty((max_tokens, TOP_K), dtype=torch.int32, device=device)
    expert_out_buf = torch.empty((max_tokens, D_EXPERT), dtype=torch.bfloat16, device=device)
    q_fp8_buf = torch.empty((max_tokens, 16, 64), dtype=torch.float8_e4m3fn, device=device)

    state = {
        "cuda_mod": cuda_mod,
        "dense": dense,
        "dense_t": dense_t,
        "router_hi": hi,
        "router_lo": lo,
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_effective_w": pre_expert_effective_w,
        "post_expert_effective_w": post_expert_effective_w,
        "qkv_1_buf": qkv_1_buf,
        "q_fp8_buf": q_fp8_buf,
        "expert_in_1_buf": expert_in_1_buf,
        "ffn_output_1_buf": ffn_output_1_buf,
        "logits_buf": logits_buf,
        "local_ids_buf": local_ids_buf,
        "expert_out_buf": expert_out_buf,
        "bank": bank,
        "side_stream": torch.cuda.Stream(device),
        "cache2_buf": cache2_buf,
        "cache13_buf": cache13_buf,
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn),
    }
    del freqs
    torch.cuda.synchronize()
    return state


def _experts(x, topk_weights, local_ids, state):
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n

    if num_tokens == 32:
        block_size_m = 16
    elif num_tokens == 256:
        block_size_m = 32
    else:
        for block_size_m in BLOCK_SIZE_M_CANDIDATES:
            if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m <= 1.0:
                break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    tot = num_tokens * TOP_K
    if tot > state["cache2_buf"].size(0):
        cache2 = torch.empty((tot, size_n), device=x.device, dtype=x.dtype)
        cache13 = torch.empty((tot * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
        cache1 = cache13[: tot * gemm1_n].view(-1, gemm1_n)
        cache3 = cache13[: tot * size_k].view(-1, size_k)
    else:
        cache2 = state["cache2_buf"][:tot]
        cache1 = state["cache13_buf"][:tot * gemm1_n].view(-1, gemm1_n)
        cache3 = state["cache13_buf"][:tot * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    
    if num_tokens <= state["expert_out_buf"].size(0):
        out = state["expert_out_buf"][:num_tokens]
    else:
        out = torch.empty((num_tokens, size_k), dtype=torch.bfloat16, device=x.device)
    cuda_mod = state["cuda_mod"]
    cuda_mod.run_fast_post_expert_norm_weighted_sum(
        y, topk_weights, local_ids, state["post_expert_effective_w"], out, num_tokens
    )
    return out


def kernel_fn(data: Inputs, state: dict):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    dense_t = state["dense_t"]
    cuda_mod = state["cuda_mod"]
    
    # 1. QKV projection: vectorized 1-pass GEMV for tokens=1, cuBLAS for tokens>1
    if tokens == 1:
        qkv = state["qkv_1_buf"]
        cuda_mod.run_gemv_bf16(data["attn_input"], dense["qkv_gate"], qkv)
    else:
        qkv = torch.mm(data["attn_input"], dense_t["qkv_gate"])

    if cfg["rope_theta"]:
        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
    else:
        q, k = state["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8),
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
    o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
    
    # 2. O projection
    attn_output = torch.mm(o, dense_t["o"])

    # 3. Post-attention boundary & Router
    ffn_input, stream, _ = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None)
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    
    if tokens == 1:
        expert_input = state["expert_in_1_buf"]
        cuda_mod.run_gemv_bf16(ffn_input, dense["pooler"], expert_input)
    else:
        expert_input = torch.mm(ffn_input, dense_t["pooler"])

    with torch.cuda.stream(side_stream):
        if tokens <= 8:
            logits = state["logits_buf"][:tokens]
            cuda_mod.run_gemv_router(ffn_input, state["router_hi"], state["router_lo"], logits)
        else:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)

    # Fast in-place pre-expert RMSNorm: runs on main_stream right after pooler without waiting for side_stream
    cuda_mod.run_fast_pre_expert_norm(
        expert_input, state["pre_expert_effective_w"], expert_input, tokens
    )

    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)

    # Fused expert mapping: replaces PyTorch modulo and bank["expert_map"] GPU indexing
    tot = tokens * TOP_K
    if tokens <= state["local_ids_buf"].size(0):
        local_ids = state["local_ids_buf"][:tokens]
    else:
        local_ids = torch.empty((tokens, TOP_K), dtype=torch.int32, device=expert_input.device)
    cuda_mod.run_fuse_expert_mapping(
        topk_ids, data["local_start"], local_ids, LOCAL_EXPERTS, tot
    )

    expert_output = _experts(expert_input, topk_weights, local_ids, state)
    
    # 4. Unpooler projection & FFN boundary
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        if tokens == 1:
            ffn_output = state["ffn_output_1_buf"]
            unpooled_slice = ffn_output[:, :UNPOOL_SLICE]
            cuda_mod.run_gemv_bf16(expert_output, dense["unpooler_slice"], unpooled_slice)
        else:
            unpooled = torch.mm(expert_output, dense_t["unpooler_slice"])
            ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))
    else:
        ffn_output = torch.mm(expert_output, dense_t["unpooler"])

    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
