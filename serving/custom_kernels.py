"""Custom fused CUDA kernels extracted from verified Muse Spark Open Kernel Challenge submissions.
Ready for integration into SGLang serving pipeline for the Serving Track.
"""

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// 1. Vectorized 128-bit float4 GEMV for bfloat16
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
    if (lane_id == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < 8) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int mask = 4; mask > 0; mask /= 2) {
            val += __shfl_down_sync(0xffffffff, val, mask);
        }
        if (lane_id == 0) y[m * N + n] = __float2bfloat16(val);
    }
}

// 2. Fused Router GEMV: x @ (hi + lo) -> float32 logits
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
                acc += __bfloat162float(x_vec[j].x) * (h0 + l0)
                     + __bfloat162float(x_vec[j].y) * (h1 + l1);
            }
        } else {
            for (int j = i; j < num_pairs; ++j) {
                __nv_bfloat162 xv = x2[j];
                __nv_bfloat162 hv = hi2[j];
                __nv_bfloat162 lv = lo2[j];
                acc += __bfloat162float(xv.x) * (__bfloat162float(hv.x) + __bfloat162float(lv.x))
                     + __bfloat162float(xv.y) * (__bfloat162float(hv.y) + __bfloat162float(lv.y));
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
    if (lane_id == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < 8) ? warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int mask = 4; mask > 0; mask /= 2) {
            val += __shfl_down_sync(0xffffffff, val, mask);
        }
        if (lane_id == 0) y[m * N + n] = val;
    }
}

// 3. In-place Pre-expert RMSNorm
__global__ void fast_pre_expert_norm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ out,
    int T
) {
    int t = blockIdx.x;
    if (t >= T) return;

    int tid = threadIdx.x;
    const int D = 4096;
    const int STRIDE = 256;
    const int PAIRS_PER_THREAD = 8;

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

// 4. Bitwise Fused Expert Mapping
__global__ void fuse_expert_mapping_kernel(
    const int* __restrict__ topk_ids,
    int* __restrict__ local_ids,
    int start,
    int count,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int gid = topk_ids[idx];
    local_ids[idx] = (gid >= start && gid < start + count) ? (gid - start) : -1;
}

// 5. Coalesced Post-Expert Norm & Weighted Top-K Sum
__global__ void fast_post_expert_norm_weighted_sum_kernel(
    const __nv_bfloat16* __restrict__ y,
    const float* __restrict__ topk_weights,
    const int* __restrict__ slot_expert_ids,
    const float* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ out,
    int T
) {
    int t = blockIdx.x;
    if (t >= T) return;

    int tid = threadIdx.x;
    const int D = 4096;
    const int STRIDE = 256;
    const int PAIRS_PER_THREAD = 8;

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
"""

CPP_SRC = """
#include <torch/extension.h>
void run_gemv_bf16(torch::Tensor x, torch::Tensor W, torch::Tensor y);
void run_gemv_router(torch::Tensor x, torch::Tensor hi, torch::Tensor lo, torch::Tensor y);
void run_fast_pre_expert_norm(torch::Tensor x, torch::Tensor norm_weight, torch::Tensor out, int T);
void run_fuse_expert_mapping(torch::Tensor topk_ids, torch::Tensor local_ids, int start, int count, int total_elements);
void run_fast_post_expert_norm_weighted_sum(
    torch::Tensor y, torch::Tensor topk_weights, torch::Tensor slot_expert_ids,
    torch::Tensor norm_weight, torch::Tensor out, int T);
"""
