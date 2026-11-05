# okc: problem=muse_moe_w4a16
"""Marlin NVFP4 routed experts with fast post-expert norm for Muse Spark Open on H200."""

from typing import TypedDict
import types
import torch
from sgl_kernel.scalar_type import scalar_types
from torch.utils.cpp_extension import load_inline

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin

CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

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
    const int STRIDE = 512;
    const int ELEMS_PER_THREAD = 8; // 512 * 8 = 4096

    float w_local[ELEMS_PER_THREAD];
    #pragma unroll
    for (int k = 0; k < ELEMS_PER_THREAD; ++k) {
        int idx = tid + k * STRIDE;
        w_local[k] = norm_weight[idx];
    }

    float out_acc[ELEMS_PER_THREAD] = {0.0f};

    const int* ids_row = slot_expert_ids + t * 8;
    const float* weights_row = topk_weights + t * 8;
    const __nv_bfloat16* y_token = y + t * 8 * D;

    __shared__ float s_red[16];
    int wid = tid / 32;
    int lane = tid % 32;

    for (int s = 0; s < 8; ++s) {
        if (ids_row[s] < 0) continue;

        float gw = weights_row[s];
        const __nv_bfloat16* y_slot = y_token + s * D;

        float scaled[ELEMS_PER_THREAD];
        float sum_sq = 0.0f;
        #pragma unroll
        for (int k = 0; k < ELEMS_PER_THREAD; ++k) {
            int idx = tid + k * STRIDE;
            float val = __bfloat162float(y_slot[idx]) * w_local[k];
            scaled[k] = val;
            sum_sq += val * val;
        }

        #pragma unroll
        for (int mask = 16; mask > 0; mask /= 2) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, mask);
        }
        if (lane == 0) s_red[wid] = sum_sq;
        __syncthreads();

        if (wid == 0) {
            float val = (tid < 16) ? s_red[tid] : 0.0f;
            #pragma unroll
            for (int mask = 8; mask > 0; mask /= 2) {
                val += __shfl_down_sync(0xffff, val, mask);
            }
            if (lane == 0) s_red[0] = val;
        }
        __syncthreads();

        float rsqrt_val = rsqrtf(s_red[0] * (1.0f / 4096.0f) + 1e-8f);

        #pragma unroll
        for (int k = 0; k < ELEMS_PER_THREAD; ++k) {
            out_acc[k] += gw * (scaled[k] * rsqrt_val);
        }
    }

    __nv_bfloat16* out_row = out + t * D;
    #pragma unroll
    for (int k = 0; k < ELEMS_PER_THREAD; ++k) {
        int idx = tid + k * STRIDE;
        out_row[idx] = __float2bfloat16(out_acc[k]);
    }
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
    dim3 block(512);
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
"""

CPP_SRC = """
#include <torch/extension.h>
void run_fast_post_expert_norm_weighted_sum(
    torch::Tensor y,
    torch::Tensor topk_weights,
    torch::Tensor slot_expert_ids,
    torch::Tensor norm_weight,
    torch::Tensor out,
    int T
);
"""

_CUDA_MODULE = None


def get_cuda_module():
    global _CUDA_MODULE
    if _CUDA_MODULE is None:
        _CUDA_MODULE = load_inline(
            name="okc_moe_fused_norm",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["run_fast_post_expert_norm_weighted_sum"],
            extra_cuda_cflags=["-O3", "-use_fast_math", "--threads=4"],
            with_cuda=True,
        )
    return _CUDA_MODULE


class Weights(TypedDict):
    w13: torch.Tensor
    w13_sf: torch.Tensor
    w13_gs: torch.Tensor
    w2: torch.Tensor
    w2_sf: torch.Tensor
    w2_gs: torch.Tensor
    post_norm_w: torch.Tensor


class Config(TypedDict):
    bs: int
    d_hidden: int
    d_expert: int
    n_experts: int
    local_start: int
    local_count: int
    top_k: int
    sharding: str


class Inputs(TypedDict):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    config: Config


N_EXPERTS = 256
TOP_K = 8
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8


def _marlin_layer(weights, start, count):
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def prepare(weights: Weights):
    cuda_mod = get_cuda_module()
    device = weights["w13"].device
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    
    max_tokens = 512
    max_tok_k = max_tokens * TOP_K
    size_n = banks[(0, 32)]["size_n"]
    gemm1_n = 2 * size_n
    cache2_buf = torch.empty((max_tok_k, size_n), device=device, dtype=torch.bfloat16)
    cache13_buf = torch.empty((max_tok_k * max(gemm1_n, 4096),), device=device, dtype=torch.bfloat16)

    torch.cuda.synchronize()
    return {
        "cuda_mod": cuda_mod,
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        "norm_w": (weights["post_norm_w"].float() + 1.0).contiguous(),
        "cache2_buf": cache2_buf,
        "cache13_buf": cache13_buf,
    }


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    local_ids = data["topk_ids"].to(torch.int32)
    if expert_map is not None:
        local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()

    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / bank["count"] / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(
        local_ids, block_size_m, N_EXPERTS if expert_map is not None else bank["count"])

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
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    
    out = torch.empty((num_tokens, size_k), dtype=torch.bfloat16, device=x.device)
    cuda_mod = state["cuda_mod"]
    cuda_mod.run_fast_post_expert_norm_weighted_sum(
        y, topk_weights, local_ids, state["norm_w"], out, num_tokens
    )
    return out
