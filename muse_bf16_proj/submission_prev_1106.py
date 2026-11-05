# okc: problem=muse_bf16_proj
"""High-performance fused BF16 GEMV and projections for Muse Spark Open on H200."""

from typing import TypedDict
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Vectorized GEMV for bfloat16: y[m, n] = x[m] . W[n]
// Block size: 256 threads. Grid: (N, M).
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

    // Warp reduction
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

// Fused Router GEMV: evaluates x @ (hi + lo) in a single pass directly to float32
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
"""

CPP_SRC = """
#include <torch/extension.h>
void run_gemv_bf16(torch::Tensor x, torch::Tensor W, torch::Tensor y);
void run_gemv_router(torch::Tensor x, torch::Tensor hi, torch::Tensor lo, torch::Tensor y);
"""

_CUDA_MODULE = None


def get_cuda_module():
    global _CUDA_MODULE
    if _CUDA_MODULE is None:
        _CUDA_MODULE = load_inline(
            name="okc_fused_gemv_v2",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            functions=["run_gemv_bf16", "run_gemv_router"],
            with_cuda=True,
        )
    return _CUDA_MODULE


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


def prepare(weights: Weights):
    cuda_mod = get_cuda_module()

    state = {
        "cuda_mod": cuda_mod,
        "weights": {name: w.contiguous() for name, w in weights.items()},
        "weights_t": {name: w.t().contiguous() for name, w in weights.items() if name != "router"},
    }

    # Pre-slice unpooler
    state["unpooler_slice"] = weights["unpooler"][:1024].contiguous()
    state["unpooler_slice_t"] = weights["unpooler"][:1024].t().contiguous()
    state["unpooler_full_t"] = weights["unpooler"].t().contiguous()

    # Router
    router = weights["router"]
    hi = router.to(torch.bfloat16).contiguous()
    lo = (router - hi.float()).to(torch.bfloat16).contiguous()
    state["router_hi"] = hi
    state["router_lo"] = lo
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()

    torch.cuda.synchronize()
    return state


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    x = data["x"]
    m = x.shape[0]
    layer = data["config"]["layer"]
    cuda_mod = state["cuda_mod"]

    if layer == "router":
        if m <= 8:
            y = torch.empty((m, 256), dtype=torch.float32, device=x.device)
            cuda_mod.run_gemv_router(x, state["router_hi"], state["router_lo"], y)
            return y
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc

    if layer == "unpooler":
        n = data["config"]["n"]
        if n <= 1024:
            if m == 1:
                y = torch.empty((1, 1024), dtype=torch.bfloat16, device=x.device)
                cuda_mod.run_gemv_bf16(x, state["unpooler_slice"], y)
                return y
            return torch.mm(x, state["unpooler_slice_t"])
        return torch.mm(x, state["unpooler_full_t"])

    if m == 1 and layer in ("qkv_gate", "pooler"):
        w = state["weights"][layer]
        y = torch.empty((1, w.shape[0]), dtype=torch.bfloat16, device=x.device)
        cuda_mod.run_gemv_bf16(x, w, y)
        return y

    return torch.mm(x, state["weights_t"][layer])
