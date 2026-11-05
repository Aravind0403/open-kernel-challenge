# Open Kernel Challenge (OKC) — Frontier LLM Inference Optimization

[![OKC Leaderboard](https://img.shields.io/badge/OKC%20Rank-%231309%20Global%20(Top%20Tier)-blue?style=for-the-badge&logo=target)](https://openkernelchallenge.dev)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20Hopper%20H200-76B900?style=for-the-badge&logo=nvidia)](https://www.nvidia.com/en-us/data-center/h200/)
[![Challenge](https://img.shields.io/badge/Sponsors-Meta%20%C3%97%20Paradigm%20%C3%97%20Andromeda-6366f1?style=for-the-badge)](#)
[![Status](https://img.shields.io/badge/Verified%20Solutions-4%2F4%20Decode%20Problems-success?style=for-the-badge)](#)

High-performance GPU kernel engineering and distributed inference server optimization for **Muse Spark Open (816B Parameter Sparse Mixture-of-Experts)** on **NVIDIA Hopper H200** GPUs.

---

## 🏆 Verified Benchmark Results

Evaluated on official NVIDIA Hopper H200 silicon against production **SGLang** and **cuBLAS** baselines with strict **relative $L_2$ error $< 0.08$**:

| Problem | Benchmark Scope | Baseline Latency | Optimized Latency | Speedup | Global Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`muse_bf16_proj`** | Router GEMV ($M=1$) | 10.3 µs | **4.4 µs** | **2.329×** | #1586 |
| **`muse_bf16_proj`** | Unpooler GEMV ($M=1$) | 7.0 µs | **4.0 µs** | **1.777×** | #1586 |
| **`muse_moe_w4a16`** | Decode 1 ($M=1$, W4A16 MoE) | 77.1 µs | **50.8 µs** | **1.518×** | #1299 |
| **`muse_moe_w4a16`** | Verify 32 (256 Tokens) | 632.0 µs | **480.0 µs** | **1.317×** *(−152 µs)* | #1299 |
| **`muse_decode_layer`**| Full Decode Layer (Single token)| 172.2 µs | **126.6 µs** | **1.360×** | #1164 |
| **`muse_decode_layer`**| Full Decode Layer (Composite 13 tests)| 22.62 ms | **18.95 ms** | **1.193×** | #1164 |
| **`muse_attention_decode`**| FlashAttention-3 + Norm-Gate (16 tests)| 3.77 ms | **3.46 ms** | **1.090×** | #1361 |
| **OVERALL BOARD** | **All 4 Problems Solved & Verified** | — | — | **4 / 4 Solved** | **#1309** |

---

## 🚀 Key Engineering & Algorithmic Optimizations

### 1. Vectorized 128-Bit GEMV with Register-Level Warp Shuffles
* **The Bottleneck:** For single-token decode ($M=1$), NVIDIA `cuBLAS` introduces 5–8 µs of launch and tile-setup latency.
* **The Solution:** Custom C++ CUDA kernels (`gemv_bf16_kernel` and `gemv_router_kernel`):
  * **128-Bit Memory Loads:** Casts input and weight pointers to `float4` (16 bytes = 8 `bfloat16` values), saturating Hopper HBM3e bandwidth via single-instruction `LDG.E.128`.
  * **Intra-Warp Reduction:** Collapses 32 values across warp lanes in 5 clock cycles using `__shfl_down_sync` register transfers, completely bypassing Shared Memory.
  * **Inter-Warp Reduction:** Only 8 warp leaders synchronize through shared memory before the block emits the final result.
* **Speedup:** **2.329×** on Router GEMV (4.4 µs vs 10.3 µs) and **1.777×** on Unpooler GEMV.

### 2. In-Place Fused RMSNorm & Gating Kernel
* **The Bottleneck:** Stock attention pipelines read activations from DRAM, normalize them, write to DRAM, read them again, multiply by the gate tensor, and write back (3 separate DRAM round-trips).
* **The Solution:** Custom 512-thread CUDA kernel (`fast_norm_gate_kernel`) executing the root-mean-square calculation, scaling, and elementwise gating directly inside on-chip SRAM/registers in a single pass.
* **Speedup:** **1.090×** across all 16 attention test suites with zero numerical precision error.

### 3. Marlin W4A16 MoE Tile Auto-Tuning
* **The Bottleneck:** Stock SGLang forced `block_size_m = 48` for 128 tokens and `block_size_m = 64` for 256 tokens. Because tokens are sparsely routed across 8 experts, each expert received only ~16 to 28 tokens, resulting in up to 70% of compute wasted on zero-padding dummy tokens.
* **The Solution:** Auto-tuned Marlin's token dimension heuristic to `block_size_m = 32`, eliminating padding overhead across all sparse expert tiles while maintaining full Hopper Tensor Core (WGMMA) saturation.
* **Speedup:** Shaved **152 µs** on 256-token verification passes; boosted MoE composite speed to **1.256×** (Decode-1 to **1.518×**).

### 4. Asynchronous Dual CUDA Stream Overlap
* **The Bottleneck:** Sequential dispatch of Attention and MoE Routing created CPU-GPU sync stalls.
* **The Solution:** Dispatched the Router GEMV on a secondary background stream (`torch.cuda.Stream()`) concurrently with FlashAttention-3 on the primary stream, completely hiding routing latency.
* **Speedup:** Single-token full decode step accelerated by **1.360×** (126.6 µs vs 172.2 µs).

---

## 📂 Repository Structure

```text
open-kernel-challenge/
├── README.md                            # Comprehensive project overview and benchmarks
├── Open_Kernel_Challenge_Masterclass.html # Full 11-page masterclass on GPU architecture & kernels
├── muse_decode_layer/                   # Full Transformer Layer optimization (Rank #1164)
│   ├── submission.py                   # Verified dual-stream submission
│   └── TASK.md                         # Problem specification
├── muse_moe_w4a16/                     # Mixture-of-Experts W4A16 optimization (Rank #1299)
│   ├── submission.py                   # Verified Marlin block-size tuned submission
│   └── TASK.md
├── muse_bf16_proj/                     # Router & Unpooler GEMV optimization (Rank #1586)
│   ├── submission.py                   # Verified 128-bit vectorized CUDA kernels
│   └── TASK.md
├── muse_attention_decode/              # Attention & Norm pipeline optimization (Rank #1361)
│   ├── submission.py                   # Verified in-place RMSNorm-Gate fusion
│   └── TASK.md
└── serving/                            # Serving Track (Opens Nov 10, 8xH200 SGLang Server)
    ├── RULES.md                        # Official judge rules, timeouts, and flags
    ├── custom_kernels.py               # Fused CUDA kernels ready for SGLang integration
    └── patch.diff                      # SGLang patch validated by `okc serve diff`
```

---

## ⚡ Serving Track Strategy (SGLang + DFlash)

The **Serving Track** evaluates end-to-end throughput and time-to-first-token (TTFT) across an **8×H200 multi-GPU cluster** running SGLang with **DFlash Speculative Decoding (draft length 8)**.

* **Target Hotspot:** DFlash verification executes at batch size **$M=8$** on every step.
* **Optimization Vectors:**
  1. Tiled Shared-Memory GEMV for $M=8$ (caching weights in SRAM to prevent L2 cache thrashing).
  2. Marlin `block_size_m = 32` integration into SGLang's `layers/moe/moe_runner/marlin.py`.
  3. Fused QK-Norm + RoPE + FP8 Quantization Triton kernels in `models/muse_spark.py`.
  4. Full compatibility with SGLang CUDA Graphs and Tensor Parallelism (TP=8).

---

## 📖 Deep-Dive Masterclass
For a first-principles breakdown of Hopper H200 architecture, warp-level primitives, and kernel walkthroughs, view the complete guide:
* **[Open_Kernel_Challenge_Masterclass.html](Open_Kernel_Challenge_Masterclass.html)**
