# Open Kernel Challenge (OKC) — Frontier LLM Inference Optimization

[![OKC Leaderboard](https://img.shields.io/badge/OKC%20Rank-%231309%20Global%20(3%2C302%20Competitors)-blue?style=for-the-badge&logo=target)](https://openkernelchallenge.dev)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20Hopper%20H200-76B900?style=for-the-badge&logo=nvidia)](https://www.nvidia.com/en-us/data-center/h200/)
[![Challenge](https://img.shields.io/badge/Sponsors-Meta%20%C3%97%20Paradigm%20%C3%97%20Andromeda-6366f1?style=for-the-badge)](#)
[![Status](https://img.shields.io/badge/Verified%20Solutions-4%2F4%20Decode%20Problems-success?style=for-the-badge)](#)

High-performance GPU kernel engineering and layer-level inference acceleration for **Muse Spark Open (816B Parameter Sparse Mixture-of-Experts shapes)** on **NVIDIA Hopper H200** GPUs.

---

## 🏆 Official Leaderboard Standings & Benchmark Results

The competition evaluates submissions on NVIDIA Hopper H200 GPUs against production SGLang and cuBLAS baselines.
Each problem reports an **official composite speedup** across its entire test suite, alongside **individual sub-case measurements** from the test runner:

### Official Composite Leaderboard Scores
| Problem | Official Rank | Submissions | Baseline Time | Optimized Time | Official Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`muse_moe_w4a16`** | **#1299** | 1,426 | 13.45 ms | **10.71 ms** | **1.256×** |
| **`muse_decode_layer`** | **#1164** | 1,300 | 22.62 ms | **18.95 ms** | **1.193×** |
| **`muse_bf16_proj`** | **#1586** | 1,747 | 4.83 ms | **4.36 ms** | **1.108×** |
| **`muse_attention_decode`** | **#1361** | 1,380 | 3.77 ms | **3.46 ms** | **1.090×** |
| **OVERALL LEADERBOARD** | **#1309** | **3,302** | — | — | **4 / 4 Solved** |

### Individual Sub-Case Latency Highlights (From Judge Test Logs)
| Sub-Case | Problem Context | Baseline Latency | Optimized Latency | Measured Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **Router GEMV** | `muse_bf16_proj` (Decode 1, M=1) | 10.3 µs | **4.4 µs** | **2.329×** |
| **Unpooler GEMV** | `muse_bf16_proj` (Decode 1, M=1) | 7.0 µs | **4.0 µs** | **1.777×** |
| **Decode 1 MoE** | `muse_moe_w4a16` (Single token) | 77.1 µs | **50.8 µs** | **1.518×** |
| **Verify 32 MoE** | `muse_moe_w4a16` (256 tokens) | 632.0 µs | **480.0 µs** | **1.317×** (−152 µs) |
| **Single-Token Decode** | `muse_decode_layer` (Sliding window)| 172.2 µs | **126.6 µs** | **1.360×** |
| **Attention Pipeline** | `muse_attention_decode` (Composite) | 3.77 ms | **3.46 ms** | **1.090×** |

---

## 🚀 Key Engineering & Algorithmic Optimizations

### 1. Vectorized 128-Bit GEMV with Register-Level Warp Shuffles
* **The Bottleneck:** For single-token decode ($M=1$), NVIDIA `cuBLAS` introduces 5–8 µs of launch and tile-setup latency.
* **The Solution:** Custom C++ CUDA kernels (`gemv_bf16_kernel` and `gemv_router_kernel`):
  * **128-Bit Memory Loads:** Casts input and weight pointers to `float4` (16 bytes = 8 `bfloat16` values), saturating Hopper HBM3e bandwidth via single-instruction `LDG.E.128`.
  * **Intra-Warp Reduction:** Collapses 32 values across warp lanes in 5 clock cycles using `__shfl_down_sync` register transfers, completely bypassing Shared Memory.
  * **Inter-Warp Reduction:** Only 8 warp leaders synchronize through shared memory before the block emits the final result.
* **Measured Speedup:** **2.329×** on single-token Router GEMV (4.4 µs vs 10.3 µs) and **1.777×** on Unpooler GEMV.

### 2. In-Place Fused RMSNorm & Gating Kernel
* **The Bottleneck:** Stock attention pipelines read activations from DRAM, normalize them, write to DRAM, read them again, multiply by the gate tensor, and write back (3 separate DRAM round-trips).
* **The Solution:** Custom 512-thread CUDA kernel (`fast_norm_gate_kernel`) executing the root-mean-square calculation, scaling, and elementwise gating directly inside on-chip SRAM/registers in a single pass.
* **Measured Speedup:** **1.090×** across all 16 attention test suites with zero numerical precision error.

### 3. Marlin W4A16 MoE Tile Auto-Tuning
* **The Bottleneck:** Stock SGLang forced `block_size_m = 48` for 128 tokens and `block_size_m = 64` for 256 tokens. Because tokens are sparsely routed across 8 experts, each expert received only ~16 to 28 tokens, resulting in up to 70% of compute wasted on zero-padding dummy tokens.
* **The Solution:** Auto-tuned Marlin's token dimension heuristic to `block_size_m = 32`, eliminating padding overhead across all sparse expert tiles while maintaining full Hopper Tensor Core (WGMMA) saturation.
* **Measured Speedup:** Shaved **152 µs** on 256-token verification passes; boosted MoE composite speed to **1.256×** (Decode-1 to **1.518×**).

### 4. Asynchronous Dual CUDA Stream Overlap
* **The Bottleneck:** Sequential dispatch of Attention and MoE Routing created CPU-GPU sync stalls.
* **The Solution:** Dispatched the Router GEMV on a secondary background stream (`torch.cuda.Stream()`) concurrently with FlashAttention-3 on the primary stream, completely hiding routing latency.
* **Measured Speedup:** Single-token full decode step accelerated by **1.360×** (126.6 µs vs 172.2 µs).

---

## 📂 Repository Structure

```text
open-kernel-challenge/
├── README.md                            # Project overview and official benchmark standings
├── muse_decode_layer/                   # Full Transformer Layer optimization (Rank #1164)
│   ├── submission.py                   # Verified dual-stream submission (1.193x composite)
│   └── TASK.md                         # Problem specification
├── muse_moe_w4a16/                     # Mixture-of-Experts W4A16 optimization (Rank #1299)
│   ├── submission.py                   # Verified Marlin block-size tuned submission (1.256x composite)
│   └── TASK.md
├── muse_bf16_proj/                     # Router & Unpooler GEMV optimization (Rank #1586)
│   ├── submission.py                   # Verified 128-bit vectorized CUDA kernels (1.108x composite)
│   └── TASK.md
└── muse_attention_decode/              # Attention & Norm pipeline optimization (Rank #1361)
    ├── submission.py                   # Verified in-place RMSNorm-Gate fusion (1.090x composite)
    └── TASK.md
```
