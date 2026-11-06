#!/usr/bin/env python3
"""Unit test for serving/tiled_m8_gemv.py interface and mathematical correctness."""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add repository root and serving/ to sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
serving_dir = os.path.join(repo_root, "serving")
if serving_dir not in sys.path:
    sys.path.insert(0, serving_dir)

import torch
from tiled_m8_gemv import fast_tiled_m8_gemv

def test_tiled_m8_interface():
    print("=" * 65)
    print("🧪 TESTING TILED M=8 GEMV INTERFACE & DISPATCHER")
    print("=" * 65)

    device = torch.device("cpu")
    dtype = torch.bfloat16
    torch.manual_seed(1234)

    # 1. Test Decode 1 Router (M=1, K=8192, N=256)
    x1 = torch.randn((1, 8192), dtype=dtype, device=device)
    W1 = torch.randn((256, 8192), dtype=dtype, device=device)
    y1_ref = torch.mm(x1.float(), W1.float().t()).to(dtype)
    y1_out = fast_tiled_m8_gemv(x1, W1)
    diff1 = torch.norm(y1_out.float() - y1_ref.float()).item()
    print(f"Test 1: M=1 Decode Router        | Shape: {list(y1_out.shape)} | Diff: {diff1:.2e} ✅")
    assert diff1 == 0.0, "M=1 mismatch"

    # 2. Test DFlash Speculative Verify 1 Req (M=8, K=8192, N=256)
    x8 = torch.randn((8, 8192), dtype=dtype, device=device)
    W8 = torch.randn((256, 8192), dtype=dtype, device=device)
    y8_ref = torch.mm(x8.float(), W8.float().t()).to(dtype)
    y8_out = fast_tiled_m8_gemv(x8, W8)
    diff8 = torch.norm(y8_out.float() - y8_ref.float()).item()
    print(f"Test 2: M=8 DFlash Verify Router | Shape: {list(y8_out.shape)} | Diff: {diff8:.2e} ✅")
    assert diff8 == 0.0, "M=8 mismatch"

    # 3. Test Fused Split-BF16 Router (M=8, K=8192, N=256)
    W_fp32 = torch.randn((256, 8192), dtype=torch.float32, device=device)
    W_hi = W_fp32.to(dtype)
    W_lo = (W_fp32 - W_hi.float()).to(dtype)
    y_split_ref = torch.mm(x8.float(), W_hi.float().t()) + torch.mm(x8.float(), W_lo.float().t())
    y_split_out = fast_tiled_m8_gemv(x8, W_hi, W_lo=W_lo)
    diff_split = torch.norm(y_split_out - y_split_ref).item()
    print(f"Test 3: M=8 Fused Split2 Router  | Shape: {list(y_split_out.shape)} | Diff: {diff_split:.2e} ✅")
    assert diff_split == 0.0, "Fused split mismatch"

    # 4. Test Pre-allocated buffer reuse
    prealloc = torch.empty((8, 256), dtype=dtype, device=device)
    fast_tiled_m8_gemv(x8, W8, out=prealloc)
    diff_prealloc = torch.norm(prealloc.float() - y8_ref.float()).item()
    print(f"Test 4: Buffer Reuse In-Place    | Shape: {list(prealloc.shape)} | Diff: {diff_prealloc:.2e} ✅")
    assert diff_prealloc == 0.0, "Pre-allocated buffer mismatch"

    print("=" * 65)
    print("🎉 ALL INTERFACE & PRECISION TESTS PASSED!")
    print("=" * 65)

if __name__ == "__main__":
    test_tiled_m8_interface()
