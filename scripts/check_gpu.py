#!/usr/bin/env python3
"""
GPU Validation Script
=====================
Verifies CUDA availability, prints GPU info, and runs a small tensor operation.

Usage:
    python scripts/check_gpu.py
"""

import sys


def main():
    print("=" * 60)
    print("  GPU VALIDATION CHECK  (PyTorch + CUDA 13.0)")
    print("=" * 60)

    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch is not installed.")
        print("  Install with:")
        print("    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130")
        sys.exit(1)

    print(f"\nPyTorch version : {torch.__version__}")
    print(f"CUDA built      : {torch.version.cuda or 'N/A (CPU build)'}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n[WARNING] CUDA is NOT available. Training will use CPU only.")
        print("  Possible reasons:")
        print("    - No NVIDIA GPU detected")
        print("    - CUDA drivers not installed or too old")
        print("    - PyTorch CPU-only build installed")
        print("  Fix:")
        print("    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130")
        sys.exit(0)

    # GPU information
    device_count = torch.cuda.device_count()
    print(f"GPU count       : {device_count}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Compute capability : {props.major}.{props.minor}")
        print(f"    Total memory       : {total_mem / 1024**3:.1f} GB")
        print(f"    Free memory        : {free_mem / 1024**3:.1f} GB")
        print(f"    Multi-processors   : {props.multi_processor_count}")

    print(f"\nCUDA runtime    : {torch.version.cuda}")
    print(f"cuDNN version   : {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled   : {torch.backends.cudnn.enabled}")
    print(f"TF32 matmul     : {torch.backends.cuda.matmul.allow_tf32}")
    print(f"TF32 cuDNN      : {torch.backends.cudnn.allow_tf32}")

    # Quick CUDA tensor operation
    print("\n--- Tensor Benchmark (1000x1000 matmul) ---")
    device = torch.device("cuda:0")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)

    # Warm-up
    for _ in range(10):
        _ = torch.mm(a, b)
    torch.cuda.synchronize()

    import time
    start = time.perf_counter()
    for _ in range(100):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"  100x matmul (1000x1000) : {elapsed:.4f}s  ({elapsed/100*1000:.2f}ms/op)")
    print(f"  Result device           : {c.device}")

    # AMP check
    print("\n--- Mixed Precision (AMP) Check ---")
    try:
        with torch.amp.autocast("cuda"):
            d = torch.mm(a.half(), b.half())
        print(f"  AMP autocast works      : True (dtype={d.dtype})")
    except Exception as exc:
        print(f"  AMP autocast works      : False ({exc})")

    print("\n[OK] GPU is working correctly!")
    print("=" * 60)


if __name__ == "__main__":
    main()
