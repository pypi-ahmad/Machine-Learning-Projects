#!/usr/bin/env python3
"""
CUDA Validation Script
======================
Checks PyTorch installation, CUDA availability, GPU details,
and runs a small tensor operation on GPU to verify compute.

Usage:
    python scripts/check_cuda.py
"""

import sys


def main() -> None:
    print("=" * 60)
    print("  CUDA / PyTorch Environment Check")
    print("=" * 60)

    # ── Python ──────────────────────────────────────────────
    print(f"\nPython        : {sys.version}")

    # ── PyTorch ─────────────────────────────────────────────
    try:
        import torch
    except ImportError:
        print("\n[ERROR] PyTorch is not installed.")
        print("        Run: pip install -r requirements.txt")
        sys.exit(1)

    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version  : {torch.version.cuda or 'N/A'}")
    print(f"cuDNN version : {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print(f"Device count  : {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        print("\n[WARN] No CUDA GPU detected. All workloads will run on CPU.")
        print("       If you have an NVIDIA GPU, ensure CUDA drivers are installed.")
        print("=" * 60)
        return

    # ── GPU Details ─────────────────────────────────────────
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_mem / (1024 ** 3)
        print(f"\nGPU {i}:")
        print(f"  Name          : {props.name}")
        print(f"  Compute       : {props.major}.{props.minor}")
        print(f"  Total Memory  : {mem_gb:.1f} GB")
        print(f"  Multi-Processor: {props.multi_processor_count}")

    # ── Smoke Test ──────────────────────────────────────────
    print("\n--- Smoke Test: small tensor matmul on GPU ---")
    device = torch.device("cuda:0")
    a = torch.randn(256, 256, device=device)
    b = torch.randn(256, 256, device=device)
    c = torch.mm(a, b)
    torch.cuda.synchronize()
    print(f"  Result shape  : {c.shape}")
    print(f"  Result device : {c.device}")
    print(f"  Sample value  : {c[0, 0].item():.4f}")
    print("\n[OK] GPU compute verified successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
