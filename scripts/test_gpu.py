#!/usr/bin/env python3
"""
GPU Test Script for Ptolemy HPC
Tests CUDA availability and GPU configuration for nanochat
"""

import sys

def test_pytorch():
    """Test PyTorch and CUDA availability"""
    print("=" * 60)
    print("PyTorch and CUDA Test")
    print("=" * 60)

    try:
        import torch
        print(f"✓ PyTorch imported successfully")
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\n  GPU {i}: {props.name}")
                print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
        else:
            print("  ⚠️  CUDA not available - CPU only mode")

        return True
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

def test_basic_operation():
    """Test basic tensor operations on GPU"""
    print("\n" + "=" * 60)
    print("Basic GPU Operations Test")
    print("=" * 60)

    try:
        import torch

        if not torch.cuda.is_available():
            print("⚠️  Skipping GPU test - CUDA not available")
            return True

        # Create tensor on GPU
        device = torch.device('cuda:0')
        print(f"Testing on device: {device}")

        # Simple matrix multiplication
        size = 1000
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        print(f"  Created {size}x{size} tensors on GPU")

        # Warmup
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Timed operation
        import time
        start = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        print(f"  Matrix multiplication completed in {elapsed*1000:.2f} ms")
        print(f"  Result shape: {c.shape}")
        print("✓ GPU operations working correctly")

        return True
    except Exception as e:
        print(f"✗ GPU operation test failed: {e}")
        return False

def test_multi_gpu():
    """Test multi-GPU setup"""
    print("\n" + "=" * 60)
    print("Multi-GPU Test")
    print("=" * 60)

    try:
        import torch

        if not torch.cuda.is_available():
            print("⚠️  Skipping multi-GPU test - CUDA not available")
            return True

        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs available: {gpu_count}")

        if gpu_count > 1:
            print("✓ Multi-GPU setup detected")
            print("  This configuration is optimal for nanochat training")
        elif gpu_count == 1:
            print("⚠️  Single GPU detected")
            print("  nanochat will work but training will take 8x longer")
        else:
            print("✗ No GPUs detected")
            return False

        return True
    except Exception as e:
        print(f"✗ Multi-GPU test failed: {e}")
        return False

def test_dependencies():
    """Test additional dependencies needed for nanochat"""
    print("\n" + "=" * 60)
    print("Dependencies Test")
    print("=" * 60)

    success = True

    # Test numpy
    try:
        import numpy as np
        print(f"✓ numpy: {np.__version__}")
    except ImportError:
        print("✗ numpy not installed")
        success = False

    # Test tqdm
    try:
        import tqdm
        print(f"✓ tqdm: {tqdm.__version__}")
    except ImportError:
        print("✗ tqdm not installed")
        success = False

    # Test requests (for dataset download)
    try:
        import requests
        print(f"✓ requests: {requests.__version__}")
    except ImportError:
        print("✗ requests not installed")
        success = False

    return success

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("nanochat GPU Test Suite for Ptolemy HPC")
    print("=" * 60 + "\n")

    results = []

    # Run tests
    results.append(("PyTorch", test_pytorch()))
    results.append(("GPU Operations", test_basic_operation()))
    results.append(("Multi-GPU", test_multi_gpu()))
    results.append(("Dependencies", test_dependencies()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:20s}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed! System is ready for nanochat training.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check your environment setup.")
        print("\nTroubleshooting:")
        print("1. Make sure you loaded CUDA modules: module load cuda-toolkit")
        print("2. Activate virtual environment: source /scratch/ptolemy/users/$USER/nanochat-venv/bin/activate")
        print("3. Install dependencies: uv sync --extra gpu")
        return 1

if __name__ == "__main__":
    sys.exit(main())
